"""Loader for combined GGUF files with multiple components."""

from typing import Dict, Tuple

import torch

from ..utils import is_gguf_available, logging


if is_gguf_available():
    import gguf

    from ..quantizers.gguf.utils import GGUFParameter


logger = logging.get_logger(__name__)


class CombinedGGUFLoader:
    """
    Loads combined GGUF files containing multiple model components and splits
    them into separate state dicts.

    Combined GGUF files (like Flux2 with Ministral3) contain tensors for multiple
    components with different prefixes:
    - Transformer: `transformer.*`
    - VAE: `vae.*`
    - Text Encoder: `blk.*`, `v.blk.*`, `mm.*`, `token_embd.*`, `text_proj.*`

    Example:
        ```python
        loader = CombinedGGUFLoader("path/to/combined.gguf")
        state_dicts = loader.load()

        # state_dicts["transformer"] - transformer weights
        # state_dicts["vae"] - VAE weights
        # state_dicts["text_encoder"] - text encoder weights
        ```
    """

    COMPONENT_PREFIXES = {
        "transformer": ["transformer."],
        "vae": ["vae."],
        "audio_vae": ["audio_vae."],
        "vocoder": ["vocoder."],
        "connectors": ["connectors."],
        "text_encoder": [
            "text_encoder.",  # Diffusers-style prefix (e.g., text_encoder.model.layers.*)
            "blk.",
            "v.blk.",
            "mm.",
            "token_embd",
            "text_proj",
            "output_norm",
        ],
    }

    def __init__(self, gguf_path: str):
        """
        Initialize the loader.

        Args:
            gguf_path: Path to the combined GGUF file.
        """
        if not is_gguf_available():
            raise ImportError(
                "Loading GGUF files requires the `gguf` package. "
                "Install it with: pip install gguf>=0.10.0"
            )
        self.gguf_path = gguf_path

    def load(self, components: list = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Read GGUF file and return split state dicts.

        Args:
            components: Optional list of components to load (e.g., ["transformer", "vae"]).
                       If None, loads all components. Use this to reduce memory usage by
                       loading one component at a time.

        Returns:
            Dict with keys: "transformer", "vae", "text_encoder", etc.
            Each value is a state dict for that component.
        """
        reader = gguf.GGUFReader(self.gguf_path)

        all_components = ["transformer", "vae", "audio_vae", "vocoder", "connectors", "text_encoder"]
        if components is None:
            components = all_components

        result = {c: {} for c in all_components}

        # Track tensor counts for logging
        counts = {c: 0 for c in all_components}

        for tensor in reader.tensors:
            name = tensor.name

            # Determine component early to skip unwanted tensors
            component, _ = self._get_component_and_key(name)
            if component not in components:
                continue

            quant_type = tensor.tensor_type
            # Use numpy array directly without copy when possible
            # The .copy() is needed because GGUF uses mmap and tensor.data is a view
            data = torch.from_numpy(tensor.data.copy())

            # Get the logical tensor shape from GGUF header (not byte shape)
            # GGUF stores shape in row-major (C) order, need to REVERSE for PyTorch
            # This matches ComfyUI-GGUF: torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            #
            # EXCEPTION: Some 4D tensors (like vision patch_embd) are stored in PyTorch order
            # already. Detect these by checking if first dim is large (out_channels) and
            # last dims are small (spatial kernel size).
            raw_shape = tuple(int(d) for d in tensor.shape)
            if len(raw_shape) == 4:
                # Heuristic: 4D tensors with large first dim and small last dims
                # are likely already in PyTorch order [out_ch, in_ch, H, W]
                if raw_shape[0] > 100 and raw_shape[2] <= 14 and raw_shape[3] <= 14:
                    logical_shape = torch.Size(raw_shape)
                else:
                    logical_shape = torch.Size(tuple(reversed(raw_shape)))
            else:
                logical_shape = torch.Size(tuple(reversed(raw_shape)))

            # Determine if tensor is quantized
            is_quantized = quant_type not in [
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
            ]

            # Create appropriate parameter type
            if is_quantized:
                param = GGUFParameter(data, quant_type=quant_type)
                # Store the correct logical shape for dequantization
                param.tensor_shape = logical_shape
            else:
                # For non-quantized, reshape data to logical shape
                param = data.reshape(logical_shape)

            # Route to correct component
            component, key = self._get_component_and_key(name)

            # Fix VAE conv weight dimension ordering
            # GGUF stores as [H, W, in_ch, out_ch], PyTorch expects [out_ch, in_ch, H, W]
            if component == "vae":
                param = self._fix_vae_tensor(key, param, is_quantized)

            result[component][key] = param
            counts[component] += 1

        logger.info(
            f"Loaded combined GGUF: "
            f"transformer={counts['transformer']}, "
            f"vae={counts['vae']}, "
            f"text_encoder={counts['text_encoder']} tensors"
        )

        return result

    def _fix_vae_tensor(self, key: str, param: torch.Tensor, is_quantized: bool) -> torch.Tensor:
        """
        Fix VAE tensor dimension ordering for Conv3D weights.

        After shape reversal, 5D tensors have shape [D, H, out_ch, in_ch, W].
        PyTorch Conv3D expects [out_ch, in_ch, D, H, W].
        Requires permutation (2, 3, 0, 1, 4).

        4D tensors are already correct after shape reversal.

        Args:
            key: Tensor key name.
            param: Tensor or GGUFParameter.
            is_quantized: Whether tensor is quantized.

        Returns:
            Fixed tensor with correct dimension ordering.
        """
        # Get the shape to check
        if is_quantized:
            shape = getattr(param, 'tensor_shape', None)
            if shape is None:
                return param
        else:
            shape = param.shape

        # Fix scalar tensors stored as [1] shape (e.g., num_batches_tracked)
        if len(shape) == 1 and shape[0] == 1 and "num_batches_tracked" in key:
            if not is_quantized:
                return param.squeeze(0)
            return param

        # Fix 5D Conv3D weights
        # After reversal: [D, H, out_ch, in_ch, W] -> PyTorch: [out_ch, in_ch, D, H, W]
        if len(shape) == 5 and "weight" in key:
            d, h, out_ch, in_ch, w = shape
            # Check if this looks like a conv weight (spatial dims small, channels larger)
            if d <= 7 and h <= 7 and w <= 7 and (in_ch > 7 or out_ch > 7):
                new_shape = torch.Size((out_ch, in_ch, d, h, w))
                if is_quantized:
                    param.tensor_shape = new_shape
                else:
                    # Need to permute data, not just reshape
                    param = param.permute(2, 3, 0, 1, 4).contiguous()
                return param

        # 4D tensors are already correct after shape reversal
        return param

    def _get_component_and_key(self, name: str) -> Tuple[str, str]:
        """
        Determine which component a tensor belongs to and get the key name.

        Args:
            name: Original tensor name from GGUF file.

        Returns:
            Tuple of (component_name, key_name).
        """
        for component, prefixes in self.COMPONENT_PREFIXES.items():
            for prefix in prefixes:
                if name.startswith(prefix):
                    if component in ("transformer", "vae", "audio_vae", "vocoder", "connectors"):
                        # Strip the prefix for these components
                        return component, name[len(prefix):]
                    elif prefix == "text_encoder.":
                        # Diffusers-style text_encoder prefix: strip it
                        return component, name[len(prefix):]
                    else:
                        # GGML-style text encoder keeps full name for conversion function
                        return component, name

        # Unknown tensors go to text_encoder (Ministral3 has many patterns)
        logger.debug(f"Unknown tensor prefix, routing to text_encoder: {name}")
        return "text_encoder", name

    def get_metadata(self) -> Dict[str, str]:
        """
        Get metadata from the GGUF file.

        Returns:
            Dict of metadata key-value pairs.
        """
        reader = gguf.GGUFReader(self.gguf_path)
        metadata = {}
        for field in reader.fields.values():
            if hasattr(field, "parts") and len(field.parts) > 0:
                # Get string value from field
                try:
                    value = str(field.parts[-1])
                    metadata[field.name] = value
                except Exception:
                    pass
        return metadata
