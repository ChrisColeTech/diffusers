# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FP8 Quantizer for diffusers models.

This quantizer handles loading FP8-quantized safetensors weights with per-tensor
scaling. Unlike simple dtype casting, it preserves the FP8 format in memory and
performs dequantization on-the-fly during forward passes.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from ..base import DiffusersQuantizer
from ...utils import get_module_from_name, is_accelerate_available, logging


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


logger = logging.get_logger(__name__)


class FP8Quantizer(DiffusersQuantizer):
    """
    Diffusers quantizer for FP8 (float8_e4m3fn) quantized models.

    This quantizer:
    1. Replaces nn.Linear modules with FP8Linear before weight loading
    2. Loads FP8 weights directly without converting to bfloat16
    3. Loads weight_scale tensors for per-tensor dequantization
    4. Performs on-the-fly dequantization during forward passes
    """

    use_keep_in_fp32_modules = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.compute_dtype = quantization_config.compute_dtype
        self.modules_to_not_convert = quantization_config.modules_to_not_convert or []

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

    def validate_environment(self, *args, **kwargs):
        """Validate that the environment supports FP8."""
        if not is_accelerate_available():
            raise ImportError(
                "Loading FP8 models requires `accelerate` installed: `pip install accelerate>=0.26.0`"
            )

        # Check PyTorch version supports FP8
        try:
            _ = torch.float8_e4m3fn
        except AttributeError:
            raise ImportError(
                "Loading FP8 models requires PyTorch 2.1+ with FP8 support. "
                "Please upgrade PyTorch: pip install torch>=2.1.0"
            )

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        """Adjust max memory for FP8 loading overhead."""
        # FP8 weights are half the size of bf16, but we need some buffer
        max_memory = {key: val * 0.95 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        """FP8 weights should stay as uint8 during loading."""
        # FP8 tensors are stored as raw bytes, similar to quantized formats
        # The actual dtype conversion happens in the loading logic
        return target_dtype

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        """Set default compute dtype if not specified."""
        if torch_dtype is None:
            torch_dtype = self.compute_dtype
        return torch_dtype

    def check_quantized_param_shape(self, param_name, current_param, loaded_param):
        """Check if loaded parameter shape matches expected shape."""
        # For FP8, shapes should match directly
        if hasattr(loaded_param, 'shape'):
            if loaded_param.shape != current_param.shape:
                raise ValueError(
                    f"{param_name} shape mismatch: expected {current_param.shape}, got {loaded_param.shape}"
                )
        return True

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: Union["torch.Tensor", Any],
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """Check if a parameter is FP8 quantized."""
        from .utils import FP8Parameter

        if isinstance(param_value, FP8Parameter):
            return True

        # Check for FP8 dtype
        if hasattr(param_value, 'dtype'):
            if param_value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                return True

        return False

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: Union["torch.Tensor", Any],
        param_name: str,
        target_device: "torch.device",
        state_dict: Optional[Dict[str, Any]] = None,
        unexpected_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Create and assign a quantized parameter to the model.

        For FP8, this handles both the weight and its associated weight_scale.
        """
        from .utils import FP8Parameter

        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name not in module._parameters and tensor_name not in module._buffers:
            raise ValueError(f"{module} does not have a parameter or buffer named {tensor_name}")

        # Handle weight_scale parameter specially
        if tensor_name == "weight_scale":
            if tensor_name in module._parameters:
                module._parameters[tensor_name] = torch.nn.Parameter(
                    param_value.to(target_device), requires_grad=False
                )
            return

        # For FP8 weights, wrap in FP8Parameter if needed
        if isinstance(param_value, FP8Parameter):
            param_to_set = param_value.to(target_device)
        elif hasattr(param_value, 'dtype') and param_value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Check for corresponding weight_scale
            scale_key = param_name.rsplit('.', 1)[0] + '.weight_scale' if '.' in param_name else 'weight_scale'
            weight_scale = state_dict.get(scale_key) if state_dict else None
            param_to_set = FP8Parameter(param_value.to(target_device), weight_scale=weight_scale)
        else:
            param_to_set = param_value.to(target_device)

        if tensor_name in module._parameters:
            module._parameters[tensor_name] = param_to_set
        if tensor_name in module._buffers:
            module._buffers[tensor_name] = param_to_set

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        """
        Process model before loading weights - replace Linear with FP8Linear.

        We replace ALL nn.Linear unconditionally because for sharded loading,
        state_dict is None at this point. The weight loading phase will handle
        assigning FP8 weights and weight_scale tensors correctly.
        """
        from .utils import _replace_with_fp8_linear

        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        self.modules_to_not_convert = [m for m in self.modules_to_not_convert if m is not None]

        _replace_with_fp8_linear(
            model, self.compute_dtype, modules_to_not_convert=self.modules_to_not_convert
        )

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        """Post-processing after weight loading."""
        return model

    @property
    def is_serializable(self):
        """FP8 models can be serialized."""
        return True

    @property
    def is_trainable(self) -> bool:
        """FP8 models are not trainable (quantized weights)."""
        return False

    @property
    def is_compileable(self) -> bool:
        """FP8 models can be compiled with torch.compile."""
        return True

    def _dequantize(self, model):
        """Dequantize FP8 model back to full precision."""
        from .utils import _dequantize_fp8_and_restore_linear

        is_model_on_cpu = model.device.type == "cpu"
        if is_model_on_cpu:
            logger.info("Moving model to accelerator for dequantization...")
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            model.to(device)

        model = _dequantize_fp8_and_restore_linear(model, self.modules_to_not_convert)

        if is_model_on_cpu:
            model.to("cpu")

        return model
