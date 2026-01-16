# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, BaseOutput, logging, scale_lora_layers, unscale_lora_layers
from ..attention import AttentionMixin
from ..controlnets.controlnet import ControlNetConditioningEmbedding, zero_module
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..transformers.transformer_flux2 import (
    Flux2PosEmbed,
    Flux2TimestepGuidanceEmbeddings,
    Flux2Modulation,
    Flux2TransformerBlock,
    Flux2SingleTransformerBlock,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Flux2ControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]
    controlnet_single_block_samples: Tuple[torch.Tensor]


class Flux2ControlNetModel(ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin):
    """
    A ControlNet model for Flux2 that produces residuals to add to the main transformer.

    Mirrors the architecture of Flux2Transformer2DModel but with controlnet projection layers.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        num_layers: int = 4,  # Fewer layers than full transformer
        num_single_layers: int = 10,  # Fewer layers than full transformer
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        num_mode: int = None,
        conditioning_embedding_channels: int = None,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Positional embedding (same as Flux2)
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Time + guidance embedding (same as Flux2)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim, bias=False
        )

        # 3. Modulation (same as Flux2)
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections (same as Flux2)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. ControlNet projection blocks (zero-initialized)
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(len(self.single_transformer_blocks)):
            self.controlnet_single_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))

        # 8. Union mode (optional)
        self.union = num_mode is not None
        if self.union:
            self.controlnet_mode_embedder = nn.Embedding(num_mode, self.inner_dim)

        # 9. ControlNet conditioning input
        if conditioning_embedding_channels is not None:
            self.input_hint_block = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=conditioning_embedding_channels,
                block_out_channels=(16, 16, 16, 16)
            )
            self.controlnet_x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        else:
            self.input_hint_block = None
            self.controlnet_x_embedder = zero_module(nn.Linear(in_channels, self.inner_dim, bias=False))

        self.gradient_checkpointing = False

    @classmethod
    def from_transformer(
        cls,
        transformer: "Flux2Transformer2DModel",
        num_layers: int = 4,
        num_single_layers: int = 10,
        load_weights_from_transformer: bool = True,
    ):
        """Create a ControlNet from an existing Flux2 transformer."""
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_guidance_embed.load_state_dict(transformer.time_guidance_embed.state_dict())
            controlnet.double_stream_modulation_img.load_state_dict(
                transformer.double_stream_modulation_img.state_dict()
            )
            controlnet.double_stream_modulation_txt.load_state_dict(
                transformer.double_stream_modulation_txt.state_dict()
            )
            controlnet.single_stream_modulation.load_state_dict(transformer.single_stream_modulation.state_dict())
            controlnet.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(
                transformer.transformer_blocks.state_dict(), strict=False
            )
            controlnet.single_transformer_blocks.load_state_dict(
                transformer.single_transformer_blocks.state_dict(), strict=False
            )
            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_mode: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Flux2ControlNetOutput]:
        """
        Forward pass for Flux2ControlNet.

        Args:
            hidden_states: Input latents [B, S, C]
            controlnet_cond: Control condition tensor
            controlnet_mode: Mode tensor for union controlnet
            conditioning_scale: Scale factor for controlnet outputs
            encoder_hidden_states: Text embeddings [B, S, D]
            timestep: Denoising timestep
            img_ids: Image position IDs
            txt_ids: Text position IDs
            guidance: Guidance scale embedding
            joint_attention_kwargs: Additional attention kwargs
            return_dict: Whether to return dataclass or tuple

        Returns:
            ControlNet residuals for transformer blocks
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # Embed hidden states
        hidden_states = self.x_embedder(hidden_states)

        # Process control condition
        if self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)
            batch_size, channels, height_pw, width_pw = controlnet_cond.shape
            height = height_pw // self.config.patch_size
            width = width_pw // self.config.patch_size
            controlnet_cond = controlnet_cond.reshape(
                batch_size, channels, height, self.config.patch_size, width, self.config.patch_size
            )
            controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
            controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1)

        # Add control condition to hidden states
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        # Time + guidance embedding
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
            temb = self.time_guidance_embed(timestep, guidance)
        else:
            # No guidance - use timestep only (would need modification for guidance-free)
            temb = self.time_guidance_embed(timestep, timestep)

        # Context embedding
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Handle deprecated 3D ids
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        # Union mode
        if self.union:
            if controlnet_mode is None:
                raise ValueError("`controlnet_mode` cannot be `None` when applying ControlNet-Union")
            controlnet_mode_emb = self.controlnet_mode_embedder(controlnet_mode)
            encoder_hidden_states = torch.cat([controlnet_mode_emb, encoder_hidden_states], dim=1)
            txt_ids = torch.cat([txt_ids[:1], txt_ids], dim=0)

        # Positional embeddings
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # Get modulation parameters
        temb_mod_params_img = self.double_stream_modulation_img(temb)
        temb_mod_params_txt = self.double_stream_modulation_txt(temb)
        temb_mod_params_single = self.single_stream_modulation(temb)

        # Double stream blocks
        block_samples = ()
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            block_samples = block_samples + (hidden_states,)

        # Concatenate for single stream
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Single stream blocks
        single_block_samples = ()
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod_params=temb_mod_params_single,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            single_block_samples = single_block_samples + (hidden_states,)

        # Apply controlnet projection blocks
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        controlnet_single_block_samples = ()
        for single_block_sample, controlnet_block in zip(single_block_samples, self.controlnet_single_blocks):
            single_block_sample = controlnet_block(single_block_sample)
            controlnet_single_block_samples = controlnet_single_block_samples + (single_block_sample,)

        # Apply conditioning scale
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return Flux2ControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )


class Flux2MultiControlNetModel(ModelMixin):
    """
    Wrapper for multiple Flux2ControlNetModel instances.
    """

    def __init__(self, controlnets: List[Flux2ControlNetModel]):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: List[torch.Tensor],
        controlnet_mode: List[torch.Tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Flux2ControlNetOutput, Tuple]:
        """Forward pass for multiple controlnets."""
        control_block_samples = None
        control_single_block_samples = None

        for i, (image, mode, scale, controlnet) in enumerate(
            zip(controlnet_cond, controlnet_mode, conditioning_scale, self.nets)
        ):
            block_samples, single_block_samples = controlnet(
                hidden_states=hidden_states,
                controlnet_cond=image,
                controlnet_mode=mode[:, None] if mode is not None else None,
                conditioning_scale=scale,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )

            if i == 0:
                control_block_samples = block_samples
                control_single_block_samples = single_block_samples
            else:
                if block_samples is not None and control_block_samples is not None:
                    control_block_samples = [
                        control + sample for control, sample in zip(control_block_samples, block_samples)
                    ]
                if single_block_samples is not None and control_single_block_samples is not None:
                    control_single_block_samples = [
                        control + sample for control, sample in zip(control_single_block_samples, single_block_samples)
                    ]

        return control_block_samples, control_single_block_samples
