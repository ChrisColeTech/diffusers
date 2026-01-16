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
FP8 quantization utilities for diffusers.

This module provides:
- FP8Parameter: A custom Parameter class that stores FP8 weights with per-tensor scales
- FP8Linear: A Linear layer that dequantizes FP8 weights on-the-fly during forward()
- Helper functions for replacing Linear layers with FP8Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import is_accelerate_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights
else:
    from contextlib import nullcontext as init_empty_weights


logger = logging.get_logger(__name__)


# FP8 dtype constant - E4M3FN is the common FP8 format for weights
FP8_DTYPE = torch.float8_e4m3fn


class FP8Parameter(torch.nn.Parameter):
    """
    A parameter class for FP8 quantized weights with per-tensor scaling.

    FP8 quantization stores weights in float8_e4m3fn format with a per-tensor
    scale factor. During forward pass, weights are dequantized via:
        weight_dequant = weight.to(compute_dtype) * weight_scale

    This preserves the memory benefits of FP8 while allowing computation in
    higher precision (bfloat16/float16).

    Attributes:
        weight_scale: Per-tensor scale factor for dequantization
    """

    def __new__(cls, data, requires_grad=False, weight_scale=None):
        data = data if data is not None else torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)

        # Store scale as a scalar tensor on CPU (will be moved with weight)
        if weight_scale is not None:
            self.weight_scale = weight_scale
        else:
            self.weight_scale = torch.tensor(1.0, dtype=torch.float32)

        return self

    def as_tensor(self):
        """Convert to regular tensor (strips FP8Parameter metadata)."""
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)

    @staticmethod
    def _extract_fp8_attrs(args):
        """Extract FP8 attributes from tensor arguments for operations."""
        for arg in args:
            if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], FP8Parameter):
                return arg[0].weight_scale
            if isinstance(arg, FP8Parameter):
                return arg.weight_scale
        return None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, torch.Tensor):
            weight_scale = cls._extract_fp8_attrs(args)
            if weight_scale is not None:
                new_param = cls(result, weight_scale=weight_scale)
                return new_param
        elif type(result) in (list, tuple):
            weight_scale = cls._extract_fp8_attrs(args)
            if weight_scale is not None:
                wrapped = []
                for x in result:
                    if isinstance(x, torch.Tensor):
                        wrapped.append(cls(x, weight_scale=weight_scale))
                    else:
                        wrapped.append(x)
                return type(result)(wrapped)

        return result


class FP8Linear(nn.Module):
    """
    Linear layer that stores weights in FP8 format with per-tensor scaling.

    During forward pass, weights are dequantized on-the-fly:
        output = F.linear(x, weight.to(x.dtype) * scale, bias)

    This preserves memory savings of FP8 while computing in full precision.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = None,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype or torch.bfloat16

        # Weight stored as FP8
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=FP8_DTYPE)
        )
        # Scale stored as float32
        self.weight_scale = nn.Parameter(
            torch.ones(1, device=device, dtype=torch.float32)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=self.compute_dtype)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly FP8 dequantization."""
        # Move scale to same device as weight
        scale = self.weight_scale.to(device=self.weight.device, dtype=x.dtype)

        # Dequantize: cast to compute dtype then scale
        weight_dequant = self.weight.to(x.dtype) * scale

        return F.linear(x, weight_dequant, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, fp8=True'


def _replace_with_fp8_linear(model, compute_dtype, prefix="", modules_to_not_convert=[]):
    """
    Recursively replace ALL nn.Linear modules with FP8Linear.

    When using FP8QuantizationConfig, we replace unconditionally because:
    - For sharded loading, state_dict is None at preprocess_model time
    - Weight loading will assign FP8 weights + weight_scale correctly
    - Non-FP8 layers work fine (scale=1.0 is a no-op)

    Args:
        model: The model to modify
        compute_dtype: Computation dtype (bfloat16/float16)
        prefix: Current module prefix for state_dict keys
        modules_to_not_convert: List of module names to skip
    """
    for name, module in model.named_children():
        module_prefix = prefix + name + "."
        _replace_with_fp8_linear(module, compute_dtype, module_prefix, modules_to_not_convert)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            with init_empty_weights():
                model._modules[name] = FP8Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                )
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)

    return model


def _dequantize_fp8_and_restore_linear(model, modules_to_not_convert=[]):
    """
    Convert FP8Linear modules back to regular nn.Linear with dequantized weights.

    Useful for saving models in full precision or for operations that don't
    support FP8.
    """
    for name, module in model.named_children():
        if isinstance(module, FP8Linear) and name not in modules_to_not_convert:
            device = module.weight.device

            with init_empty_weights():
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    device=device,
                )

            # Dequantize weight
            scale = module.weight_scale.to(device=device, dtype=torch.bfloat16)
            weight_dequant = module.weight.to(torch.bfloat16) * scale
            new_module.weight = nn.Parameter(weight_dequant)

            if module.bias is not None:
                new_module.bias = module.bias

            new_module.to(device)
            model._modules[name] = new_module

        # Recurse into children
        has_children = list(module.children())
        if has_children:
            _dequantize_fp8_and_restore_linear(module, modules_to_not_convert)

    return model


def dequantize_fp8_tensor(tensor, scale=None):
    """
    Dequantize an FP8 tensor to bfloat16.

    Args:
        tensor: FP8 tensor or FP8Parameter
        scale: Optional scale factor (uses tensor.weight_scale if FP8Parameter)

    Returns:
        Dequantized bfloat16 tensor
    """
    if isinstance(tensor, FP8Parameter):
        scale = tensor.weight_scale

    if scale is None:
        scale = torch.tensor(1.0)

    return tensor.to(torch.bfloat16) * scale.to(torch.bfloat16)
