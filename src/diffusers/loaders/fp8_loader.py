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
Loader for FP8 quantized safetensor files.

This module provides utilities for loading FP8-quantized safetensor files
that contain weight + weight_scale tensor pairs. The loader preserves FP8
dtype and creates FP8Parameter objects that the FP8Quantizer can use.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from safetensors import safe_open

from ..utils import logging


logger = logging.get_logger(__name__)


def load_fp8_safetensors(
    path: Union[str, Path],
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load FP8 safetensor file(s) preserving FP8 dtype and weight scales.

    This function handles both single safetensor files and sharded safetensor
    directories with an index.json file.

    Args:
        path: Path to safetensor file or directory containing sharded files
        device: Device to load tensors to (default: "cpu")

    Returns:
        Dict mapping tensor names to tensors (FP8 weights as FP8Parameter)
    """
    from ..quantizers.fp8.utils import FP8Parameter

    path = Path(path)

    if path.is_dir():
        # Sharded safetensors with index file
        return _load_sharded_fp8_safetensors(path, device)
    else:
        # Single safetensor file
        return _load_single_fp8_safetensor(path, device)


def _load_single_fp8_safetensor(
    path: Path,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load a single FP8 safetensor file."""
    from ..quantizers.fp8.utils import FP8Parameter

    state_dict = {}
    weight_scales = {}

    with safe_open(str(path), framework="pt", device=device) as f:
        # First pass: collect all weight_scale tensors
        for key in f.keys():
            if "weight_scale" in key:
                weight_scales[key] = f.get_tensor(key)

        # Second pass: load weights and pair with scales
        for key in f.keys():
            if "weight_scale" in key:
                continue

            tensor = f.get_tensor(key)

            # Check if this is an FP8 weight with a scale
            scale_key = key.replace(".weight", ".weight_scale")
            if scale_key in weight_scales:
                # Create FP8Parameter with scale
                tensor = FP8Parameter(tensor, weight_scale=weight_scales[scale_key])
                state_dict[key] = tensor
                # Also add the scale separately for the quantizer
                state_dict[scale_key] = weight_scales[scale_key]
            else:
                state_dict[key] = tensor

    return state_dict


def _load_sharded_fp8_safetensors(
    path: Path,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load sharded FP8 safetensors from a directory with index.json."""
    from ..quantizers.fp8.utils import FP8Parameter

    # Find index file
    index_file = path / "diffusion_pytorch_model.safetensors.index.json"
    if not index_file.exists():
        index_file = path / "model.safetensors.index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"No safetensors index file found in {path}")

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Group keys by shard file
    shard_files = {}
    for key, shard_file in weight_map.items():
        if shard_file not in shard_files:
            shard_files[shard_file] = []
        shard_files[shard_file].append(key)

    state_dict = {}
    weight_scales = {}

    # First pass: collect all weight_scale tensors
    for shard_file in shard_files:
        shard_path = path / shard_file
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "weight_scale" in key:
                    weight_scales[key] = f.get_tensor(key)

    # Second pass: load weights and pair with scales
    for shard_file in shard_files:
        shard_path = path / shard_file
        with safe_open(str(shard_path), framework="pt", device=device) as f:
            for key in f.keys():
                if "weight_scale" in key:
                    state_dict[key] = weight_scales[key]
                    continue

                tensor = f.get_tensor(key)

                # Check if this is an FP8 weight with a scale
                scale_key = key.replace(".weight", ".weight_scale")
                if scale_key in weight_scales:
                    # Create FP8Parameter with scale
                    tensor = FP8Parameter(tensor, weight_scale=weight_scales[scale_key])

                state_dict[key] = tensor

    logger.info(f"Loaded {len(state_dict)} tensors from {len(shard_files)} shards, "
                f"including {len(weight_scales)} weight_scale tensors")

    return state_dict


def is_fp8_safetensor(path: Union[str, Path]) -> bool:
    """
    Check if a safetensor file contains FP8 weights.

    Args:
        path: Path to safetensor file or directory

    Returns:
        True if the file contains FP8 weights (float8_e4m3fn or weight_scale)
    """
    path = Path(path)

    if path.is_dir():
        # Check first shard
        index_file = path / "diffusion_pytorch_model.safetensors.index.json"
        if not index_file.exists():
            index_file = path / "model.safetensors.index.json"
        if not index_file.exists():
            return False

        with open(index_file) as f:
            index = json.load(f)

        # Check if any key contains weight_scale
        weight_map = index.get("weight_map", {})
        for key in weight_map:
            if "weight_scale" in key:
                return True

        # Check first shard file for FP8 dtype
        if weight_map:
            first_shard = list(weight_map.values())[0]
            shard_path = path / first_shard
            return _check_file_for_fp8(shard_path)

        return False
    else:
        return _check_file_for_fp8(path)


def _check_file_for_fp8(path: Path) -> bool:
    """Check a single safetensor file for FP8 content."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "weight_scale" in key:
                    return True

            # Check dtype of first tensor
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    return True
                break  # Only check first tensor

        return False
    except Exception:
        return False
