# Copyright 2025 The HuggingFace Team and City96. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

import inspect
import os
from contextlib import nullcontext

import gguf
import torch
import torch.nn as nn

from ...utils import is_accelerate_available, is_kernels_available


if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module


can_use_cuda_kernels = (
    os.getenv("DIFFUSERS_GGUF_CUDA_KERNELS", "false").lower() in ["1", "true", "yes"]
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 7
)
if can_use_cuda_kernels and is_kernels_available():
    from kernels import get_kernel

    ops = get_kernel("Isotr0py/ggml")
else:
    ops = None

UNQUANTIZED_TYPES = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
STANDARD_QUANT_TYPES = {
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q4_1,
    gguf.GGMLQuantizationType.Q5_0,
    gguf.GGMLQuantizationType.Q5_1,
    gguf.GGMLQuantizationType.Q8_0,
    gguf.GGMLQuantizationType.Q8_1,
}
KQUANT_TYPES = {
    gguf.GGMLQuantizationType.Q2_K,
    gguf.GGMLQuantizationType.Q3_K,
    gguf.GGMLQuantizationType.Q4_K,
    gguf.GGMLQuantizationType.Q5_K,
    gguf.GGMLQuantizationType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    gguf.GGMLQuantizationType.IQ1_M,
    gguf.GGMLQuantizationType.IQ1_S,
    gguf.GGMLQuantizationType.IQ2_XXS,
    gguf.GGMLQuantizationType.IQ2_XS,
    gguf.GGMLQuantizationType.IQ2_S,
    gguf.GGMLQuantizationType.IQ3_XXS,
    gguf.GGMLQuantizationType.IQ3_S,
    gguf.GGMLQuantizationType.IQ4_XS,
    gguf.GGMLQuantizationType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T

    # TODO(Isotr0py): GGUF's MMQ and MMVQ implementation are designed for
    # contiguous batching and inefficient with diffusers' batching,
    # so we disabled it now.

    # elif qweight_type in MMVQ_QUANT_TYPES:
    #     y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # elif qweight_type in MMQ_QUANT_TYPES:
    #     y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])

    # If there is no available MMQ kernel, fallback to dequantize
    if qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape)
        y = x @ weight.to(x.dtype).T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = gguf.GGMLQuantizationType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y.as_tensor()


# Copied from diffusers.quantizers.bitsandbytes.utils._create_accelerate_new_hook
def _create_accelerate_new_hook(old_hook):
    r"""
    Creates a new hook based on the old hook. Use it only if you know what you are doing ! This method is a copy of:
    https://github.com/huggingface/peft/blob/748f7968f3a31ec06a1c2b0328993319ad9a150a/src/peft/utils/other.py#L245 with
    some changes
    """
    old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
    old_hook_attr = old_hook.__dict__
    filtered_old_hook_attr = {}
    old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
    for k in old_hook_attr.keys():
        if k in old_hook_init_signature.parameters:
            filtered_old_hook_attr[k] = old_hook_attr[k]
    new_hook = old_hook_cls(**filtered_old_hook_attr)
    return new_hook


def _replace_with_gguf_linear(model, compute_dtype, state_dict, prefix="", modules_to_not_convert=[]):
    def _should_convert_to_gguf(state_dict, prefix):
        weight_key = prefix + "weight"
        return weight_key in state_dict and isinstance(state_dict[weight_key], GGUFParameter)

    has_children = list(model.children())
    if not has_children:
        return

    for name, module in model.named_children():
        module_prefix = prefix + name + "."
        _replace_with_gguf_linear(module, compute_dtype, state_dict, module_prefix, modules_to_not_convert)

        if (
            isinstance(module, nn.Linear)
            and _should_convert_to_gguf(state_dict, module_prefix)
            and name not in modules_to_not_convert
        ):
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                model._modules[name] = GGUFLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                )
            model._modules[name].source_cls = type(module)
            # Force requires_grad to False to avoid unexpected errors
            model._modules[name].requires_grad_(False)

    return model


def _dequantize_gguf_and_restore_linear(model, modules_to_not_convert=[]):
    for name, module in model.named_children():
        if isinstance(module, GGUFLinear) and name not in modules_to_not_convert:
            device = module.weight.device
            bias = getattr(module, "bias", None)

            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    device=device,
                )
            new_module.weight = nn.Parameter(dequantize_gguf_tensor(module.weight))
            if bias is not None:
                new_module.bias = bias

            # Create a new hook and attach it in case we use accelerate
            if hasattr(module, "_hf_hook"):
                old_hook = module._hf_hook
                new_hook = _create_accelerate_new_hook(old_hook)

                remove_hook_from_module(module)
                add_hook_to_module(new_module, new_hook)

            new_module.to(device)
            model._modules[name] = new_module

        has_children = list(module.children())
        if has_children:
            _dequantize_gguf_and_restore_linear(module, modules_to_not_convert)

    return model


# dequantize operations based on torch ports of GGUF dequantize_functions
# from City96
# more info: https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py


QK_K = 256
K_SCALE_SIZE = 12


def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = ql | (qh << 4)
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    (
        ql,
        qh,
        scales,
        d,
    ) = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(0, 8, device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 8, 1)
    )
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 2, 1)
    )
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.arange(0, 8, device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 8, 1)
    )
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# this part from calcuis (gguf.org)
# more info: https://github.com/calcuis/gguf-connector/blob/main/src/gguf_connector/quant2c.py


def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32,
        device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs


def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32,
        device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor(
        [2 * i for i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8
    ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)) & 0x03
    scales = (scales_l | (scales_h << 4)) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2], 16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)


# IQ2_XXS lookup tables from llama.cpp
IQ2_XXS_KSIGNS = bytes(
    b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
    b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
    b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
    b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
    b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
    b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
    b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
    b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
)

# iq2xxs_grid as ASCII hex string (256 entries of 8 values each)
# Each byte contains 4 2-bit indices that map to grid_map values
IQ2_XXS_GRID_HEX = (
    b"00000200050008000a00110014002000220028002a0041004400500058006100"
    b"6400800082008a00a20001010401100115014001840198010002020222028202"
    b"010404041004210424044004420448046004810484049004a404000502050805"
    b"200546056905800591050906100640068406a406000805080808140828084108"
    b"440850085208880804094009020a140a01100410101021104010601084109010"
    b"951000110811201150115a118011241245120014081420142514491480141815"
    b"6215001616160118041810184018811800190519a019511a002002200a204420"
    b"6120802082202921482100220222012404241024402456240025412564259026"
    b"082820289428442a014004401040184021402440404048405640604081408440"
    b"9040004120416141804185410142104248425642684200440844204480449944"
    b"124524450046014804481048404845480049584961498249454a904a00500850"
    b"1150195020508050885004514251a4519152905492540a550156545600581158"
    b"195864584059085a046010604060686000615561186260620064056410651265"
    b"84654268008002800a8041808280048118814081118201840484108415844084"
    b"608400854685948509864086608602880489118a0490109024904090a1901691"
    b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)
IQ2_XXS_GRID_MAP = (0x08, 0x19, 0x2b)  # 8, 25, 43


def _build_iq2_xxs_grid(device):
    """Build the IQ2_XXS dequantization grid (256 entries x 8 values each).

    Follows llama.cpp's init_grid() logic:
    1. Decode ASCII hex chars to bytes
    2. Unpack each byte to 4 2-bit values
    3. Map 2-bit indices to actual values (8, 25, 43)
    """
    import numpy as np

    # Decode hex chars: pairs of ASCII hex -> byte values
    grid_bytes = np.frombuffer(IQ2_XXS_GRID_HEX, dtype=np.uint8)
    grid_bytes = grid_bytes.reshape(-1, 2)
    # Convert ASCII to nibbles: '0'-'9' -> 0-9, 'a'-'f' -> 10-15
    grid_bytes = np.where(grid_bytes > 0x40, grid_bytes + 9, grid_bytes) & 0x0F
    # Combine nibbles to bytes
    grid_bytes = (grid_bytes[:, 0] << 4) | grid_bytes[:, 1]

    # Unpack each byte to 4 2-bit values (shifts: 0, 2, 4, 6)
    grid_bytes = grid_bytes.reshape(-1, 1)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    grid_indices = (grid_bytes >> shifts) & 0x03

    # Map 2-bit indices to actual values
    grid_map = np.array(IQ2_XXS_GRID_MAP, dtype=np.float32)
    grid_values = grid_map[grid_indices.flatten()]

    # Reshape to (256, 8)
    grid = grid_values.reshape(256, 8)
    return torch.from_numpy(grid).to(device)


_IQ2_XXS_GRID_CACHE = {}


def _get_iq2_xxs_grid(device):
    """Get cached IQ2_XXS grid for device."""
    if device not in _IQ2_XXS_GRID_CACHE:
        _IQ2_XXS_GRID_CACHE[device] = _build_iq2_xxs_grid(device)
    return _IQ2_XXS_GRID_CACHE[device]


def dequantize_blocks_IQ2_XXS(blocks, block_size, type_size, dtype=None):
    """
    Dequantize IQ2_XXS quantized blocks.

    IQ2_XXS format (block_size=256, type_size=66):
    - 2 bytes: fp16 scale (d)
    - 64 bytes: quantized values (8 groups of 8 bytes each = 8 pairs of uint32)

    Each group (8 bytes = 2 uint32):
    - qs[0]: 4 uint8 indices into the 256-entry grid (each entry has 8 values)
    - qs[1]: top 4 bits = sub-scale, bits 0-27 = 4 sign indices (7 bits each)
    """
    n_blocks = blocks.shape[0]
    device = blocks.device
    target_dtype = dtype if dtype else torch.float32

    # Split: 2 bytes scale, 64 bytes quantized data
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(target_dtype)  # [n_blocks, 1]

    # View qs as uint32 pairs: 64 bytes / 4 bytes per uint32 = 16 uint32s = 8 pairs
    # First reshape to bytes, then convert to uint32
    qs_bytes = qs.reshape(n_blocks, 8, 8)  # 8 groups of 8 bytes each

    # Build uint32 values manually (little-endian)
    qs_u32 = qs_bytes.reshape(n_blocks, 8, 2, 4)  # [n_blocks, 8, 2, 4 bytes]
    qs_u32 = (
        qs_u32[..., 0].to(torch.int64)
        | (qs_u32[..., 1].to(torch.int64) << 8)
        | (qs_u32[..., 2].to(torch.int64) << 16)
        | (qs_u32[..., 3].to(torch.int64) << 24)
    )  # [n_blocks, 8, 2]

    # Extract sub-block scale factor from top 4 bits of qs[..., 1]
    # db = d * (0.5 + (qs[..., 1] >> 28)) * 0.25
    scale_bits = ((qs_u32[..., 1] >> 28) & 0x0F).to(target_dtype)  # [n_blocks, 8]
    db = d * (0.5 + scale_bits) * 0.25  # [n_blocks, 8]
    db = db.reshape(n_blocks, 8, 1, 1)  # [n_blocks, 8, 1, 1]

    # Extract sign indices from qs[..., 1] at bit positions 0, 7, 14, 21 (7 bits each)
    # Each 7-bit value indexes into ksigns table to get 8 sign bits
    ksigns = torch.tensor(list(IQ2_XXS_KSIGNS), dtype=torch.int64, device=device)
    qs1 = qs_u32[..., 1]  # [n_blocks, 8]

    # Extract 4 sign indices per group
    sign_idx_0 = (qs1 >> 0) & 0x7F
    sign_idx_1 = (qs1 >> 7) & 0x7F
    sign_idx_2 = (qs1 >> 14) & 0x7F
    sign_idx_3 = (qs1 >> 21) & 0x7F
    sign_indices = torch.stack([sign_idx_0, sign_idx_1, sign_idx_2, sign_idx_3], dim=-1)  # [n_blocks, 8, 4]

    # Look up sign bytes from ksigns table
    sign_bytes = ksigns[sign_indices.reshape(-1)].reshape(n_blocks, 8, 4)  # [n_blocks, 8, 4]

    # Unpack 8 sign bits from each byte
    bit_shifts = torch.arange(8, dtype=torch.int64, device=device)
    signs = (sign_bytes.unsqueeze(-1) >> bit_shifts) & 1  # [n_blocks, 8, 4, 8]
    signs = torch.where(signs == 0, 1.0, -1.0).to(target_dtype)

    # Get grid indices from qs[..., 0] viewed as 4 uint8 values
    # Each uint8 indexes a row of 8 values in the grid
    grid = _get_iq2_xxs_grid(device)  # [256, 8]
    qs0_bytes = qs_bytes[..., :4]  # First 4 bytes of each group = qs[..., 0] as uint8
    grid_indices = qs0_bytes.reshape(n_blocks, 8, 4).to(torch.long)  # [n_blocks, 8, 4]

    # Look up grid values
    grid_values = grid[grid_indices.reshape(-1)].reshape(n_blocks, 8, 4, 8)  # [n_blocks, 8, 4, 8]
    grid_values = grid_values.to(target_dtype)

    # Final result: db * grid * signs
    result = db * grid_values * signs  # [n_blocks, 8, 4, 8]
    return result.reshape(n_blocks, -1)  # [n_blocks, 256]


# IQ2_S grid (1024 entries x 8 values each)
IQ2_S_GRID_HEX = (
    b"00000200050008000a0011001400160019002000220025002800410044004600"
    b"490050005200550058006100640066006900800082008500880091009400a000"
    b"a500aa0001010401060109011001120115011801210124014001420145014801"
    b"510154015601590160016501680181018401900192019501a101a40100020202"
    b"050208021102140220022a02410244024602490250025502800285028a029402"
    b"a202010404040604090410041204150418042104240426042904400442044504"
    b"48044a0451045404560459046004620465048104840486048904900495049804"
    b"a104a40400050205050508050a05110514051605190520052505280541054405"
    b"46054905500552055505580561056405800582058505880591059405a0050106"
    b"0406060609061006150640064506480651065406600681068406900600080208"
    b"050808081108140816081908200825082a084108440846084908500852085508"
    b"580861086408800885089408aa08010904091009120915091809210940094509"
    b"480951095409600981099009000a110a140a220a280a2a0a500a990a01100410"
    b"0610091010101210151018102110241026104010421045104810511054105610"
    b"59106010621065106810811084108610901095109810a110a410001102110511"
    b"08110a1111111411161119112011221125112811411144114611491150115211"
    b"5511581161116411801182118511881191119411011204120912101215122112"
    b"2412401245125112541281128412901200140214051408141114141416141914"
    b"2014251428144114441446144914501452145514581461146414801482148514"
    b"881491149414a014011504150615091510151215151518152115241540154215"
    b"4515481551155415601581158415901500160516081611161416201641164416"
    b"50168016aa160118041806180918101815181818211840184218451848185118"
    b"541860188118841800190219051908191119141920194119441950196919a219"
    b"041a101a401a561a00200220052008201120142016201920202025202a204120"
    b"4420502052205520642080208a209420aa200121042110211221152121214021"
    b"4221452151215421602181218421902100220a22222228222a22442250228822"
    b"8a22a82201240424062409241024152418242124242440244224452448245124"
    b"5424602481248424902400250525082511251425202541254425502566258025"
    b"0126042610264026592600280528112814284128442850288a28aa2801290429"
    b"102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
    b"21402440264040404240454048404a4051405440564059406040624065408140"
    b"8440904095409840a140a4400041024105410841114114411641194120412241"
    b"2541414144414641494150415241554158416141644180418241854188419141"
    b"9441a04101420442104212421542184224424042454248425142544260428142"
    b"844200440244054408440a441144144416441944204422442544284441444444"
    b"46444944504452445544584461446444804482448544884491449444a0440145"
    b"0445064509451045124515451845214524454045424545454845514554456045"
    b"6a4581458445904500460246054608461146144620464146444650468046a546"
    b"0148044809481048124815481848214824484048424845484848514854486048"
    b"84489048004902490549084911491449204941494449504980499649014a044a"
    b"104a404a00500250055008501150145016501950205022502550285041504450"
    b"4650495050505250555058506150645080508250855088509150945001510451"
    b"0651095110511251155118512151245140514251455148515151545160518151"
    b"8451905100520552085211521452205241524452505269528052015404540654"
    b"0954105412541554185421542454405442544554485451545454605481548454"
    b"9054005502550555085511551455205541554455505580550156045610562656"
    b"405600580258055808581158145820584158445850585a588058015904591059"
    b"4059005a195a855aa85a01600460066010601260156018602160246040604560"
    b"4860516054606060846090600061026105610861116114612061416144615061"
    b"806199610462106240625662a162006405640864116414642064416444645064"
    b"806401650465106540654a656865926500669466016804681068656898680069"
    b"2a69426aa16a0080028005800880118014801980208025804180448050805280"
    b"5580588061808080858091809480018104810981108112811581188121812481"
    b"408142814581488151815481818184819081a981008205820a82118214824182"
    b"4482508201840484068409841084128415841884218440844284458448845184"
    b"5484608481848484908400850285058508851185148520854185448550858085"
    b"8a85018604861086298640860088058811881488418844885088a28801890489"
    b"40896589228a588a5a8a828aa28a019004900990109012901590189024904090"
    b"4290459048905190549060908190849090900091059111911491419144915091"
    b"5a910192049210924092a6920094029405940894119414942094419444945094"
    b"8094969401950495109540959895a19500964696649601980498109826984098"
    b"a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
    b"40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4"
    b"a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
)
IQ2_S_GRID_MAP = (0x08, 0x19, 0x2b)  # 8, 25, 43

_IQ2_S_GRID_CACHE = {}


def _build_iq2_s_grid(device):
    """Build IQ2_S grid (1024 entries x 8 values)."""
    import numpy as np
    grid_bytes = np.frombuffer(IQ2_S_GRID_HEX, dtype=np.uint8).reshape(-1, 2)
    grid_bytes = np.where(grid_bytes > 0x40, grid_bytes + 9, grid_bytes) & 0x0F
    grid_bytes = (grid_bytes[:, 0] << 4) | grid_bytes[:, 1]
    grid_bytes = grid_bytes.reshape(-1, 1)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    grid_indices = (grid_bytes >> shifts) & 0x03
    grid_map = np.array(IQ2_S_GRID_MAP, dtype=np.float32)
    grid_values = grid_map[grid_indices.flatten()]
    return torch.from_numpy(grid_values.reshape(1024, 8)).to(device)


def _get_iq2_s_grid(device):
    if device not in _IQ2_S_GRID_CACHE:
        _IQ2_S_GRID_CACHE[device] = _build_iq2_s_grid(device)
    return _IQ2_S_GRID_CACHE[device]


def dequantize_blocks_IQ2_S(blocks, block_size, type_size, dtype=None):
    """
    Dequantize IQ2_S blocks.

    IQ2_S format (block_size=256, type_size=82):
    - 2 bytes: fp16 scale
    - 32 bytes: qs (lower 8 bits of indices)
    - 32 bytes: signs
    - 8 bytes: qh (upper 2 bits of indices)
    - 8 bytes: scales (4-bit sub-scales)
    """
    n_blocks = blocks.shape[0]
    device = blocks.device
    target_dtype = dtype if dtype else torch.float32

    # Split block: d(2) + qs(32) + signs(32) + qh(8) + scales(8) = 82 bytes
    d, rest = split_block_dims(blocks, 2)
    qs, rest = split_block_dims(rest, QK_K // 8)  # 32 bytes
    signs_raw, rest = split_block_dims(rest, QK_K // 8)  # 32 bytes
    qh, scales_raw = split_block_dims(rest, QK_K // 32)  # 8, 8 bytes

    d = d.view(torch.float16).to(target_dtype)

    # Unpack scales (4 bits each)
    scales_raw = scales_raw.reshape(n_blocks, -1, 1)
    shifts_s = torch.tensor([0, 4], dtype=torch.uint8, device=device).reshape(1, 1, 2)
    scales = ((scales_raw >> shifts_s) & 0x0F).reshape(n_blocks, -1)
    db = d * (0.5 + scales.to(target_dtype)) * 0.25
    db = db.reshape(n_blocks, -1, 1, 1)

    # Unpack sign bits
    signs_raw = signs_raw.reshape(n_blocks, -1, 1)
    bit_shifts = torch.arange(8, dtype=torch.uint8, device=device)
    signs = (signs_raw >> bit_shifts) & 1
    signs = torch.where(signs == 0, 1.0, -1.0).to(target_dtype)
    signs = signs.reshape(n_blocks, -1, 2, 8)

    # Combine qs and qh to get 10-bit indices
    qh = qh.reshape(n_blocks, -1, 1)
    qh_shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=device).reshape(1, 1, 4)
    qh_bits = ((qh >> qh_shifts) & 0x03).to(torch.int64).reshape(n_blocks, -1)
    qs_16 = qs.to(torch.int64) | (qh_bits << 8)

    # Look up grid values
    grid = _get_iq2_s_grid(device)
    grid_values = grid[qs_16.reshape(-1)].reshape(n_blocks, -1, 2, 8)
    grid_values = grid_values.to(target_dtype)

    return (db * grid_values * signs).reshape(n_blocks, -1)


# IQ3_S grid (512 entries x 4 values each)
IQ3_S_GRID_HEX = (
    b"0000010002000500070010001100120014001600200021002500330040004200"
    b"4500470051005300600062007100740077000001010102010401100111011501"
    b"2001230127013101350144016101650172010002010205020702100213021602"
    b"2102250230023402420245024702510253027002730203031103150320032203"
    b"3103330336034403500352036703710375030004130417042104240432044004"
    b"4304510470040205040520052205260533054105450547056605730506061106"
    b"1306310652067106000702070407200722072607330750075407001001100210"
    b"0410101011101310151017102010221031103410361054105610611072100011"
    b"0111031106111011141121113011331141115011521170117611001212121512"
    b"1712201224123212401243125512601272120113041307131013131321132713"
    b"3013341341136213701303140514121414143114331442144614501454140115"
    b"1015131521153015321551152016241627164416461601170317101712172117"
    b"3517411762177017002001200320052007201020122014201620212023202720"
    b"3020322041204320452050205220672070207320752000210221102113211721"
    b"2221252131213421422151210122042207222122232230223722412253225722"
    b"7122742200230223052311232223242331233323422350236623012407242024"
    b"2324322435244124722475240425112522253725402553257025002602260726"
    b"2126552661260527112726273027432750270230113013301530173022303130"
    b"3330353042304430473051306330713001310331053114312131233140316031"
    b"7231763100321232203232323432503201331033143321332333273330334133"
    b"4333473355337333033411341634223431345234603464340135103512352535"
    b"3235443556357335163641360137033720372237353700400440124020402440"
    b"2740324041405040704002410741114113412241304135414341514155410142"
    b"0342104215422142334240425742624270420443114313432043224331433543"
    b"0044024424443744404471440545074521456245134634466046104715473047"
    b"4347514702501050145022504050445047505250665074500151035105511251"
    b"2151325172510052115223523052365253520253075310532753445351536553"
    b"7353015404542054325446541255265551555355425602570457225711601360"
    b"1560316033606060006120612761646112623462426255626262706200631463"
    b"2163406325644364626400650365346560650566406611671367007004700770"
    b"2070227036704070547062700271117124714371457101720472107216722172"
    b"3072517202733273357353730174057413742074507422754275027631760077"
)
IQ3_S_GRID_MAP = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f)

_IQ3_S_GRID_CACHE = {}


def _build_iq3_s_grid(device):
    """Build IQ3_S grid (512 entries x 4 values)."""
    import numpy as np
    grid_bytes = np.frombuffer(IQ3_S_GRID_HEX, dtype=np.uint8).reshape(-1, 2)
    grid_bytes = np.where(grid_bytes > 0x40, grid_bytes + 9, grid_bytes) & 0x0F
    grid_bytes = (grid_bytes[:, 0] << 4) | grid_bytes[:, 1]
    # IQ3_S uses 3 bits per value, 8 values -> need different unpacking
    # Actually grid_map has 8 values (0-7 index), so 3 bits each
    grid_bytes = grid_bytes.reshape(-1, 1)
    shifts = np.array([0, 3, 6], dtype=np.uint8)  # 3 bits each, but only 3 fit in a byte
    # Actually need to unpack differently - grid is 512x4, so 4 values per entry
    # Each entry is 4 3-bit values = 12 bits = 1.5 bytes
    # Hex encodes as 2 chars per byte, so 3 chars per entry?
    # Let me re-examine: grid_shape = (512, 4), each value is indexed 0-7
    # Hex string has 1024 bytes = 2048 nibbles
    # 512 entries * 4 values = 2048 values, so 1 nibble per value? No, 3 bits...
    # Actually looking at grid_map = (1,3,5,7,9,11,13,15), these are 4-bit vals
    # So each nibble is one value. 2048 nibbles / 4 = 512 entries. Makes sense!
    shifts = np.array([0, 4], dtype=np.uint8)  # 4 bits each, 2 per byte
    grid_values = []
    for byte in grid_bytes.flatten():
        grid_values.append(IQ3_S_GRID_MAP[(byte >> 0) & 0x0F])
        grid_values.append(IQ3_S_GRID_MAP[(byte >> 4) & 0x0F])
    # Wait, that gives 1024 values, but we need 512*4=2048
    # Let me reconsider: the hex string is 1024 bytes after fromhex
    # After nibble conversion, we get 1024 bytes
    # grid_shape is (512, 4), so 2048 total values
    # 1024 bytes with 2 nibbles each = 2048 nibbles = 2048 values. Correct!
    grid_values = []
    for byte_val in grid_bytes.flatten():
        grid_values.append(IQ3_S_GRID_MAP[byte_val & 0x0F])
        grid_values.append(IQ3_S_GRID_MAP[(byte_val >> 4) & 0x0F])
    return torch.tensor(grid_values, dtype=torch.float32, device=device).reshape(512, 4)


def _get_iq3_s_grid(device):
    if device not in _IQ3_S_GRID_CACHE:
        _IQ3_S_GRID_CACHE[device] = _build_iq3_s_grid(device)
    return _IQ3_S_GRID_CACHE[device]


def dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    """
    Dequantize IQ3_S blocks.

    IQ3_S format (block_size=256, type_size=110):
    - 2 bytes: fp16 scale
    - 64 bytes: qs (lower 8 bits of indices)
    - 8 bytes: qh (upper 1 bit of indices)
    - 32 bytes: signs
    - 4 bytes: scales (4-bit sub-scales)
    """
    n_blocks = blocks.shape[0]
    device = blocks.device
    target_dtype = dtype if dtype else torch.float32

    # Split: d(2) + qs(64) + qh(8) + signs(32) + scales(4) = 110 bytes
    d, rest = split_block_dims(blocks, 2)
    qs, rest = split_block_dims(rest, QK_K // 4)  # 64 bytes
    qh, rest = split_block_dims(rest, QK_K // 32)  # 8 bytes
    signs_raw, scales_raw = split_block_dims(rest, QK_K // 8)  # 32, 4 bytes

    d = d.view(torch.float16).to(target_dtype)

    # Unpack scales
    scales_raw = scales_raw.reshape(n_blocks, -1, 1)
    shifts_s = torch.tensor([0, 4], dtype=torch.uint8, device=device).reshape(1, 1, 2)
    scales = ((scales_raw >> shifts_s) & 0x0F).reshape(n_blocks, -1)
    db = d * (1 + 2 * scales.to(target_dtype))
    db = db.reshape(n_blocks, -1, 1, 1)

    # Unpack sign bits
    signs_raw = signs_raw.reshape(n_blocks, -1, 1)
    bit_shifts = torch.arange(8, dtype=torch.uint8, device=device)
    signs = (signs_raw >> bit_shifts) & 1
    signs = torch.where(signs == 0, 1.0, -1.0).to(target_dtype)
    signs = signs.reshape(n_blocks, -1, 4, 8)

    # Combine qs and qh to get 9-bit indices
    qh = qh.reshape(n_blocks, -1, 1)
    qh_bits = (qh >> torch.arange(8, dtype=torch.uint8, device=device)) & 1
    qh_bits = qh_bits.to(torch.int64).reshape(n_blocks, -1)
    qs_16 = qs.to(torch.int64) | (qh_bits << 8)

    # Look up grid values
    grid = _get_iq3_s_grid(device)
    grid_values = grid[qs_16.reshape(-1)].reshape(n_blocks, -1, 4, 8)
    grid_values = grid_values.to(target_dtype)

    return (db * grid_values * signs).reshape(n_blocks, -1)


GGML_QUANT_SIZES = gguf.GGML_QUANT_SIZES
dequantize_functions = {
    gguf.GGMLQuantizationType.IQ2_XXS: dequantize_blocks_IQ2_XXS,
    gguf.GGMLQuantizationType.IQ2_S: dequantize_blocks_IQ2_S,
    gguf.GGMLQuantizationType.IQ3_S: dequantize_blocks_IQ3_S,
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
}
SUPPORTED_GGUF_QUANT_TYPES = list(dequantize_functions.keys())


def _quant_shape_from_byte_shape(shape, type_size, block_size):
    return (*shape[:-1], shape[-1] // type_size * block_size)


def dequantize_gguf_tensor(tensor):
    if not hasattr(tensor, "quant_type"):
        return tensor

    quant_type = tensor.quant_type
    dequant_fn = dequantize_functions[quant_type]

    block_size, type_size = GGML_QUANT_SIZES[quant_type]

    # Use stored tensor_shape if available (from CombinedGGUFLoader),
    # otherwise compute from byte shape
    if hasattr(tensor, 'tensor_shape') and tensor.tensor_shape is not None:
        shape = tensor.tensor_shape
    else:
        shape = _quant_shape_from_byte_shape(tensor.shape, type_size, block_size)

    tensor_bytes = tensor.view(torch.uint8)
    n_blocks = tensor_bytes.numel() // type_size
    blocks = tensor_bytes.reshape((n_blocks, type_size))

    dequant = dequant_fn(blocks, block_size, type_size)
    dequant = dequant.reshape(shape)

    return dequant.as_tensor()


class GGUFParameter(torch.nn.Parameter):
    def __new__(cls, data, requires_grad=False, quant_type=None):
        data = data if data is not None else torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_type = quant_type
        block_size, type_size = GGML_QUANT_SIZES[quant_type]
        self.quant_shape = _quant_shape_from_byte_shape(self.shape, type_size, block_size)

        return self

    def as_tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)

    @staticmethod
    def _extract_gguf_attrs(args):
        # When converting from original format checkpoints we often use splits, cats etc on tensors
        # this method ensures that the returned tensor type from those operations remains GGUFParameter
        # so that we preserve quant_type, tensor_shape, and needs_transpose information
        for arg in args:
            if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], GGUFParameter):
                return (
                    arg[0].quant_type,
                    getattr(arg[0], 'tensor_shape', None),
                    getattr(arg[0], 'needs_transpose', False),
                )
            if isinstance(arg, GGUFParameter):
                return (
                    arg.quant_type,
                    getattr(arg, 'tensor_shape', None),
                    getattr(arg, 'needs_transpose', False),
                )
        return None, None, False

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, torch.Tensor):
            quant_type, tensor_shape, needs_transpose = cls._extract_gguf_attrs(args)
            new_param = cls(result, quant_type=quant_type)
            if tensor_shape is not None:
                new_param.tensor_shape = tensor_shape
            if needs_transpose:
                new_param.needs_transpose = needs_transpose
            return new_param
        # Handle tuples and lists
        elif type(result) in (list, tuple):
            # Preserve the original type (tuple or list)
            quant_type, tensor_shape, needs_transpose = cls._extract_gguf_attrs(args)
            wrapped = []
            for x in result:
                if isinstance(x, torch.Tensor):
                    new_param = cls(x, quant_type=quant_type)
                    if tensor_shape is not None:
                        new_param.tensor_shape = tensor_shape
                    if needs_transpose:
                        new_param.needs_transpose = needs_transpose
                    wrapped.append(new_param)
                else:
                    wrapped.append(x)
            return type(result)(wrapped)
        else:
            return result


class GGUFLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.device = device

    def forward(self, inputs: torch.Tensor):
        # Use native path for transposed weights (GGML convention mismatch)
        needs_transpose = getattr(self.weight, 'needs_transpose', False)
        if ops is not None and self.weight.is_cuda and inputs.is_cuda and not needs_transpose:
            return self.forward_cuda(inputs)
        return self.forward_native(inputs)

    def forward_native(self, inputs: torch.Tensor):
        weight = dequantize_gguf_tensor(self.weight)

        # Handle GGML [in,out] vs PyTorch [out,in] convention mismatch
        if getattr(self.weight, 'needs_transpose', False):
            weight = weight.T.contiguous()

        weight = weight.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
        inputs = inputs.to(self.compute_dtype)

        output = torch.nn.functional.linear(inputs, weight, bias)
        return output

    def forward_cuda(self, inputs: torch.Tensor):
        quant_type = self.weight.quant_type
        output = _fused_mul_mat_gguf(inputs.to(self.compute_dtype), self.weight, quant_type)
        if self.bias is not None:
            output += self.bias.to(self.compute_dtype)
        return output
