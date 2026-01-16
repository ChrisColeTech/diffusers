"""Check AdaLN swap_scale_shift."""
import torch
from diffusers.loaders import CombinedGGUFLoader
from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor, GGUFParameter

loader = CombinedGGUFLoader(r'D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf')
state_dicts = loader.load()

# Check the raw norm_out key before conversion
print("Looking for final_layer.adaLN_modulation in raw state:")
for key in state_dicts["transformer"]:
    if "adaLN" in key or "final_layer" in key:
        print(f"  {key}")

# Now convert and check
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

# Reload to get fresh state
loader = CombinedGGUFLoader(r'D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf')
state_dicts = loader.load()
converted = convert_flux2_transformer_checkpoint_to_diffusers(state_dicts["transformer"])

print("\nLooking for norm_out in converted state:")
for key in converted:
    if "norm_out" in key:
        t = converted[key]
        if isinstance(t, GGUFParameter):
            dequant = dequantize_gguf_tensor(t)
        else:
            dequant = t
        print(f"  {key}: shape={dequant.shape}")

        # Check the weight structure - for AdaLN linear projecting to scale+shift
        # If shape is [2*hidden_dim, hidden_dim], first half is scale, second is shift
        if 'weight' in key and len(dequant.shape) == 2:
            out_dim, in_dim = dequant.shape
            half = out_dim // 2
            first_half = dequant[:half]
            second_half = dequant[half:]
            print(f"    First half (should be scale): mean={first_half.mean():.4f}, std={first_half.std():.4f}")
            print(f"    Second half (should be shift): mean={second_half.mean():.4f}, std={second_half.std():.4f}")
