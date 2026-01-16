"""Test GGUF dequantization produces reasonable values."""
import torch
from diffusers.loaders import CombinedGGUFLoader
from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor, GGUFParameter

GGUF_PATH = r"D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf"

print("Loading GGUF...")
loader = CombinedGGUFLoader(GGUF_PATH)
state_dicts = loader.load()

print("\n=== Transformer Weights ===")
# Check a transformer weight
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
transformer_state = convert_flux2_transformer_checkpoint_to_diffusers(state_dicts["transformer"])

key = "transformer_blocks.0.attn.to_q.weight"
if key in transformer_state:
    t = transformer_state[key]
    print(f"\n{key}:")
    print(f"  Type: {type(t).__name__}")
    if isinstance(t, GGUFParameter):
        print(f"  quant_type: {t.quant_type}")
        print(f"  tensor_shape: {getattr(t, 'tensor_shape', None)}")
        print(f"  needs_transpose: {getattr(t, 'needs_transpose', False)}")
        print(f"  Raw data shape: {t.shape}, dtype: {t.dtype}")

        # Dequantize
        print("\n  Dequantizing...")
        dequant = dequantize_gguf_tensor(t)
        print(f"  Dequantized shape: {dequant.shape}")
        print(f"  Dequantized dtype: {dequant.dtype}")
        print(f"  Dequantized stats: min={dequant.min():.4f}, max={dequant.max():.4f}, mean={dequant.mean():.4f}, std={dequant.std():.4f}")

        # Check if values are reasonable for a neural network weight
        if dequant.std() < 0.001:
            print("  WARNING: Very low std - weights might be zeros!")
        elif dequant.std() > 10:
            print("  WARNING: Very high std - weights might be corrupted!")
        else:
            print("  Stats look reasonable for NN weights")

print("\n=== Text Encoder Weights ===")
from diffusers.loaders.single_file_utils import convert_ministral3_checkpoint_to_diffusers
te_state = convert_ministral3_checkpoint_to_diffusers(state_dicts["text_encoder"])

key = "model.language_model.layers.0.self_attn.q_proj.weight"
if key in te_state:
    t = te_state[key]
    print(f"\n{key}:")
    print(f"  Type: {type(t).__name__}")
    if isinstance(t, GGUFParameter):
        print(f"  quant_type: {t.quant_type}")
        print(f"  tensor_shape: {getattr(t, 'tensor_shape', None)}")
        print(f"  needs_transpose: {getattr(t, 'needs_transpose', False)}")

        # Dequantize
        print("\n  Dequantizing...")
        dequant = dequantize_gguf_tensor(t)
        print(f"  Dequantized shape: {dequant.shape}")
        print(f"  Dequantized stats: min={dequant.min():.4f}, max={dequant.max():.4f}, mean={dequant.mean():.4f}, std={dequant.std():.4f}")

print("\n=== VAE Weights ===")
key = "decoder.conv_in.weight"
if key in state_dicts["vae"]:
    t = state_dicts["vae"][key]
    print(f"\n{key}:")
    print(f"  Type: {type(t).__name__}")
    print(f"  Shape: {t.shape}")
    print(f"  Stats: min={t.min():.4f}, max={t.max():.4f}, mean={t.mean():.4f}, std={t.std():.4f}")
