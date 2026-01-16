"""Check VAE tensor shapes from GGUF."""
from diffusers.loaders import CombinedGGUFLoader

loader = CombinedGGUFLoader(r'D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf')

# Read raw tensors without the fix
import gguf
reader = gguf.GGUFReader(loader.gguf_path)

print("VAE conv weight shapes (raw from GGUF):")
for tensor in reader.tensors:
    if tensor.name.startswith("vae.") and "conv" in tensor.name and "weight" in tensor.name:
        shape = tuple(int(d) for d in tensor.shape)
        print(f"  {tensor.name}: {shape}")
        if len(shape) == 4:
            # Show interpretation
            print(f"    -> If [H,W,in,out]: kernel={shape[0]}x{shape[1]}, in={shape[2]}, out={shape[3]}")
            print(f"    -> If [out,in,H,W]: out={shape[0]}, in={shape[1]}, kernel={shape[2]}x{shape[3]}")
