"""Check transformer QKV weight shapes."""
import gguf

reader = gguf.GGUFReader(r'D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf')

print("Transformer QKV weights:")
for tensor in reader.tensors:
    if tensor.name.startswith('transformer.') and 'qkv' in tensor.name:
        shape = tuple(int(d) for d in tensor.shape)
        print(f"  {tensor.name}: {shape}")

print("\nTransformer img_in/txt_in (context embedders):")
for tensor in reader.tensors:
    if tensor.name.startswith('transformer.') and ('img_in' in tensor.name or 'txt_in' in tensor.name):
        shape = tuple(int(d) for d in tensor.shape)
        print(f"  {tensor.name}: {shape}")

print("\nTransformer time_in/guidance_in:")
for tensor in reader.tensors:
    if tensor.name.startswith('transformer.') and ('time_in' in tensor.name or 'guidance_in' in tensor.name):
        shape = tuple(int(d) for d in tensor.shape)
        print(f"  {tensor.name}: {shape}")
