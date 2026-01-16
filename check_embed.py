from diffusers.loaders import CombinedGGUFLoader
from diffusers.loaders.single_file_utils import convert_ministral3_checkpoint_to_diffusers
from diffusers.quantizers.gguf.utils import GGUFParameter

loader = CombinedGGUFLoader(r'D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf')
state_dicts = loader.load()
converted = convert_ministral3_checkpoint_to_diffusers(state_dicts['text_encoder'])

# Check embed_tokens
key = 'model.language_model.embed_tokens.weight'
if key in converted:
    t = converted[key]
    print(f'{key}:')
    print(f'  Type: {type(t).__name__}')
    print(f'  Shape: {t.shape}')
    if isinstance(t, GGUFParameter):
        print(f'  tensor_shape: {getattr(t, "tensor_shape", None)}')
        print(f'  quant_type: {t.quant_type}')
        print(f'  needs_transpose: {getattr(t, "needs_transpose", False)}')
