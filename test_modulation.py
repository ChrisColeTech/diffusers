"""Check modulation outputs."""
import torch
DEVICE = 'cuda:0'
COMPUTE_DTYPE = torch.bfloat16

from diffusers.loaders import CombinedGGUFLoader
from diffusers.models import Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
from diffusers import GGUFQuantizationConfig
from diffusers.quantizers.gguf import GGUFQuantizer
from diffusers.utils import get_module_from_name
from accelerate.utils import set_module_tensor_to_device
from accelerate import init_empty_weights

print("Loading...")
loader = CombinedGGUFLoader(r'D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf')
state_dicts = loader.load()
transformer_state = convert_flux2_transformer_checkpoint_to_diffusers(state_dicts['transformer'])

config = Flux2Transformer2DModel.load_config(r'D:\Projects\giga-images-v2\generator-api-v3\configs\flux2_pipeline\transformer')
with init_empty_weights():
    transformer = Flux2Transformer2DModel.from_config(config)

quantizer = GGUFQuantizer(GGUFQuantizationConfig(compute_dtype=COMPUTE_DTYPE))
quantizer.validate_environment()
quantizer._process_model_before_weight_loading(transformer, device_map=None, state_dict=transformer_state)

for param_name in list(transformer_state.keys()):
    param_value = transformer_state.pop(param_name)
    is_quantized = quantizer.check_if_quantized_param(transformer, param_value, param_name, {})
    if is_quantized:
        module, tensor_name = get_module_from_name(transformer, param_name)
        current_param = getattr(module, tensor_name, None)
        if current_param is not None:
            quantizer.check_quantized_param_shape(param_name, current_param, param_value)
        quantizer.create_quantized_param(transformer, param_value, param_name, DEVICE, {})
    else:
        if param_value.is_floating_point():
            param_value = param_value.to(COMPUTE_DTYPE)
        set_module_tensor_to_device(transformer, param_name, DEVICE, param_value)

for name, module in transformer.named_modules():
    if 'norm' in name.lower():
        module.to(COMPUTE_DTYPE)

print("Loaded!")

# Check modulation output
temb = torch.randn(1, 6144, device=DEVICE, dtype=COMPUTE_DTYPE) * 0.1
print(f"\nInput temb: shape={temb.shape}, std={temb.std():.4f}")

mod = transformer.double_stream_modulation_img(temb)
print(f'\nModulation output type: {type(mod)}')
if isinstance(mod, tuple):
    for i, m in enumerate(mod):
        print(f'  [{i}]: shape={m.shape}, min={m.min():.4f}, max={m.max():.4f}, std={m.std():.4f}')
else:
    print(f'  shape={mod.shape}, min={mod.min():.4f}, max={mod.max():.4f}, std={mod.std():.4f}')

# The modulation should produce scale and shift values
# For AdaLN, these are typically around 1 for scale and 0 for shift
print("\nExpected: scale ~1, shift ~0 for stable training")
