"""Debug transformer block internals."""
import torch

GGUF_PATH = r"D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf"
CONFIG_DIR = r"D:\Projects\giga-images-v2\generator-api-v3\configs\flux2_pipeline"
DEVICE = "cuda:0"
COMPUTE_DTYPE = torch.bfloat16


def main():
    from diffusers.loaders import CombinedGGUFLoader
    from diffusers.models import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    from diffusers.utils import get_module_from_name
    from accelerate.utils import set_module_tensor_to_device
    from accelerate import init_empty_weights

    print("Loading transformer...")
    loader = CombinedGGUFLoader(GGUF_PATH)
    state_dicts = loader.load()
    transformer_state = convert_flux2_transformer_checkpoint_to_diffusers(state_dicts["transformer"])

    config = Flux2Transformer2DModel.load_config(f"{CONFIG_DIR}/transformer")
    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(config)

    quantizer = GGUFQuantizer(GGUFQuantizationConfig(compute_dtype=COMPUTE_DTYPE))
    quantizer.validate_environment()
    quantizer._process_model_before_weight_loading(transformer, device_map=None, state_dict=transformer_state)

    param_names = list(transformer_state.keys())
    for param_name in param_names:
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

    # Create inputs that match what the pipeline would create
    batch_size = 1
    seq_len_img = 64
    seq_len_txt = 32

    # These are the embedded inputs (after x_embedder and context_embedder)
    hidden_states = torch.randn(batch_size, seq_len_img, 6144, device=DEVICE, dtype=COMPUTE_DTYPE) * 0.02
    encoder_hidden_states = torch.randn(batch_size, seq_len_txt, 6144, device=DEVICE, dtype=COMPUTE_DTYPE) * 0.02
    temb = torch.randn(batch_size, 6144, device=DEVICE, dtype=COMPUTE_DTYPE) * 0.1

    print(f"\nInputs to block 0:")
    print(f"  hidden_states: {hidden_states.shape}, std={hidden_states.std():.4f}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}, std={encoder_hidden_states.std():.4f}")
    print(f"  temb: {temb.shape}, std={temb.std():.4f}")

    # Get first transformer block
    block = transformer.transformer_blocks[0]

    # Hook into attention components
    hooks = []
    values = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0] if output[0] is not None else output[1]
            else:
                out = output
            if out is not None and hasattr(out, 'shape'):
                values[name] = f"shape={out.shape}, min={out.min():.2f}, max={out.max():.2f}, std={out.std():.4f}"
        return hook

    # Register hooks on attention submodules
    if hasattr(block, 'attn'):
        attn = block.attn
        if hasattr(attn, 'to_q'):
            hooks.append(attn.to_q.register_forward_hook(make_hook('attn.to_q')))
        if hasattr(attn, 'to_k'):
            hooks.append(attn.to_k.register_forward_hook(make_hook('attn.to_k')))
        if hasattr(attn, 'to_v'):
            hooks.append(attn.to_v.register_forward_hook(make_hook('attn.to_v')))
        if hasattr(attn, 'to_out') and len(attn.to_out) > 0:
            hooks.append(attn.to_out[0].register_forward_hook(make_hook('attn.to_out')))

    if hasattr(block, 'ff'):
        ff = block.ff
        if hasattr(ff, 'linear_in'):
            hooks.append(ff.linear_in.register_forward_hook(make_hook('ff.linear_in')))
        if hasattr(ff, 'linear_out'):
            hooks.append(ff.linear_out.register_forward_hook(make_hook('ff.linear_out')))

    # Also hook modulation
    if hasattr(transformer, 'double_stream_modulation_img'):
        hooks.append(transformer.double_stream_modulation_img.register_forward_hook(make_hook('modulation_img')))

    print("\nRunning block 0 forward...")
    block.eval()
    transformer.eval()

    with torch.no_grad():
        # First compute modulation
        mod_img = transformer.double_stream_modulation_img(temb)
        mod_txt = transformer.double_stream_modulation_txt(temb)

        print(f"\nModulation outputs:")
        print(f"  mod_img: shape={mod_img.shape}, min={mod_img.min():.4f}, max={mod_img.max():.4f}, std={mod_img.std():.4f}")
        print(f"  mod_txt: shape={mod_txt.shape}, min={mod_txt.min():.4f}, max={mod_txt.max():.4f}, std={mod_txt.std():.4f}")

        # Run the block
        output = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

    print(f"\nBlock output:")
    if isinstance(output, tuple):
        for i, o in enumerate(output):
            if o is not None:
                print(f"  output[{i}]: shape={o.shape}, min={o.min():.2f}, max={o.max():.2f}, std={o.std():.4f}")
    else:
        print(f"  output: shape={output.shape}, min={output.min():.2f}, max={output.max():.2f}, std={output.std():.4f}")

    print(f"\nIntermediate values:")
    for name, val in values.items():
        print(f"  {name}: {val}")

    # Clean up
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
