"""Test transformer forward pass with debug output."""
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

    print("Loading GGUF...")
    loader = CombinedGGUFLoader(GGUF_PATH)
    state_dicts = loader.load()

    print("Converting and loading transformer...")
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

    # Convert norm layers
    for name, module in transformer.named_modules():
        if 'norm' in name.lower():
            module.to(COMPUTE_DTYPE)

    print("Transformer loaded!")

    # Create test inputs
    batch_size = 1
    seq_len_img = 64  # Small for testing
    seq_len_txt = 32
    hidden_dim = 6144

    # Random inputs
    hidden_states = torch.randn(batch_size, seq_len_img, 128, device=DEVICE, dtype=COMPUTE_DTYPE) * 0.1
    encoder_hidden_states = torch.randn(batch_size, seq_len_txt, 15360, device=DEVICE, dtype=COMPUTE_DTYPE) * 0.1
    timestep = torch.tensor([0.5], device=DEVICE, dtype=COMPUTE_DTYPE)
    guidance = torch.tensor([3.5], device=DEVICE, dtype=torch.float32)
    img_ids = torch.zeros(batch_size, seq_len_img, 4, device=DEVICE, dtype=COMPUTE_DTYPE)
    txt_ids = torch.zeros(batch_size, seq_len_txt, 4, device=DEVICE, dtype=COMPUTE_DTYPE)

    print(f"\nInput shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  timestep: {timestep}")
    print(f"  guidance: {guidance}")

    # Hook to capture intermediate values
    intermediate_values = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out is not None and hasattr(out, 'shape'):
                intermediate_values[name] = {
                    'shape': out.shape,
                    'min': out.min().item(),
                    'max': out.max().item(),
                    'mean': out.mean().item(),
                    'std': out.std().item(),
                }
        return hook

    # Register hooks on key modules
    hooks = []
    hooks.append(transformer.x_embedder.register_forward_hook(make_hook('x_embedder')))
    hooks.append(transformer.context_embedder.register_forward_hook(make_hook('context_embedder')))
    hooks.append(transformer.time_guidance_embed.register_forward_hook(make_hook('time_guidance_embed')))

    if hasattr(transformer, 'transformer_blocks') and len(transformer.transformer_blocks) > 0:
        hooks.append(transformer.transformer_blocks[0].register_forward_hook(make_hook('transformer_block_0')))

    if hasattr(transformer, 'single_transformer_blocks') and len(transformer.single_transformer_blocks) > 0:
        hooks.append(transformer.single_transformer_blocks[0].register_forward_hook(make_hook('single_block_0')))

    hooks.append(transformer.proj_out.register_forward_hook(make_hook('proj_out')))

    # Forward pass
    print("\nRunning forward pass...")
    transformer.eval()
    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    print(f"\nOutput shape: {output.shape}")
    print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}, std={output.std():.4f}")

    print("\n=== Intermediate Values ===")
    for name, stats in intermediate_values.items():
        print(f"{name}:")
        print(f"  shape: {stats['shape']}")
        print(f"  min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")

        # Check for problems
        if stats['std'] < 0.0001:
            print(f"  ⚠️  WARNING: Very low std - might be dead/constant!")
        if abs(stats['mean']) > 10:
            print(f"  ⚠️  WARNING: Large mean - might be exploding!")
        if stats['std'] > 100:
            print(f"  ⚠️  WARNING: Very high std - might be exploding!")

    # Clean up hooks
    for h in hooks:
        h.remove()

    print("\n✓ Forward pass completed")


if __name__ == "__main__":
    main()
