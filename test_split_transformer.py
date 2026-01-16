"""Test split transformer GGUF independently."""
import torch
import time

TRANSFORMER_GGUF = r"D:\models\image-models\flux2-dev\split\flux2-dev-Q6_K.gguf"
VAE_GGUF = r"D:\models\image-models\flux2-dev\split\flux2-vae-f16.gguf"
CONFIG_DIR = r"D:\Projects\giga-images-v2\generator-api-v3\configs\flux2_pipeline"
DEVICE = "cuda:0"
COMPUTE_DTYPE = torch.bfloat16


def load_single_gguf(gguf_path):
    """Load a single-component GGUF file."""
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter

    reader = gguf.GGUFReader(gguf_path)
    state_dict = {}

    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type
        data = torch.from_numpy(tensor.data.copy())
        logical_shape = torch.Size(tuple(int(d) for d in tensor.shape))

        is_quantized = quant_type not in [
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        ]

        if is_quantized:
            param = GGUFParameter(data, quant_type=quant_type)
            param.tensor_shape = logical_shape
        else:
            param = data.reshape(logical_shape)

        state_dict[name] = param

    return state_dict


def load_component_with_gguf(model, state_dict, device):
    """Load state dict into model using GGUF quantizer."""
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    from diffusers.utils import get_module_from_name
    from accelerate.utils import set_module_tensor_to_device

    quantizer = GGUFQuantizer(GGUFQuantizationConfig(compute_dtype=COMPUTE_DTYPE))
    quantizer.validate_environment()
    quantizer._process_model_before_weight_loading(model, device_map=None, state_dict=state_dict)

    param_names = list(state_dict.keys())
    for param_name in param_names:
        param_value = state_dict.pop(param_name)
        is_quantized = quantizer.check_if_quantized_param(model, param_value, param_name, {})

        if is_quantized:
            module, tensor_name = get_module_from_name(model, param_name)
            current_param = getattr(module, tensor_name, None)
            if current_param is not None:
                quantizer.check_quantized_param_shape(param_name, current_param, param_value)
            quantizer.create_quantized_param(model, param_value, param_name, device, {})
        else:
            if param_value.is_floating_point():
                param_value = param_value.to(COMPUTE_DTYPE)
            set_module_tensor_to_device(model, param_name, device, param_value)

    return model


def main():
    print("=" * 60)
    print("Testing Split Transformer GGUF")
    print("=" * 60)

    # Load transformer
    print("\n[1/3] Loading split transformer GGUF...")
    start = time.time()
    from diffusers.models import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from accelerate import init_empty_weights

    raw_state = load_single_gguf(TRANSFORMER_GGUF)
    print(f"      Raw tensors: {len(raw_state)}")

    # Check a sample weight shape
    sample_key = list(raw_state.keys())[0]
    sample_tensor = raw_state[sample_key]
    print(f"      Sample: {sample_key}")
    if hasattr(sample_tensor, 'tensor_shape'):
        print(f"        tensor_shape: {sample_tensor.tensor_shape}")
    print(f"        shape: {sample_tensor.shape}")

    transformer_state = convert_flux2_transformer_checkpoint_to_diffusers(raw_state)
    print(f"      Converted tensors: {len(transformer_state)}")

    config = Flux2Transformer2DModel.load_config(f"{CONFIG_DIR}/transformer")
    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(config)
    transformer = load_component_with_gguf(transformer, transformer_state, DEVICE)

    # Convert norm layers
    for name, module in transformer.named_modules():
        if 'norm' in name.lower():
            module.to(COMPUTE_DTYPE)

    print(f"      Loaded in {time.time() - start:.1f}s")

    # Load VAE
    print("\n[2/3] Loading split VAE GGUF...")
    start = time.time()
    from diffusers.models import AutoencoderKLFlux2

    vae_state = load_single_gguf(VAE_GGUF)

    # Fix VAE shapes (reshape, not permute)
    for key in list(vae_state.keys()):
        param = vae_state[key]
        if not hasattr(param, 'tensor_shape'):
            shape = param.shape
        else:
            shape = param.tensor_shape

        if len(shape) == 4:
            h, w, in_ch, out_ch = shape
            if h <= 7 and w <= 7 and (in_ch > 7 or out_ch > 7):
                new_shape = torch.Size((out_ch, in_ch, h, w))
                if hasattr(param, 'tensor_shape'):
                    param.tensor_shape = new_shape
                else:
                    vae_state[key] = param.reshape(new_shape)

    config = AutoencoderKLFlux2.load_config(f"{CONFIG_DIR}/vae")
    with init_empty_weights():
        vae = AutoencoderKLFlux2.from_config(config)
    vae = load_component_with_gguf(vae, vae_state, DEVICE)
    vae = vae.to(COMPUTE_DTYPE)
    print(f"      Loaded in {time.time() - start:.1f}s")

    # Test denoising with random embeddings
    print("\n[3/3] Testing denoising with random embeddings...")
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{CONFIG_DIR}/scheduler")

    # Create random latents
    batch_size = 1
    height, width = 512, 512
    latent_h, latent_w = height // 16, width // 16
    num_channels = transformer.config.in_channels // 4

    latents = torch.randn(batch_size, latent_h * latent_w, num_channels * 4, device=DEVICE, dtype=COMPUTE_DTYPE)

    # Create random text embeddings (correct shape: batch, seq_len, 15360)
    seq_len = 128
    prompt_embeds = torch.randn(batch_size, seq_len, 15360, device=DEVICE, dtype=COMPUTE_DTYPE) * 0.1

    # Create IDs
    latent_ids = torch.zeros(batch_size, latent_h * latent_w, 4, device=DEVICE, dtype=COMPUTE_DTYPE)
    text_ids = torch.zeros(batch_size, seq_len, 4, device=DEVICE, dtype=COMPUTE_DTYPE)

    # Setup scheduler
    num_steps = 10
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    guidance = torch.tensor([3.5], device=DEVICE, dtype=torch.float32).expand(batch_size)

    print(f"      Latents shape: {latents.shape}")
    print(f"      Prompt embeds shape: {prompt_embeds.shape}")
    print(f"      Running {num_steps} denoising steps...")

    transformer.eval()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size).to(COMPUTE_DTYPE) / 1000

            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if i == 0:
                print(f"      Step 0 - noise_pred: min={noise_pred.min():.3f}, max={noise_pred.max():.3f}, std={noise_pred.std():.3f}")

    print(f"      Final latents: min={latents.min():.3f}, max={latents.max():.3f}, std={latents.std():.3f}")

    # Decode
    print("\n      Decoding...")
    latents_for_vae = latents.reshape(batch_size, latent_h, latent_w, -1).permute(0, 3, 1, 2)
    latents_for_vae = latents_for_vae / vae.config.scaling_factor

    with torch.no_grad():
        image = vae.decode(latents_for_vae, return_dict=False)[0]

    print(f"      Decoded image: min={image.min():.3f}, max={image.max():.3f}")

    # Save
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image[0].permute(1, 2, 0).cpu().float().numpy()
    image = (image * 255).astype('uint8')

    from PIL import Image
    Image.fromarray(image).save("test_split_transformer_output.png")
    print("\n      Saved to test_split_transformer_output.png")
    print("\n[DONE] Check if output is noise or has any structure")


if __name__ == "__main__":
    main()
