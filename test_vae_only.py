"""Test VAE encode/decode independently."""
import torch
from PIL import Image
import numpy as np

GGUF_PATH = r"D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf"
CONFIG_DIR = r"D:\Projects\giga-images-v2\generator-api-v3\configs\flux2_pipeline"
DEVICE = "cuda:0"
COMPUTE_DTYPE = torch.bfloat16

def load_vae():
    from diffusers.loaders import CombinedGGUFLoader
    from diffusers.models import AutoencoderKLFlux2
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    from diffusers.utils import get_module_from_name
    from accelerate.utils import set_module_tensor_to_device
    from accelerate import init_empty_weights

    print("Loading GGUF...")
    loader = CombinedGGUFLoader(GGUF_PATH)
    state_dicts = loader.load()
    vae_state = state_dicts["vae"]

    print("Creating VAE model...")
    config = AutoencoderKLFlux2.load_config(f"{CONFIG_DIR}/vae")
    with init_empty_weights():
        vae = AutoencoderKLFlux2.from_config(config)

    print("Loading weights...")
    quantizer = GGUFQuantizer(GGUFQuantizationConfig(compute_dtype=COMPUTE_DTYPE))
    quantizer.validate_environment()
    quantizer._process_model_before_weight_loading(vae, device_map=None, state_dict=vae_state)

    param_names = list(vae_state.keys())
    for param_name in param_names:
        param_value = vae_state.pop(param_name)
        is_quantized = quantizer.check_if_quantized_param(vae, param_value, param_name, {})

        if is_quantized:
            module, tensor_name = get_module_from_name(vae, param_name)
            current_param = getattr(module, tensor_name, None)
            if current_param is not None:
                quantizer.check_quantized_param_shape(param_name, current_param, param_value)
            quantizer.create_quantized_param(vae, param_value, param_name, DEVICE, {})
        else:
            if param_value.is_floating_point():
                param_value = param_value.to(COMPUTE_DTYPE)
            set_module_tensor_to_device(vae, param_name, DEVICE, param_value)

    vae = vae.to(COMPUTE_DTYPE)
    return vae

def main():
    vae = load_vae()
    vae.eval()

    # Create a simple test image (gradient)
    print("\nCreating test image...")
    width, height = 256, 256
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [int(255 * x / width), int(255 * y / height), 128]

    test_img = Image.fromarray(img_array)
    test_img.save("test_vae_input.png")
    print("Saved test_vae_input.png")

    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor * 2 - 1  # Scale to [-1, 1]
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE, COMPUTE_DTYPE)

    print(f"Input shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")

    # Encode
    print("Encoding...")
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
    print(f"Latent shape: {latent.shape}, min: {latent.min():.3f}, max: {latent.max():.3f}")

    # Decode
    print("Decoding...")
    with torch.no_grad():
        decoded = vae.decode(latent).sample
    print(f"Decoded shape: {decoded.shape}, min: {decoded.min():.3f}, max: {decoded.max():.3f}")

    # Convert back to image
    decoded = (decoded + 1) / 2  # Scale to [0, 1]
    decoded = decoded.clamp(0, 1)
    decoded = decoded[0].permute(1, 2, 0).cpu().float().numpy()
    decoded = (decoded * 255).astype(np.uint8)

    output_img = Image.fromarray(decoded)
    output_img.save("test_vae_output.png")
    print("Saved test_vae_output.png")
    print("\nCompare test_vae_input.png and test_vae_output.png to verify VAE works!")

if __name__ == "__main__":
    main()
