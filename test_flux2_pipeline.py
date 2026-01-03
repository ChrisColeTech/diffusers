"""Test Flux2 pipeline assembly and image generation from combined GGUF."""
import torch
import time

# Paths
GGUF_PATH = r"D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf"
CONFIG_DIR = r"D:\Projects\giga-images-v2\generator-api-v3\configs\flux2_pipeline"
TOKENIZER_PATH = r"D:\models\image-models\flux2-dev\split\text_encoders\tokenizers"

# Multi-GPU setup: split components across GPUs
DEVICE_TRANSFORMER = "cuda:0"  # ~32B params
DEVICE_TEXT_ENCODER = "cuda:1"  # ~4B params
DEVICE_VAE = "cuda:0"  # ~84M params (small, share with transformer)
COMPUTE_DTYPE = torch.bfloat16


def load_component_with_gguf(model, state_dict, device):
    """Load state dict into model using GGUF quantizer."""
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    from diffusers.utils import get_module_from_name
    from accelerate.utils import set_module_tensor_to_device

    quantizer = GGUFQuantizer(GGUFQuantizationConfig(compute_dtype=COMPUTE_DTYPE))
    quantizer.validate_environment()

    # Replace Linear with GGUFLinear
    quantizer._process_model_before_weight_loading(
        model, device_map=None, state_dict=state_dict
    )

    # Load weights - delete from state_dict as we go to save memory
    param_names = list(state_dict.keys())
    for param_name in param_names:
        param_value = state_dict.pop(param_name)  # Remove from dict to free memory

        is_quantized = quantizer.check_if_quantized_param(
            model, param_value, param_name, {}
        )

        if is_quantized:
            module, tensor_name = get_module_from_name(model, param_name)
            current_param = getattr(module, tensor_name, None)
            if current_param is not None:
                quantizer.check_quantized_param_shape(param_name, current_param, param_value)

            quantizer.create_quantized_param(
                model, param_value, param_name, device, {}
            )
        else:
            # Convert non-quantized params to compute dtype
            if param_value.is_floating_point():
                param_value = param_value.to(COMPUTE_DTYPE)
            set_module_tensor_to_device(model, param_name, device, param_value)

        del param_value  # Explicitly delete reference

    return model


def main():
    print("=" * 60)
    print("Flux2 Pipeline Assembly and Image Generation Test")
    print("=" * 60)

    # Step 1: Load GGUF
    print("\n[1/6] Loading combined GGUF...")
    start = time.time()
    from diffusers.loaders import CombinedGGUFLoader
    loader = CombinedGGUFLoader(GGUF_PATH)
    state_dicts = loader.load()
    print(f"      Loaded in {time.time() - start:.1f}s")
    print(f"      Transformer: {len(state_dicts['transformer'])} tensors")
    print(f"      VAE: {len(state_dicts['vae'])} tensors")
    print(f"      Text Encoder: {len(state_dicts['text_encoder'])} tensors")

    # Step 2: Load Transformer
    print("\n[2/6] Loading transformer...")
    start = time.time()
    from diffusers.models import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from accelerate import init_empty_weights

    transformer_state = convert_flux2_transformer_checkpoint_to_diffusers(state_dicts["transformer"])
    del state_dicts["transformer"]  # Free memory
    config = Flux2Transformer2DModel.load_config(f"{CONFIG_DIR}/transformer")
    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(config)
    transformer = load_component_with_gguf(transformer, transformer_state, DEVICE_TRANSFORMER)
    del transformer_state  # Free memory
    # Convert norm layers to bfloat16 (fixes RMS norm dtype mismatch)
    for name, module in transformer.named_modules():
        if 'norm' in name.lower():
            module.to(COMPUTE_DTYPE)
    print(f"      Loaded to {DEVICE_TRANSFORMER} in {time.time() - start:.1f}s")

    # Step 3: Load VAE
    print("\n[3/6] Loading VAE...")
    start = time.time()
    from diffusers.models import AutoencoderKLFlux2

    vae_state = state_dicts["vae"]  # Already in diffusers format
    del state_dicts["vae"]  # Free memory
    config = AutoencoderKLFlux2.load_config(f"{CONFIG_DIR}/vae")
    with init_empty_weights():
        vae = AutoencoderKLFlux2.from_config(config)
    vae = load_component_with_gguf(vae, vae_state, DEVICE_VAE)
    vae = vae.to(COMPUTE_DTYPE)  # Ensure all params in correct dtype
    del vae_state  # Free memory
    print(f"      Loaded to {DEVICE_VAE} in {time.time() - start:.1f}s")

    # Step 4: Load Text Encoder
    print("\n[4/6] Loading text encoder...")
    start = time.time()
    # Clear cache and sync to avoid OOM on GPU 1
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    from transformers import Mistral3ForConditionalGeneration, AutoConfig
    from diffusers.loaders.single_file_utils import convert_ministral3_checkpoint_to_diffusers

    te_state = convert_ministral3_checkpoint_to_diffusers(state_dicts["text_encoder"])
    del state_dicts  # Free memory - we're done with all state dicts
    # Remove extra keys
    for k in ["model.vision_tower.image_break_token", "text_projection.bias", "text_projection.weight"]:
        te_state.pop(k, None)

    config = AutoConfig.from_pretrained(f"{CONFIG_DIR}/text_encoder")
    with init_empty_weights():
        text_encoder = Mistral3ForConditionalGeneration(config)
    text_encoder = load_component_with_gguf(text_encoder, te_state, DEVICE_TEXT_ENCODER)
    del te_state  # Free memory
    print(f"      Loaded to {DEVICE_TEXT_ENCODER} in {time.time() - start:.1f}s")

    # Step 5: Load Tokenizer and Scheduler
    print("\n[5/6] Loading tokenizer and scheduler...")
    start = time.time()
    from transformers import PreTrainedTokenizerFast
    from diffusers import FlowMatchEulerDiscreteScheduler

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{TOKENIZER_PATH}/tokenizer.json",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    # Add Mistral chat template
    tokenizer.chat_template = "{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{eos_token}}{% endif %}{% endfor %}"
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{CONFIG_DIR}/scheduler")
    print(f"      Loaded in {time.time() - start:.1f}s")

    # Step 6: Assemble Pipeline
    print("\n[6/6] Assembling pipeline...")
    start = time.time()
    from diffusers import Flux2Pipeline

    pipeline = Flux2Pipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    print(f"      Assembled in {time.time() - start:.1f}s")

    # Check VRAM usage on both GPUs
    if torch.cuda.is_available():
        for gpu_id in [0, 1]:
            device = f"cuda:{gpu_id}"
            vram_used = torch.cuda.memory_allocated(device) / 1024**3
            vram_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"      GPU {gpu_id}: {vram_used:.1f}GB used, {vram_reserved:.1f}GB reserved")

    # Generate image
    print("\n" + "=" * 60)
    print("Generating test image...")
    print("=" * 60)

    prompt = "a beautiful sunset over mountains, photorealistic"
    print(f"Prompt: {prompt}")

    start = time.time()
    # Ministral3 has 26 layers, transformer expects 5 layers (15360 = 5 * 3072)
    result = pipeline(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=3.5,
        height=512,
        width=512,
        text_encoder_out_layers=(5, 10, 15, 20, 25),  # 5 layers for 15360-dim
    )
    gen_time = time.time() - start

    image = result.images[0]
    output_path = "test_flux2_output.png"
    image.save(output_path)

    print(f"\nGeneration completed in {gen_time:.1f}s")
    print(f"Image saved to: {output_path}")
    print("\n[SUCCESS] Pipeline test complete!")


if __name__ == "__main__":
    main()
