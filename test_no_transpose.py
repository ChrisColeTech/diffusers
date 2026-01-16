"""Test combined GGUF with transpose detection disabled for transformer."""
import torch
import time

GGUF_PATH = r"D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf"
CONFIG_DIR = r"D:\Projects\giga-images-v2\generator-api-v3\configs\flux2_pipeline"
TOKENIZER_PATH = r"D:\models\image-models\flux2-dev\split\text_encoders\tokenizers"

DEVICE_TRANSFORMER = "cuda:0"
DEVICE_TEXT_ENCODER = "cuda:1"
DEVICE_VAE = "cuda:0"
COMPUTE_DTYPE = torch.bfloat16


def load_component_with_gguf(model, state_dict, device, skip_transpose_check=False):
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

            if current_param is not None and not skip_transpose_check:
                quantizer.check_quantized_param_shape(param_name, current_param, param_value)

            # For transformer with skip_transpose_check, clear any transpose flag
            if skip_transpose_check and hasattr(param_value, 'needs_transpose'):
                param_value.needs_transpose = False

            quantizer.create_quantized_param(model, param_value, param_name, device, {})
        else:
            if param_value.is_floating_point():
                param_value = param_value.to(COMPUTE_DTYPE)
            set_module_tensor_to_device(model, param_name, device, param_value)

    return model


def main():
    print("=" * 60)
    print("Test: Combined GGUF with NO transpose for transformer")
    print("=" * 60)

    # Load GGUF
    print("\n[1/6] Loading combined GGUF...")
    start = time.time()
    from diffusers.loaders import CombinedGGUFLoader
    loader = CombinedGGUFLoader(GGUF_PATH)
    state_dicts = loader.load()
    print(f"      Loaded in {time.time() - start:.1f}s")

    # Load Transformer - skip transpose check
    print("\n[2/6] Loading transformer (NO transpose)...")
    start = time.time()
    from diffusers.models import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from accelerate import init_empty_weights

    transformer_state = convert_flux2_transformer_checkpoint_to_diffusers(state_dicts["transformer"])
    del state_dicts["transformer"]

    # Check a weight's needs_transpose status
    sample_key = "transformer_blocks.0.attn.to_q.weight"
    if sample_key in transformer_state:
        t = transformer_state[sample_key]
        print(f"      Sample weight: {sample_key}")
        print(f"        shape: {t.shape if not hasattr(t, 'tensor_shape') else t.tensor_shape}")
        print(f"        needs_transpose before: {getattr(t, 'needs_transpose', False)}")

    config = Flux2Transformer2DModel.load_config(f"{CONFIG_DIR}/transformer")
    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(config)

    # Load with skip_transpose_check=True to avoid setting transpose flag
    transformer = load_component_with_gguf(transformer, transformer_state, DEVICE_TRANSFORMER, skip_transpose_check=True)
    del transformer_state

    for name, module in transformer.named_modules():
        if 'norm' in name.lower():
            module.to(COMPUTE_DTYPE)
    print(f"      Loaded to {DEVICE_TRANSFORMER} in {time.time() - start:.1f}s")

    # Load VAE
    print("\n[3/6] Loading VAE...")
    start = time.time()
    from diffusers.models import AutoencoderKLFlux2

    vae_state = state_dicts["vae"]
    del state_dicts["vae"]
    config = AutoencoderKLFlux2.load_config(f"{CONFIG_DIR}/vae")
    with init_empty_weights():
        vae = AutoencoderKLFlux2.from_config(config)
    vae = load_component_with_gguf(vae, vae_state, DEVICE_VAE)
    vae = vae.to(COMPUTE_DTYPE)
    del vae_state
    print(f"      Loaded to {DEVICE_VAE} in {time.time() - start:.1f}s")

    # Load Text Encoder - keep transpose check (text encoder needs it)
    print("\n[4/6] Loading text encoder (WITH transpose)...")
    start = time.time()
    torch.cuda.empty_cache()
    from transformers import Mistral3ForConditionalGeneration, AutoConfig
    from diffusers.loaders.single_file_utils import convert_ministral3_checkpoint_to_diffusers

    te_state = convert_ministral3_checkpoint_to_diffusers(state_dicts["text_encoder"])
    del state_dicts
    for k in ["model.vision_tower.image_break_token", "text_projection.bias", "text_projection.weight"]:
        te_state.pop(k, None)

    config = AutoConfig.from_pretrained(f"{CONFIG_DIR}/text_encoder")
    with init_empty_weights():
        text_encoder = Mistral3ForConditionalGeneration(config)
    text_encoder = load_component_with_gguf(text_encoder, te_state, DEVICE_TEXT_ENCODER, skip_transpose_check=False)
    del te_state
    print(f"      Loaded to {DEVICE_TEXT_ENCODER} in {time.time() - start:.1f}s")

    # Load Tokenizer and Scheduler
    print("\n[5/6] Loading tokenizer and scheduler...")
    from transformers import PreTrainedTokenizerFast
    from diffusers import FlowMatchEulerDiscreteScheduler

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{TOKENIZER_PATH}/tokenizer.json",
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )
    tokenizer.chat_template = "{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{eos_token}}{% endif %}{% endfor %}"
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{CONFIG_DIR}/scheduler")
    print("      Done")

    # Assemble Pipeline
    print("\n[6/6] Assembling pipeline...")
    from diffusers import Flux2Pipeline

    pipeline = Flux2Pipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    # Generate
    print("\n" + "=" * 60)
    print("Generating test image...")
    print("=" * 60)

    prompt = "a beautiful sunset over mountains, photorealistic"
    print(f"Prompt: {prompt}")

    start = time.time()
    result = pipeline(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=3.5,
        height=512,
        width=512,
        text_encoder_out_layers=(5, 10, 15, 20, 25),
    )

    image = result.images[0]
    image.save("test_no_transpose_output.png")
    print(f"\nGeneration completed in {time.time() - start:.1f}s")
    print("Saved to test_no_transpose_output.png")


if __name__ == "__main__":
    main()
