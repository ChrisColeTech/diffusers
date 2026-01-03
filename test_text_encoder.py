"""Test text encoder output."""
import torch

GGUF_PATH = r"D:\models\image-models\flux2-dev\combined\flux2-dev-ministral3b-q6k-v2.gguf"
CONFIG_DIR = r"D:\Projects\giga-images-v2\generator-api-v3\configs\flux2_pipeline"
TOKENIZER_PATH = r"D:\models\image-models\flux2-dev\split\text_encoders\tokenizers"
DEVICE = "cuda:0"
COMPUTE_DTYPE = torch.bfloat16

def main():
    from diffusers.loaders import CombinedGGUFLoader
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    from diffusers.utils import get_module_from_name
    from diffusers.loaders.single_file_utils import convert_ministral3_checkpoint_to_diffusers
    from accelerate.utils import set_module_tensor_to_device
    from accelerate import init_empty_weights
    from transformers import Mistral3ForConditionalGeneration, AutoConfig, PreTrainedTokenizerFast

    print("Loading GGUF...")
    loader = CombinedGGUFLoader(GGUF_PATH)
    state_dicts = loader.load()

    print("Converting text encoder state dict...")
    te_state = convert_ministral3_checkpoint_to_diffusers(state_dicts["text_encoder"])
    for k in ["model.vision_tower.image_break_token", "text_projection.bias", "text_projection.weight"]:
        te_state.pop(k, None)

    print("Creating text encoder...")
    config = AutoConfig.from_pretrained(f"{CONFIG_DIR}/text_encoder")
    with init_empty_weights():
        text_encoder = Mistral3ForConditionalGeneration(config)

    print("Loading weights...")
    quantizer = GGUFQuantizer(GGUFQuantizationConfig(compute_dtype=COMPUTE_DTYPE))
    quantizer.validate_environment()
    quantizer._process_model_before_weight_loading(text_encoder, device_map=None, state_dict=te_state)

    param_names = list(te_state.keys())
    for param_name in param_names:
        param_value = te_state.pop(param_name)
        is_quantized = quantizer.check_if_quantized_param(text_encoder, param_value, param_name, {})

        if is_quantized:
            module, tensor_name = get_module_from_name(text_encoder, param_name)
            current_param = getattr(module, tensor_name, None)
            if current_param is not None:
                quantizer.check_quantized_param_shape(param_name, current_param, param_value)
            quantizer.create_quantized_param(text_encoder, param_value, param_name, DEVICE, {})
        else:
            if param_value.is_floating_point():
                param_value = param_value.to(COMPUTE_DTYPE)
            set_module_tensor_to_device(text_encoder, param_name, DEVICE, param_value)

    # Convert norm layers
    for name, module in text_encoder.named_modules():
        if 'norm' in name.lower():
            module.to(COMPUTE_DTYPE)

    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{TOKENIZER_PATH}/tokenizer.json",
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )
    tokenizer.chat_template = "{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{eos_token}}{% endif %}{% endfor %}"

    print("\nTesting text encoding...")
    prompt = "a beautiful sunset over mountains"

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        [messages],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input tokens: {tokenizer.decode(input_ids[0])}")

    # Forward pass
    text_encoder.eval()
    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    print(f"\nNumber of hidden states: {len(output.hidden_states)}")
    print(f"Last hidden state shape: {output.hidden_states[-1].shape}")

    # Check layers 5, 10, 15, 20, 25
    for layer_idx in [5, 10, 15, 20, 25]:
        hs = output.hidden_states[layer_idx]
        print(f"Layer {layer_idx}: shape={hs.shape}, min={hs.min():.3f}, max={hs.max():.3f}, mean={hs.mean():.3f}, std={hs.std():.3f}")

    # Stack like pipeline does
    hidden_states_layers = (5, 10, 15, 20, 25)
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    print(f"\nStacked hidden states shape: {out.shape}")

    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    print(f"Final prompt embeds shape: {prompt_embeds.shape}")
    print(f"Prompt embeds stats: min={prompt_embeds.min():.3f}, max={prompt_embeds.max():.3f}, mean={prompt_embeds.mean():.3f}")

if __name__ == "__main__":
    main()
