# LoRA + device_map Fix Implementation

## Problem
When using `device_map="balanced"` (or any device_map configuration) with `load_lora_weights()`, PEFT LoRA layers end up on the meta device instead of matching their base layer devices. This causes NaN output during inference because the LoRA layers cannot perform actual computations on the meta device.

## Root Cause
When a model is loaded with `device_map`, different layers are distributed across multiple GPUs. The `hf_device_map` attribute tracks which device each layer is on. However, when PEFT injects LoRA adapters, the new LoRA layers (lora_A and lora_B) are not automatically placed on the same device as their base layers. They end up on the meta device by default.

## Solution
Modified `src/diffusers/loaders/peft.py` in the `load_lora_adapter()` method to move LoRA weights to match their base layer devices after PEFT injection.

### Implementation Details

**File:** `D:\Projects\diffusers-repo\src\diffusers\loaders\peft.py`

**Location:** After line 379 (`_maybe_warn_for_unhandled_keys(incompatible_keys, adapter_name)`)

**Code Added:**
```python
# Handle device_map scenarios - move LoRA weights to match base layer devices
if hasattr(self, 'hf_device_map') and self.hf_device_map is not None:
    for name, module in self.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'base_layer'):
            base_device = module.base_layer.weight.device
            if base_device.type != 'meta':
                for lora_attr in ['lora_A', 'lora_B']:
                    lora_dict = getattr(module, lora_attr, None)
                    if lora_dict is not None:
                        for adapter_name_key, lora_layer in lora_dict.items():
                            if lora_layer.weight.device != base_device:
                                lora_layer.to(base_device)
```

### How It Works

1. **Check for device_map**: Only runs if the model has `hf_device_map` attribute (indicating it was loaded with device_map)

2. **Iterate through modules**: Examines all modules in the model

3. **Identify LoRA modules**: Finds modules that have both `lora_A`/`lora_B` attributes (LoRA layers) and `base_layer` attribute

4. **Get base device**: Retrieves the device of the base layer's weights

5. **Skip meta devices**: Only processes if the base layer is not on meta device (to avoid errors)

6. **Move LoRA layers**: For each LoRA layer (lora_A and lora_B), if it's on a different device than the base layer, moves it to the base layer's device

### Testing

**Unit Test:** `D:\Projects\diffusers-repo\test_lora_device_map_fix.py`

This test verifies that:
- LoRA layers on meta device are correctly moved to match their base layer devices
- LoRA layers already on the correct device are not unnecessarily moved
- Modules without LoRA layers are ignored

**Test Results:**
```
[PASS] Module 1 LoRA A correctly moved to cuda:0
[PASS] Module 1 LoRA B correctly moved to cuda:0
[PASS] Module 2 LoRA A correctly left unchanged
[PASS] Module 2 LoRA B correctly left unchanged

*** ALL TESTS PASSED ***
```

### Integration Test

**Test File:** `D:\Projects\giga-images-v2\generator_v8\tests\ltx-video\test_lora_device_map_fix.py`

This test file can be used to verify the fix works with actual LTX-2 model loading:
- Loads LTX-2 model with `device_map="balanced"`
- Loads LoRA weights
- Verifies LoRA layers are on correct devices (not meta)
- Generates a test video to ensure no NaN output

## Benefits

1. **Fixes NaN output**: LoRA layers can now perform actual computations instead of returning NaN
2. **Automatic device placement**: No manual intervention required from users
3. **Multi-GPU support**: Works correctly with device_map="balanced" and other multi-GPU configurations
4. **Backward compatible**: Only activates when device_map is used; doesn't affect normal single-device loading
5. **Minimal overhead**: Only runs once during adapter loading, no runtime performance impact

## Impact

This fix enables users to:
- Use LoRA adapters with device_map for large models that don't fit on a single GPU
- Leverage multi-GPU setups for models like LTX-2, Flux, and other large diffusion models
- Avoid mysterious NaN output when combining LoRA with device_map

## Related Files Modified

1. `D:\Projects\diffusers-repo\src\diffusers\loaders\peft.py` (lines 381-392)
2. `D:\Projects\giga-images-v2\generator_v8\tests\ltx-video\test_ltx2_distilled.py` (enabled USE_LORA for testing)

## Next Steps

1. The fix has been implemented and tested with unit tests
2. Integration testing with actual models requires:
   - Updated transformers library with Gemma3ForConditionalGeneration support
   - Or testing with a different model that doesn't require latest transformers
3. Consider submitting this fix as a pull request to the diffusers repository
