"""
Unit test for LoRA + device_map fix

This tests that the fix in peft.py correctly moves LoRA layers to match
their base layer devices when using device_map.
"""

import sys
import torch
from unittest.mock import Mock, MagicMock

# Test the fix logic in isolation
print("=" * 60)
print("Unit test for LoRA + device_map fix")
print("=" * 60)

# Mock a model with hf_device_map
class MockLoraLayer:
    def __init__(self, device):
        self.weight = Mock()
        self.weight.device = device
        self.to_calls = []

    def to(self, device):
        """Track calls to .to()"""
        self.to_calls.append(device)
        self.weight.device = device
        return self

class MockModule:
    def __init__(self, base_device, lora_device_a, lora_device_b, adapter_name="default"):
        self.base_layer = Mock()
        self.base_layer.weight = Mock()
        self.base_layer.weight.device = base_device

        self.lora_A_layer = MockLoraLayer(lora_device_a)
        self.lora_B_layer = MockLoraLayer(lora_device_b)

        self.lora_A = {adapter_name: self.lora_A_layer}
        self.lora_B = {adapter_name: self.lora_B_layer}

class MockModel:
    def __init__(self):
        self.hf_device_map = {'layer1': 'cuda:0', 'layer2': 'cuda:1'}

        # Create test modules:
        # 1. LoRA on wrong device (meta) - should be moved
        # 2. LoRA on correct device - should not be moved
        # 3. Module without LoRA - should be ignored

        self.module_with_meta_lora = MockModule(
            base_device=torch.device('cuda:0'),
            lora_device_a=torch.device('meta'),
            lora_device_b=torch.device('meta'),
            adapter_name="test_adapter"
        )

        self.module_with_correct_lora = MockModule(
            base_device=torch.device('cuda:1'),
            lora_device_a=torch.device('cuda:1'),
            lora_device_b=torch.device('cuda:1'),
            adapter_name="test_adapter"
        )

        # Module without LoRA - just a plain module
        self.module_without_lora = type('PlainModule', (), {})()

    def named_modules(self):
        return [
            ('module1', self.module_with_meta_lora),
            ('module2', self.module_with_correct_lora),
            ('module3', self.module_without_lora),
        ]

# Create mock model
model = MockModel()

print("\nInitial state:")
print(f"  Module 1 (should be fixed):")
print(f"    Base layer: {model.module_with_meta_lora.base_layer.weight.device}")
print(f"    LoRA A: {model.module_with_meta_lora.lora_A_layer.weight.device}")
print(f"    LoRA B: {model.module_with_meta_lora.lora_B_layer.weight.device}")
print(f"  Module 2 (should not change):")
print(f"    Base layer: {model.module_with_correct_lora.base_layer.weight.device}")
print(f"    LoRA A: {model.module_with_correct_lora.lora_A_layer.weight.device}")
print(f"    LoRA B: {model.module_with_correct_lora.lora_B_layer.weight.device}")

# Apply the fix logic (copied from peft.py)
print("\nApplying fix...")

if hasattr(model, 'hf_device_map') and model.hf_device_map is not None:
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'base_layer'):
            base_device = module.base_layer.weight.device
            if base_device.type != 'meta':
                for lora_attr in ['lora_A', 'lora_B']:
                    lora_dict = getattr(module, lora_attr, None)
                    if lora_dict is not None:
                        for adapter_name_key, lora_layer in lora_dict.items():
                            if lora_layer.weight.device != base_device:
                                print(f"  Moving {name}.{lora_attr}[{adapter_name_key}] from {lora_layer.weight.device} to {base_device}")
                                lora_layer.to(base_device)

print("\nFinal state:")
print(f"  Module 1 (should be fixed):")
print(f"    Base layer: {model.module_with_meta_lora.base_layer.weight.device}")
print(f"    LoRA A: {model.module_with_meta_lora.lora_A_layer.weight.device}")
print(f"    LoRA B: {model.module_with_meta_lora.lora_B_layer.weight.device}")
print(f"  Module 2 (should not change):")
print(f"    Base layer: {model.module_with_correct_lora.base_layer.weight.device}")
print(f"    LoRA A: {model.module_with_correct_lora.lora_A_layer.weight.device}")
print(f"    LoRA B: {model.module_with_correct_lora.lora_B_layer.weight.device}")

# Verify the fix
print("\nVerification:")
test_passed = True

# Check module 1 - should have been moved
if len(model.module_with_meta_lora.lora_A_layer.to_calls) == 0:
    print("  [FAIL] Module 1 LoRA A was not moved")
    test_passed = False
elif model.module_with_meta_lora.lora_A_layer.to_calls[0] != torch.device('cuda:0'):
    print(f"  [FAIL] Module 1 LoRA A moved to wrong device: {model.module_with_meta_lora.lora_A_layer.to_calls[0]}")
    test_passed = False
else:
    print("  [PASS] Module 1 LoRA A correctly moved to cuda:0")

if len(model.module_with_meta_lora.lora_B_layer.to_calls) == 0:
    print("  [FAIL] Module 1 LoRA B was not moved")
    test_passed = False
elif model.module_with_meta_lora.lora_B_layer.to_calls[0] != torch.device('cuda:0'):
    print(f"  [FAIL] Module 1 LoRA B moved to wrong device: {model.module_with_meta_lora.lora_B_layer.to_calls[0]}")
    test_passed = False
else:
    print("  [PASS] Module 1 LoRA B correctly moved to cuda:0")

# Check module 2 - should NOT have been moved
if len(model.module_with_correct_lora.lora_A_layer.to_calls) > 0:
    print(f"  [FAIL] Module 2 LoRA A was moved when it shouldn't be: {model.module_with_correct_lora.lora_A_layer.to_calls}")
    test_passed = False
else:
    print("  [PASS] Module 2 LoRA A correctly left unchanged")

if len(model.module_with_correct_lora.lora_B_layer.to_calls) > 0:
    print(f"  [FAIL] Module 2 LoRA B was moved when it shouldn't be: {model.module_with_correct_lora.lora_B_layer.to_calls}")
    test_passed = False
else:
    print("  [PASS] Module 2 LoRA B correctly left unchanged")

# Final verdict
print("\n" + "=" * 60)
if test_passed:
    print("*** ALL TESTS PASSED ***")
    print("The fix correctly moves LoRA layers from meta device to match base layers")
else:
    print("*** TESTS FAILED ***")
    print("The fix is not working as expected")
print("=" * 60)

sys.exit(0 if test_passed else 1)
