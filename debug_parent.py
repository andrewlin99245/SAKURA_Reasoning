#!/usr/bin/env python3
"""Debug the parent class forward method."""

import os
import sys
import torch

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src", "utils"))

# Configure shared cache
from cache_config import set_hf_cache_env
set_hf_cache_env()

from transformers import Qwen2_5OmniForConditionalGeneration, AutoConfig

print("Testing parent class...")
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Omni-3B")

# Test the parent class directly
print("Creating parent class instance...")
parent_model = Qwen2_5OmniForConditionalGeneration(config)
print("✓ Parent model created")

# Test calling forward on parent
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

print("Testing parent forward method...")
try:
    with torch.no_grad():
        output = parent_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    print("✓ Parent forward works")
    print(f"Output type: {type(output)}")
except Exception as e:
    print(f"✗ Parent forward failed: {e}")
    import traceback
    traceback.print_exc()

print("\nChecking parent class forward method...")
print(f"Parent forward method: {parent_model.forward}")
print(f"Parent forward callable: {callable(parent_model.forward)}")

print("\nTesting method resolution...")
print("Parent class MRO:", Qwen2_5OmniForConditionalGeneration.__mro__)