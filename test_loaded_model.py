#!/usr/bin/env python3
"""Test the forward method on a loaded model."""

import os
import sys
import torch

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src", "utils"))
sys.path.insert(0, os.path.join(current_dir, "src", "models"))

# Configure shared cache
from cache_config import set_hf_cache_env
set_hf_cache_env()

print("Importing...")
from transformers import Qwen2_5OmniProcessor
from Qwen2_5Omni_patch import Qwen2_5OmniSLAForCausalLM

MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"

print("Loading processor...")
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = Qwen2_5OmniSLAForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

print("Model loaded successfully!")

print("Testing forward method with loaded model...")
# Create simple text-only inputs
input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(model.device)
attention_mask = torch.tensor([[1, 1, 1, 1, 1]]).to(model.device)

try:
    with torch.no_grad():
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    print("✓ Forward call succeeded with loaded model")
    print(f"Output type: {type(output)}")
except Exception as e:
    print(f"✗ Forward call failed even with loaded model: {e}")
    import traceback
    traceback.print_exc()