#!/usr/bin/env python3
"""Test the original model without our custom class."""

import torch
import os
import sys
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Add cache config
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src', 'utils'))
from cache_config import set_hf_cache_env

# Configure shared cache
set_hf_cache_env()

MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"

print("Loading processor...")
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

print("Loading original model (not our custom class)...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

print("Model loaded successfully!")

print("Testing forward method with original model...")
# Create simple text-only inputs
input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(model.device)
attention_mask = torch.tensor([[1, 1, 1, 1, 1]]).to(model.device)

try:
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
    print("✓ Original model call succeeded")
    print(f"Output type: {type(output)}")
    print(f"Output has logits: {hasattr(output, 'logits')}")
    if hasattr(output, 'logits'):
        print(f"Logits shape: {output.logits.shape}")
except Exception as e:
    print(f"✗ Original model call failed: {e}")
    import traceback
    traceback.print_exc()