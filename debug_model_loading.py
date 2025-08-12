#!/usr/bin/env python3
"""Debug script to isolate model loading issues."""

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

print("1. Importing transformers...")
try:
    from transformers import Qwen2_5OmniProcessor
    print("   ✓ Successfully imported Qwen2_5OmniProcessor")
except Exception as e:
    print(f"   ✗ Failed to import Qwen2_5OmniProcessor: {e}")
    sys.exit(1)

print("2. Importing custom patch...")
try:
    from Qwen2_5Omni_patch import Qwen2_5OmniSLAForCausalLM
    print("   ✓ Successfully imported Qwen2_5OmniSLAForCausalLM")
except Exception as e:
    print(f"   ✗ Failed to import Qwen2_5OmniSLAForCausalLM: {e}")
    sys.exit(1)

print("3. Checking CUDA availability...")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA devices: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name()}")

MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
print(f"4. Attempting to load processor from {MODEL_PATH}...")
try:
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    print("   ✓ Successfully loaded processor")
except Exception as e:
    print(f"   ✗ Failed to load processor: {e}")

print(f"5. Attempting to load model from {MODEL_PATH}...")
print("   This may take several minutes for the first time...")
try:
    model = Qwen2_5OmniSLAForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    print("   ✓ Successfully loaded model")
    print("   Model loaded successfully!")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()