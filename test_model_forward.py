#!/usr/bin/env python3
"""Test the custom model forward method."""

import os
import sys
import torch

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src", "utils"))

print("Importing custom model...")
try:
    from Qwen2_5Omni_patch import Qwen2_5OmniSLAForCausalLM
    print("✓ Successfully imported Qwen2_5OmniSLAForCausalLM")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

print("Checking if forward method exists...")
if hasattr(Qwen2_5OmniSLAForCausalLM, 'forward'):
    print("✓ Forward method exists")
    print(f"Forward method: {Qwen2_5OmniSLAForCausalLM.forward}")
else:
    print("✗ Forward method missing")

print("Checking parent class...")
from transformers import Qwen2_5OmniForConditionalGeneration
if hasattr(Qwen2_5OmniForConditionalGeneration, 'forward'):
    print("✓ Parent class has forward method")
else:
    print("✗ Parent class missing forward method")

print("Testing method resolution order...")
print("MRO:", Qwen2_5OmniSLAForCausalLM.__mro__)

print("Checking all methods...")
methods = [method for method in dir(Qwen2_5OmniSLAForCausalLM) if not method.startswith('_')]
print("Available methods:", methods[:10])  # Show first 10 methods