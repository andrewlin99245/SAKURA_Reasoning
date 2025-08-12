#\!/usr/bin/env python3

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

# Test if the parent class forward method works
print("Testing parent class forward method...")

try:
    # Load the parent class directly
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype=torch.float16, device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    
    print("Model loaded successfully")
    print(f"Model class: {type(model)}")
    print(f"Has forward method: {hasattr(model, 'forward')}")
    
    # Try a simple forward pass
    text = "Hello world"
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("Input keys:", list(inputs.keys()))
    
    with torch.no_grad():
        output = model(**inputs)
    
    print("Forward pass successful\!")
    print(f"Output type: {type(output)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF < /dev/null
