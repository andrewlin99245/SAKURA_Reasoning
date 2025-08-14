"""
Shared cache configuration for SAKURA_Reasoning project.
All programs should use this to access the shared cache directory in the home directory.
"""
import os

# Shared cache directory in home directory
SHARED_CACHE_DIR = os.path.expanduser("~/.cache/sakura_reasoning")

def get_cache_dir():
    """Get the shared cache directory path as string."""
    return SHARED_CACHE_DIR

def set_hf_cache_env():
    """Set HuggingFace cache environment variables to use shared cache."""
    # Create cache directory if it doesn't exist
    os.makedirs(SHARED_CACHE_DIR, exist_ok=True)
    
    # Set all relevant HuggingFace cache environment variables
    os.environ["HF_HOME"] = SHARED_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = SHARED_CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = SHARED_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = SHARED_CACHE_DIR
    
    # Override XDG_CACHE_HOME to prevent conflicts with system settings
    os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.cache")
    
    # Set PyTorch cache to use our cache directory
    os.environ["TORCH_HOME"] = SHARED_CACHE_DIR
    
    # Also explicitly unset any conflicting cache variables
    if "PYTORCH_CACHE_HOME" in os.environ:
        del os.environ["PYTORCH_CACHE_HOME"]
    
    print(f"Cache directory set to: {SHARED_CACHE_DIR}")
    return SHARED_CACHE_DIR

def generate_env_setup_script():
    """Generate a shell script to set up environment variables."""
    script_content = f"""#!/bin/bash
# SAKURA Reasoning Cache Configuration
# Source this file to set up cache environment variables

export HF_HOME="{SHARED_CACHE_DIR}"
export TRANSFORMERS_CACHE="{SHARED_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="{SHARED_CACHE_DIR}"
export HF_HUB_CACHE="{SHARED_CACHE_DIR}"
export XDG_CACHE_HOME="{os.path.expanduser('~/.cache')}"
export TORCH_HOME="{SHARED_CACHE_DIR}"

# Unset conflicting variables
unset PYTORCH_CACHE_HOME

echo "Cache environment configured: {SHARED_CACHE_DIR}"
"""
    return script_content