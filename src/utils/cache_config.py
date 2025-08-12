"""
Shared cache configuration for SAKURA_Reasoning project.
All programs should use this to access the shared cache directory at /work/.cache.
"""
import os

# Shared cache directory at root level
SHARED_CACHE_DIR = "/work/.cache"

def get_cache_dir():
    """Get the shared cache directory path as string."""
    return SHARED_CACHE_DIR

def set_hf_cache_env():
    """Set HuggingFace cache environment variables to use shared cache."""
    os.environ["HF_HOME"] = SHARED_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = SHARED_CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = SHARED_CACHE_DIR
    return SHARED_CACHE_DIR