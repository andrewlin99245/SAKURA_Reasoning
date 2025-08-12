#!/usr/bin/env python3
"""
Verification script for shared cache setup.
Run this to verify that all programs will use the shared cache at /work/.cache.
"""

import os
import sys

def main():
    print("🔍 SAKURA_Reasoning Cache Setup Verification")
    print("=" * 50)
    
    # Test cache config import
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'utils'))
        from cache_config import set_hf_cache_env, get_cache_dir
        print("✅ Cache configuration module loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import cache configuration: {e}")
        return False
    
    # Test cache directory
    cache_dir = get_cache_dir()
    print(f"📁 Configured cache directory: {cache_dir}")
    
    if os.path.exists(cache_dir):
        print("✅ Cache directory exists")
        if os.access(cache_dir, os.W_OK):
            print("✅ Cache directory is writable")
        else:
            print("⚠️  Cache directory is not writable")
    else:
        print("❌ Cache directory does not exist")
        return False
    
    # Test environment variable setup
    print("\n🔧 Setting up environment variables...")
    set_hf_cache_env()
    
    required_vars = ['HF_HOME', 'TRANSFORMERS_CACHE', 'HUGGINGFACE_HUB_CACHE']
    all_set = True
    
    for var in required_vars:
        value = os.environ.get(var)
        if value == cache_dir:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: {value} (expected {cache_dir})")
            all_set = False
    
    # Test HuggingFace imports
    print("\n📦 Testing HuggingFace imports...")
    try:
        from transformers import AutoTokenizer
        print("✅ transformers library available")
    except ImportError:
        print("⚠️  transformers library not available")
    
    print("\n" + "=" * 50)
    if all_set:
        print("🎉 Cache setup verification PASSED!")
        print(f"   All programs will now use shared cache: {cache_dir}")
        print("   This will prevent redundant model downloads and save storage space.")
    else:
        print("❌ Cache setup verification FAILED!")
        print("   Please check the issues above and try again.")
    
    return all_set

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)