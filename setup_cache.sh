#!/bin/bash
# SAKURA Reasoning Cache Configuration
# Source this file to set up cache environment variables

export HF_HOME="$HOME/.cache/sakura_reasoning"
export TRANSFORMERS_CACHE="$HOME/.cache/sakura_reasoning"
export HUGGINGFACE_HUB_CACHE="$HOME/.cache/sakura_reasoning"
export HF_HUB_CACHE="$HOME/.cache/sakura_reasoning"
export XDG_CACHE_HOME="$HOME/.cache"
export TORCH_HOME="$HOME/.cache/sakura_reasoning"

# Unset conflicting variables
unset PYTORCH_CACHE_HOME

# Create cache directory if it doesn't exist
mkdir -p "$HOME/.cache/sakura_reasoning"

echo "Cache environment configured: $HOME/.cache/sakura_reasoning"