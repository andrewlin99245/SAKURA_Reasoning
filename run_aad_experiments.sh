#!/bin/bash

# Audio-Aware Decoding Experiments Script
# This script runs AAD experiments with different VSV lambda values

set -e  # Exit on any error

echo "🚀 Starting Audio-Aware Decoding Experiments"
echo "=================================================="

# Configuration
SCRIPT_PATH="src/models/hal_inference_aad.py"
AAD_ALPHA=1.0
DATASET_FILE="./understanding_sound_data/metadata/balanced_merged_test_2871.txt"
AUDIO_ROOT="./understanding_sound_data/audio"
BASE_OUTPUT="./aad_experiments"

# Create output directory if it doesn't exist
mkdir -p $BASE_OUTPUT

# Experiment 1: AAD with alpha=1.0, VSV disabled (lambda=0.0)
echo ""
echo "🧪 Experiment 1: AAD (α=1.0) + VSV disabled"
echo "---------------------------------------------"
OUTPUT_FILE_1="${BASE_OUTPUT}/aad_alpha1.0_vsv_disabled.csv"

python $SCRIPT_PATH \
    --enable_aad \
    --aad_alpha $AAD_ALPHA \
    --vsv_lambda 0.0 \
    --dataset_file "$DATASET_FILE" \
    --audio_root_dir "$AUDIO_ROOT" \
    --output_path "$OUTPUT_FILE_1" \
    --verbose

echo "✅ Experiment 1 completed. Results saved to: $OUTPUT_FILE_1"

# Experiment 2: AAD with alpha=1.0, VSV enabled (lambda=0.05)  
echo ""
echo "🧪 Experiment 2: AAD (α=1.0) + VSV enabled (λ=0.05)"
echo "---------------------------------------------------"
OUTPUT_FILE_2="${BASE_OUTPUT}/aad_alpha1.0_vsv_lambda0.05.csv"

python $SCRIPT_PATH \
    --enable_aad \
    --aad_alpha $AAD_ALPHA \
    --enable_vsv \
    --vsv_lambda 0.05 \
    --dataset_file "$DATASET_FILE" \
    --audio_root_dir "$AUDIO_ROOT" \
    --output_path "$OUTPUT_FILE_2" \
    --verbose

echo "✅ Experiment 2 completed. Results saved to: $OUTPUT_FILE_2"

# Summary
echo ""
echo "🏁 All experiments completed!"
echo "================================"
echo "Results saved in: $BASE_OUTPUT/"
echo "Files generated:"
echo "  - $(basename $OUTPUT_FILE_1)"
echo "  - $(basename $OUTPUT_FILE_2)"
echo ""
echo "To compare results, check the accuracy in the final output logs above."