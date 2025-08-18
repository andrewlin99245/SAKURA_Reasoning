#!/bin/bash

# Script to run HAL inference with Vector Steering enabled for different lambda values
# This script runs the original Qwen2Audio hal_inference.py with VSV enabled

echo "🚀 Starting HAL Inference with Vector Steering Experiments"
echo "=========================================================="

# Create results directory
mkdir -p ./hal_inference_results/

# Function to run inference with given lambda
run_inference() {
    local lambda=$1
    local output_suffix=$2
    
    echo ""
    echo "🎯 Running HAL Inference with VSV Lambda = $lambda"
    echo "------------------------------------------------"
    
    # Set output path with lambda suffix
    output_path="./hal_inference_results/hal_inference_vsv_lambda${output_suffix}.csv"
    
    echo "📂 Output will be saved to: $output_path"
    echo "⏰ Start time: $(date)"
    
    # Run the inference
    python src/models/hal_inference.py \
        --enable_vsv \
        --vsv_lambda $lambda \
        --dataset_file "./understanding_sound_data/metadata/balanced_merged_test_2871.txt" \
        --audio_root_dir "./understanding_sound_data/audio" \
        --output_path "$output_path" \
    
    # Check if inference completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Inference completed successfully for lambda = $lambda"
        echo "📊 Results saved to: $output_path"
        
        # Show file size and sample count
        if [ -f "$output_path" ]; then
            echo "📋 File size: $(du -h "$output_path" | cut -f1)"
            echo "📈 Sample count: $(tail -n +2 "$output_path" | wc -l)"
        fi
    else
        echo "❌ Inference failed for lambda = $lambda"
        return 1
    fi
    
    echo "🕒 End time: $(date)"
}

# Experiment 1: Lambda = 0.0 (No steering baseline)
echo "🔬 Experiment 1: VSV with Lambda = 0.0 (Baseline - No Steering)"
run_inference 0.0 "0.0"

# Experiment 2: Lambda = 0.05 (Standard steering)
echo ""
echo "🔬 Experiment 2: VSV with Lambda = 0.05 (Standard Steering)"
run_inference 0.05 "0.05"

echo ""
echo "🏁 All HAL Inference VSV experiments completed!"
echo "=============================================="

# Summary
echo ""
echo "📊 EXPERIMENT SUMMARY:"
echo "---------------------"
echo "Results are saved in ./hal_inference_results/"

for lambda in "0.0" "0.05"; do
    result_file="./hal_inference_results/hal_inference_vsv_lambda${lambda}.csv"
    if [ -f "$result_file" ]; then
        sample_count=$(tail -n +2 "$result_file" | wc -l)
        file_size=$(du -h "$result_file" | cut -f1)
        echo "✅ Lambda $lambda: $sample_count samples, $file_size"
    else
        echo "❌ Lambda $lambda: Results file not found"
    fi
done

echo ""
echo "🔍 Next Steps:"
echo "- Compare accuracy between lambda 0.0 and 0.05"
echo "- Analyze the effectiveness of vector steering"
echo "- Use analysis scripts to visualize the results"

echo ""
echo "🎯 Script completed at: $(date)"