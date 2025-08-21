#!/bin/bash

# First, find the best beta weights from previous sweep
if [ ! -f "parameter_sweep_results.csv" ]; then
    echo "❌ Run parameter_sweep_beta.sh first to find optimal beta weights"
    exit 1
fi

echo "🎯 PARAMETER SWEEP: Flip Threshold Optimization"
echo "Finding optimal threshold with best beta weights"
echo ""

# Extract best beta weights from previous results
best_line=$(tail -n +2 parameter_sweep_results.csv | sort -t',' -k5 -nr | head -1)
best_beta1=$(echo $best_line | cut -d',' -f1)
best_beta2=$(echo $best_line | cut -d',' -f2)
best_beta3=$(echo $best_line | cut -d',' -f3)
best_improvement=$(echo $best_line | cut -d',' -f5)

echo "📈 Using best beta weights: β1=$best_beta1, β2=$best_beta2, β3=$best_beta3"
echo "📈 Previous best improvement: +$best_improvement pts"
echo ""

# Create threshold results file
echo "threshold,accuracy,improvement,flips,flipped_pct,precision_estimate" > threshold_sweep_results.csv

# Test different thresholds
for threshold in 0.3 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8; do
  echo "Testing threshold: $threshold"
  
  # Run evaluation
  python src/models/cosine_correlation_bos_with_lda.py \
    --bos_csv BOS_features_BOS_features.csv \
    --lda_stats BOS_features_BOS_stats_with_correlations.json \
    --corr_template correlation_template.json \
    --use_augmented \
    --flip_threshold $threshold \
    --beta1 $best_beta1 --beta2 $best_beta2 --beta3 $best_beta3 \
    --output_dir ./ > /dev/null 2>&1
  
  # Extract results
  accuracy=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"augmented_accuracy\"]:.4f}')")
  improvement=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"improvement\"]:.4f}')")
  flips=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(data['flipped_count'])")
  flipped_pct=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"flipped_count\"]/data[\"total_samples\"]*100:.2f}')")
  
  # Estimate precision (improvement per flip)
  precision_est=$(python -c "print(f'{float('$improvement')/max(float('$flips'),1)*1000:.2f}')")
  
  # Save to CSV
  echo "$threshold,$accuracy,$improvement,$flips,$flipped_pct,$precision_est" >> threshold_sweep_results.csv
  
  echo "  → Accuracy: $accuracy% (+$improvement pts, $flips flips, precision~$precision_est)"
done

echo ""
echo "✅ Threshold optimization completed!"
echo "📊 Results saved to: threshold_sweep_results.csv"