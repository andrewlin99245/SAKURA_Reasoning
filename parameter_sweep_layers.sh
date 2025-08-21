#!/bin/bash

echo "📊 PARAMETER SWEEP: Template Layer Count Optimization"
echo "Testing different numbers of layers in correlation template"
echo ""

# Create layer results file
echo "layers,accuracy,improvement,flips,flipped_pct" > layer_sweep_results.csv

# Test different layer counts
for layers in 4 6 8 10 12; do
  echo "Testing template with $layers layers..."
  
  # Generate new template with this layer count
  template_file="correlation_template_${layers}L.json"
  python src/models/cosine_correlation_bos_with_lda.py \
    --fit_corr_template_from BOS_features_BOS_features.csv \
    --corr_template_out $template_file \
    --template_last_k $layers > /dev/null 2>&1
  
  # Test with optimal beta weights (use 0.1, 0.1, 0.1 as reasonable default)
  python src/models/cosine_correlation_bos_with_lda.py \
    --bos_csv BOS_features_BOS_features.csv \
    --lda_stats BOS_features_BOS_stats_with_correlations.json \
    --corr_template $template_file \
    --use_augmented \
    --flip_threshold 0.5 \
    --beta1 0.1 --beta2 0.1 --beta3 0.1 \
    --output_dir ./ > /dev/null 2>&1
  
  # Extract results
  accuracy=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"augmented_accuracy\"]:.4f}')")
  improvement=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"improvement\"]:.4f}')")
  flips=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(data['flipped_count'])")
  flipped_pct=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"flipped_count\"]/data[\"total_samples\"]*100:.2f}')")
  
  # Save to CSV
  echo "$layers,$accuracy,$improvement,$flips,$flipped_pct" >> layer_sweep_results.csv
  
  echo "  → Accuracy: $accuracy% (+$improvement pts, $flips flips)"
done

echo ""
echo "✅ Layer count optimization completed!"
echo "📊 Results saved to: layer_sweep_results.csv"