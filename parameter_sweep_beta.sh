#!/bin/bash

echo "🔍 PARAMETER SWEEP: Beta Weight Grid Search"
echo "Testing 3x3x3 = 27 combinations of beta weights"
echo "Range: [0.05, 0.1, 0.2] for each beta"
echo ""

# Create results file
echo "beta1,beta2,beta3,accuracy,improvement,flips,flipped_pct" > parameter_sweep_results.csv

# Grid search over beta weights
for beta1 in 0.05 0.1 0.2; do
  for beta2 in 0.05 0.1 0.2; do
    for beta3 in 0.05 0.1 0.2; do
      echo "Testing: β1=$beta1, β2=$beta2, β3=$beta3"
      
      # Run evaluation
      python src/models/cosine_correlation_bos_with_lda.py \
        --bos_csv BOS_features_BOS_features.csv \
        --lda_stats BOS_features_BOS_stats_with_correlations.json \
        --corr_template correlation_template.json \
        --use_augmented \
        --flip_threshold 0.5 \
        --beta1 $beta1 --beta2 $beta2 --beta3 $beta3 \
        --output_dir ./ > /dev/null 2>&1
      
      # Extract results from JSON
      accuracy=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"augmented_accuracy\"]:.4f}')")
      improvement=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"improvement\"]:.4f}')")
      flips=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(data['flipped_count'])")
      flipped_pct=$(python -c "import json; data=json.load(open('lda_correction_results.json')); print(f'{data[\"flipped_count\"]/data[\"total_samples\"]*100:.2f}')")
      
      # Save to CSV
      echo "$beta1,$beta2,$beta3,$accuracy,$improvement,$flips,$flipped_pct" >> parameter_sweep_results.csv
      
      echo "  → Accuracy: $accuracy% (+$improvement pts, $flips flips)"
    done
  done
done

echo ""
echo "✅ Beta weight grid search completed!"
echo "📊 Results saved to: parameter_sweep_results.csv"