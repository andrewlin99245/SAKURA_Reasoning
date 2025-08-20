#!/usr/bin/env python3
"""
Visualize inter-layer cosine correlations from analysis JSON files.

Usage:
    python visualize_inter_layer_correlations.py analysis_file.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_analysis_data(json_path):
    """Load analysis results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_correlation_matrices(correlation_data):
    """Extract correlation matrices from the analysis data"""
    correct_correlations = correlation_data.get('correct_correlations', {})
    incorrect_correlations = correlation_data.get('incorrect_correlations', {})
    
    # Get all unique layer IDs
    all_layers = set()
    for pair_key in correct_correlations.keys():
        layer1, layer2 = pair_key.split('-')
        all_layers.update([int(layer1), int(layer2)])
    
    for pair_key in incorrect_correlations.keys():
        layer1, layer2 = pair_key.split('-')
        all_layers.update([int(layer1), int(layer2)])
    
    all_layers = sorted(list(all_layers))
    n_layers = len(all_layers)
    
    if n_layers == 0:
        return None, None, []
    
    # Create correlation matrices
    correct_matrix = np.eye(n_layers)  # Diagonal is 1 (self-correlation)
    incorrect_matrix = np.eye(n_layers)
    
    layer_to_idx = {layer: idx for idx, layer in enumerate(all_layers)}
    
    # Fill in the correlation values
    for pair_key, stats in correct_correlations.items():
        layer1, layer2 = pair_key.split('-')
        idx1, idx2 = layer_to_idx[int(layer1)], layer_to_idx[int(layer2)]
        corr = stats['correlation']
        correct_matrix[idx1, idx2] = corr
        correct_matrix[idx2, idx1] = corr  # Symmetric
    
    for pair_key, stats in incorrect_correlations.items():
        layer1, layer2 = pair_key.split('-')
        idx1, idx2 = layer_to_idx[int(layer1)], layer_to_idx[int(layer2)]
        corr = stats['correlation']
        incorrect_matrix[idx1, idx2] = corr
        incorrect_matrix[idx2, idx1] = corr  # Symmetric
    
    return correct_matrix, incorrect_matrix, all_layers

def create_correlation_heatmaps(correct_matrix, incorrect_matrix, layer_ids, output_dir):
    """Create side-by-side correlation heatmaps"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common settings
    vmin, vmax = -1, 1
    cmap = 'RdBu_r'  # Red for positive, Blue for negative
    
    # Correct predictions heatmap
    sns.heatmap(correct_matrix, 
                xticklabels=layer_ids, 
                yticklabels=layer_ids,
                cmap=cmap, 
                vmin=vmin, 
                vmax=vmax,
                center=0,
                annot=False,
                cbar_kws={'label': 'Correlation'},
                ax=ax1)
    ax1.set_title('Correct Predictions\nInter-layer Correlations')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Layer')
    
    # Incorrect predictions heatmap
    sns.heatmap(incorrect_matrix, 
                xticklabels=layer_ids, 
                yticklabels=layer_ids,
                cmap=cmap, 
                vmin=vmin, 
                vmax=vmax,
                center=0,
                annot=False,
                cbar_kws={'label': 'Correlation'},
                ax=ax2)
    ax2.set_title('Incorrect Predictions\nInter-layer Correlations')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Layer')
    
    # Difference heatmap (Correct - Incorrect)
    diff_matrix = correct_matrix - incorrect_matrix
    sns.heatmap(diff_matrix, 
                xticklabels=layer_ids, 
                yticklabels=layer_ids,
                cmap='RdBu_r', 
                center=0,
                annot=False,
                cbar_kws={'label': 'Correlation Difference'},
                ax=ax3)
    ax3.set_title('Difference\n(Correct - Incorrect)')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Layer')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'inter_layer_correlation_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved correlation heatmaps: {output_path}")
    
    return fig

def create_correlation_comparison_plot(correlation_comparison, output_dir):
    """Create scatter plot comparing correlations for correct vs incorrect predictions"""
    if not correlation_comparison:
        print("⚠️  No correlation comparison data available")
        return None
    
    # Extract data for plotting
    layer_pairs = []
    correct_corrs = []
    incorrect_corrs = []
    differences = []
    
    for pair_key, stats in correlation_comparison.items():
        layer_pairs.append(pair_key)
        correct_corrs.append(stats['correct_correlation'])
        incorrect_corrs.append(stats['incorrect_correlation'])
        differences.append(stats['correlation_difference'])
    
    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot: Correct vs Incorrect correlations
    scatter = ax1.scatter(correct_corrs, incorrect_corrs, 
                         c=differences, cmap='RdBu_r', 
                         s=100, alpha=0.7)
    
    # Add diagonal line (y=x) for reference
    min_val = min(min(correct_corrs), min(incorrect_corrs))
    max_val = max(max(correct_corrs), max(incorrect_corrs))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax1.set_xlabel('Correlation (Correct Predictions)')
    ax1.set_ylabel('Correlation (Incorrect Predictions)')
    ax1.set_title('Inter-layer Correlations:\nCorrect vs Incorrect Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Correlation Difference\n(Correct - Incorrect)')
    
    # Bar plot of correlation differences
    sorted_indices = np.argsort(np.abs(differences))[::-1]  # Sort by absolute difference
    sorted_pairs = [layer_pairs[i] for i in sorted_indices]
    sorted_diffs = [differences[i] for i in sorted_indices]
    
    colors = ['red' if d > 0 else 'blue' for d in sorted_diffs]
    bars = ax2.bar(range(len(sorted_diffs)), sorted_diffs, color=colors, alpha=0.7)
    ax2.set_xlabel('Layer Pairs (sorted by |difference|)')
    ax2.set_ylabel('Correlation Difference\n(Correct - Incorrect)')
    ax2.set_title('Correlation Differences by Layer Pair')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax2.set_xticks(range(len(sorted_pairs)))
    ax2.set_xticklabels(sorted_pairs, rotation=45, ha='right')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Higher when Correct'),
                      Patch(facecolor='blue', alpha=0.7, label='Higher when Incorrect')]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'correlation_comparison_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved correlation comparison plot: {output_path}")
    
    return fig

def create_summary_statistics_plot(correlation_data, output_dir):
    """Create summary statistics visualization"""
    summary = correlation_data.get('summary', {})
    if not summary:
        print("⚠️  No summary data available")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sample sizes
    correct_samples = summary.get('num_correct_samples', 0)
    incorrect_samples = summary.get('num_incorrect_samples', 0)
    ax1.bar(['Correct', 'Incorrect'], [correct_samples, incorrect_samples], 
            color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Sizes by Prediction Accuracy')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    ax1.text(0, correct_samples + max(correct_samples, incorrect_samples)*0.02, 
             str(correct_samples), ha='center', va='bottom')
    ax1.text(1, incorrect_samples + max(correct_samples, incorrect_samples)*0.02, 
             str(incorrect_samples), ha='center', va='bottom')
    
    # 2. Mean correlation difference
    mean_diff = summary.get('mean_correlation_difference', 0)
    std_diff = summary.get('std_correlation_difference', 0)
    
    ax2.bar(['Mean Difference'], [mean_diff], yerr=[std_diff], 
            color='purple', alpha=0.7, capsize=5)
    ax2.set_ylabel('Correlation Difference')
    ax2.set_title('Mean Correlation Difference\n(Correct - Incorrect)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add text annotation
    ax2.text(0, mean_diff + (std_diff if mean_diff >= 0 else -std_diff) + 0.01, 
             f'{mean_diff:.3f}±{std_diff:.3f}', ha='center', va='bottom' if mean_diff >= 0 else 'top')
    
    # 3. Range of correlation differences
    max_diff = summary.get('max_correlation_difference', 0)
    min_diff = summary.get('min_correlation_difference', 0)
    
    ax3.bar(['Min', 'Max'], [min_diff, max_diff], 
            color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Correlation Difference')
    ax3.set_title('Range of Correlation Differences')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    ax3.text(0, min_diff - 0.01, f'{min_diff:.3f}', ha='center', va='top')
    ax3.text(1, max_diff + 0.01, f'{max_diff:.3f}', ha='center', va='bottom')
    
    # 4. Number of layer pairs analyzed
    num_pairs = summary.get('num_layer_pairs', 0)
    ax4.pie([num_pairs, max(0, 100-num_pairs)], labels=[f'{num_pairs} Pairs\nAnalyzed', ''], 
            colors=['lightblue', 'lightgray'], startangle=90, 
            wedgeprops=dict(width=0.5))
    ax4.set_title('Layer Pairs Analyzed')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'summary_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved summary statistics: {output_path}")
    
    return fig

def create_all_visualizations(json_path):
    """Create all visualizations from analysis JSON file"""
    print(f"📊 Loading analysis data from: {json_path}")
    
    # Load data
    try:
        data = load_analysis_data(json_path)
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return
    
    # Check if inter-layer correlation data exists
    correlation_data = data.get('inter_layer_correlation_analysis', {})
    if not correlation_data:
        print("❌ No inter-layer correlation analysis found in JSON file")
        return
    
    # Create output directory
    json_file = Path(json_path)
    output_dir = json_file.parent / f"{json_file.stem}_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Creating visualizations in: {output_dir}")
    
    # Extract correlation matrices and create heatmaps
    correct_matrix, incorrect_matrix, layer_ids = extract_correlation_matrices(correlation_data)
    
    if correct_matrix is not None and incorrect_matrix is not None:
        print("🎨 Creating correlation heatmaps...")
        create_correlation_heatmaps(correct_matrix, incorrect_matrix, layer_ids, output_dir)
    else:
        print("⚠️  Could not extract correlation matrices")
    
    # Create comparison scatter plot
    correlation_comparison = correlation_data.get('correlation_comparison', {})
    if correlation_comparison:
        print("🎨 Creating correlation comparison plots...")
        create_correlation_comparison_plot(correlation_comparison, output_dir)
    
    # Create summary statistics plot
    print("🎨 Creating summary statistics plots...")
    create_summary_statistics_plot(correlation_data, output_dir)
    
    # Print summary information
    summary = correlation_data.get('summary', {})
    if summary:
        print(f"\n📈 Analysis Summary:")
        print(f"  • Layer pairs analyzed: {summary.get('num_layer_pairs', 0)}")
        print(f"  • Correct prediction samples: {summary.get('num_correct_samples', 0)}")
        print(f"  • Incorrect prediction samples: {summary.get('num_incorrect_samples', 0)}")
        print(f"  • Mean correlation difference: {summary.get('mean_correlation_difference', 0):.3f}")
        
        # Interpretation
        mean_diff = summary.get('mean_correlation_difference', 0)
        if mean_diff > 0.1:
            print(f"  💡 Insight: Inter-layer correlations are generally HIGHER for correct predictions")
        elif mean_diff < -0.1:
            print(f"  💡 Insight: Inter-layer correlations are generally HIGHER for incorrect predictions")
        else:
            print(f"  💡 Insight: Inter-layer correlations are similar for correct and incorrect predictions")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize inter-layer cosine correlations from analysis JSON")
    parser.add_argument("json_file", type=str, help="Path to the analysis JSON file")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"❌ File not found: {args.json_file}")
        return
    
    create_all_visualizations(args.json_file)

if __name__ == "__main__":
    main()