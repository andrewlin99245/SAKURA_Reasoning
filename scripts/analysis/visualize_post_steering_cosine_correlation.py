#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import seaborn as sns
from pathlib import Path
import argparse

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_data(csv_path, json_path):
    """Load both CSV and JSON data"""
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Load JSON analysis data
    with open(json_path, 'r') as f:
        analysis_data = json.load(f)
    
    return df, analysis_data

def plot_accuracy_overview(df, analysis_data):
    """Plot overall accuracy and basic statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall accuracy
    total_samples = len(df)
    correct_samples = df['correct'].sum()
    accuracy = correct_samples / total_samples * 100
    
    ax1.bar(['Correct', 'Incorrect'], [correct_samples, total_samples - correct_samples], 
            color=['green', 'red'], alpha=0.7)
    ax1.set_title(f'Overall Accuracy: {accuracy:.2f}%\n({correct_samples}/{total_samples})')
    ax1.set_ylabel('Number of Samples')
    
    # Distribution of mean cosine similarities
    ax2.hist(df['mean_cosine'].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Distribution of Mean Cosine Similarities (Post-Steering)')
    ax2.set_xlabel('Mean Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df['mean_cosine'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["mean_cosine"].mean():.4f}')
    ax2.legend()
    
    # Cosine similarity by correctness
    correct_cosines = df[df['correct'] == True]['mean_cosine'].dropna()
    incorrect_cosines = df[df['correct'] == False]['mean_cosine'].dropna()
    
    ax3.hist([correct_cosines, incorrect_cosines], bins=20, alpha=0.7, 
             label=['Correct', 'Incorrect'], color=['green', 'red'])
    ax3.set_title('Cosine Similarity Distribution by Correctness')
    ax3.set_xlabel('Mean Cosine Similarity')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Box plot comparison
    data_for_box = [correct_cosines, incorrect_cosines]
    ax4.boxplot(data_for_box, labels=['Correct', 'Incorrect'])
    ax4.set_title('Cosine Similarity: Correct vs Incorrect')
    ax4.set_ylabel('Mean Cosine Similarity')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    if len(correct_cosines) > 0 and len(incorrect_cosines) > 0:
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(correct_cosines, incorrect_cosines)
        ax4.text(0.5, 0.95, f'p-value: {p_value:.4f}', transform=ax4.transAxes, 
                ha='center', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    return fig

def plot_layer_wise_analysis(analysis_data):
    """Plot layer-wise cosine similarity analysis"""
    layer_analysis = analysis_data.get('layer_wise_analysis', {})
    
    if not layer_analysis:
        print("No layer-wise analysis data found")
        return None
    
    # Extract data for plotting
    layers = sorted([int(k) for k in layer_analysis.keys()])
    correct_means = []
    incorrect_means = []
    differences = []
    p_values = []
    effect_sizes = []
    significant_layers = []
    
    for layer in layers:
        layer_data = layer_analysis[str(layer)]
        correct_means.append(layer_data['correct_predictions']['mean_cosine'])
        incorrect_means.append(layer_data['incorrect_predictions']['mean_cosine'])
        differences.append(layer_data['comparison']['mean_difference'])
        p_values.append(layer_data['comparison']['p_value'])
        effect_sizes.append(layer_data['comparison']['cohens_d'])
        significant_layers.append(layer_data['comparison']['significant'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Mean cosine similarities by layer
    x = np.array(layers)
    width = 0.35
    ax1.bar(x - width/2, correct_means, width, label='Correct', alpha=0.8, color='green')
    ax1.bar(x + width/2, incorrect_means, width, label='Incorrect', alpha=0.8, color='red')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Cosine Similarity (Post-Steering)')
    ax1.set_title('Layer-wise Post-Steering Cosine Similarities: Correct vs Incorrect')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers[::2])  # Show every other layer to avoid crowding
    
    # Differences between correct and incorrect
    colors = ['red' if sig else 'gray' for sig in significant_layers]
    bars = ax2.bar(layers, differences, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Difference (Correct - Incorrect)')
    ax2.set_title('Post-Steering Cosine Similarity Differences by Layer\n(Red = Significant, p < 0.05)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers[::2])
    
    # Highlight significant layers
    for i, (layer, sig) in enumerate(zip(layers, significant_layers)):
        if sig:
            ax2.text(layer, differences[i], f'{differences[i]:.3f}', 
                    ha='center', va='bottom' if differences[i] > 0 else 'top', fontsize=8)
    
    # P-values
    ax3.bar(layers, [-np.log10(p) for p in p_values], 
            color=['red' if sig else 'gray' for sig in significant_layers], alpha=0.7)
    ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p = 0.05 threshold')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('-log10(p-value)')
    ax3.set_title('Statistical Significance by Layer')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(layers[::2])
    
    # Effect sizes (Cohen's d)
    ax4.bar(layers, effect_sizes, 
            color=['red' if sig else 'gray' for sig in significant_layers], alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect')
    ax4.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Medium effect') 
    ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel("Cohen's d")
    ax4.set_title('Effect Size by Layer')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(layers[::2])
    
    plt.tight_layout()
    return fig

def plot_cosine_similarity_patterns(df):
    """Plot detailed cosine similarity patterns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scatter plot: Mean vs Std cosine similarity
    correct_mask = df['correct'] == True
    ax1.scatter(df[correct_mask]['mean_cosine'], df[correct_mask]['std_cosine'], 
               alpha=0.6, label='Correct', color='green', s=30)
    ax1.scatter(df[~correct_mask]['mean_cosine'], df[~correct_mask]['std_cosine'], 
               alpha=0.6, label='Incorrect', color='red', s=30)
    ax1.set_xlabel('Mean Cosine Similarity')
    ax1.set_ylabel('Std Cosine Similarity')
    ax1.set_title('Mean vs Std Cosine Similarity (Post-Steering)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution of cosine count
    ax2.hist(df['cosine_count'].dropna(), bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax2.set_xlabel('Number of Cosine Measurements')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Measurement Counts per Sample')
    ax2.grid(True, alpha=0.3)
    
    # Correlation between mean cosine and accuracy
    # Create bins for mean cosine similarity
    df_clean = df.dropna(subset=['mean_cosine'])
    bins = np.linspace(df_clean['mean_cosine'].min(), df_clean['mean_cosine'].max(), 10)
    df_clean['cosine_bin'] = pd.cut(df_clean['mean_cosine'], bins)
    
    accuracy_by_bin = df_clean.groupby('cosine_bin')['correct'].agg(['mean', 'count']).reset_index()
    
    # Only plot bins with sufficient samples
    sufficient_samples = accuracy_by_bin['count'] >= 5
    accuracy_by_bin = accuracy_by_bin[sufficient_samples]
    
    if len(accuracy_by_bin) > 0:
        bin_centers = [interval.mid for interval in accuracy_by_bin['cosine_bin']]
        ax3.bar(range(len(bin_centers)), accuracy_by_bin['mean'] * 100, alpha=0.7, color='blue')
        ax3.set_xlabel('Cosine Similarity Bins')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy vs Post-Steering Cosine Similarity')
        ax3.set_xticks(range(len(bin_centers)))
        ax3.set_xticklabels([f'{c:.3f}' for c in bin_centers], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add sample counts on bars
        for i, (acc, count) in enumerate(zip(accuracy_by_bin['mean'] * 100, accuracy_by_bin['count'])):
            ax3.text(i, acc + 1, f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Sampling method analysis (if available)
    if 'sampling_method' in df.columns:
        sampling_accuracy = df.groupby('sampling_method')['correct'].agg(['mean', 'count']).reset_index()
        sampling_accuracy = sampling_accuracy[sampling_accuracy['count'] >= 5]  # Filter small groups
        
        if len(sampling_accuracy) > 0:
            ax4.bar(sampling_accuracy['sampling_method'], sampling_accuracy['mean'] * 100, 
                   alpha=0.7, color='orange')
            ax4.set_xlabel('Sampling Method')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Accuracy by Sampling Method')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add sample counts
            for i, (method, acc, count) in enumerate(zip(sampling_accuracy['sampling_method'], 
                                                        sampling_accuracy['mean'] * 100,
                                                        sampling_accuracy['count'])):
                ax4.text(i, acc + 1, f'n={count}', ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No sampling method data\nor insufficient samples', 
                    ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No sampling method data available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig

def print_summary_statistics(df, analysis_data):
    """Print comprehensive summary statistics"""
    print("=" * 80)
    print("POST-STEERING COSINE CORRELATION ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Basic statistics
    total_samples = len(df)
    correct_samples = df['correct'].sum()
    accuracy = correct_samples / total_samples * 100
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total samples: {total_samples}")
    print(f"   Correct predictions: {correct_samples}")
    print(f"   Incorrect predictions: {total_samples - correct_samples}")
    print(f"   Overall accuracy: {accuracy:.2f}%")
    
    # Cosine similarity statistics
    print(f"\n🎯 POST-STEERING COSINE SIMILARITY STATISTICS:")
    cosine_stats = df['mean_cosine'].describe()
    print(f"   Mean: {cosine_stats['mean']:.4f}")
    print(f"   Std:  {cosine_stats['std']:.4f}")
    print(f"   Min:  {cosine_stats['min']:.4f}")
    print(f"   Max:  {cosine_stats['max']:.4f}")
    
    # Correct vs Incorrect comparison
    correct_cosines = df[df['correct'] == True]['mean_cosine'].dropna()
    incorrect_cosines = df[df['correct'] == False]['mean_cosine'].dropna()
    
    if len(correct_cosines) > 0 and len(incorrect_cosines) > 0:
        print(f"\n📈 CORRECT vs INCORRECT COMPARISON:")
        print(f"   Correct predictions - Mean cosine: {correct_cosines.mean():.4f}")
        print(f"   Incorrect predictions - Mean cosine: {incorrect_cosines.mean():.4f}")
        print(f"   Difference (Correct - Incorrect): {correct_cosines.mean() - incorrect_cosines.mean():.4f}")
        
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(correct_cosines, incorrect_cosines)
        print(f"   Statistical test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"   Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Layer-wise analysis summary
    layer_analysis = analysis_data.get('layer_wise_analysis', {})
    if layer_analysis:
        print(f"\n🔍 LAYER-WISE ANALYSIS:")
        print(f"   Layers analyzed: {len(layer_analysis)}")
        
        significant_layers = [k for k, v in layer_analysis.items() if v['comparison']['significant']]
        print(f"   Significant layers (p < 0.05): {len(significant_layers)}")
        
        if significant_layers:
            print(f"   Significant layer IDs: {', '.join(significant_layers)}")
            
            # Find layers with largest effects
            large_effects = []
            medium_effects = []
            for layer_id in significant_layers:
                effect_size = abs(layer_analysis[layer_id]['comparison']['cohens_d'])
                if effect_size >= 0.8:
                    large_effects.append(layer_id)
                elif effect_size >= 0.5:
                    medium_effects.append(layer_id)
            
            if large_effects:
                print(f"   Large effect sizes (|d| ≥ 0.8): {', '.join(large_effects)}")
            if medium_effects:
                print(f"   Medium effect sizes (0.5 ≤ |d| < 0.8): {', '.join(medium_effects)}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    if len(correct_cosines) > 0 and len(incorrect_cosines) > 0:
        if correct_cosines.mean() > incorrect_cosines.mean():
            print(f"   ✅ Higher post-steering cosine similarity correlates with CORRECT predictions")
            print(f"      → Steering is working as intended - successfully aligning hidden states")
        else:
            print(f"   ⚠️  Higher post-steering cosine similarity correlates with INCORRECT predictions")
            print(f"      → Steering may be counterproductive or needs adjustment")
    
    if significant_layers:
        print(f"   📍 {len(significant_layers)} layers show significant correlation patterns")
        print(f"   🔧 These layers are most important for steering effectiveness")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Visualize post-steering cosine correlation results")
    parser.add_argument("--csv", type=str, 
                       default="first_eng_cosine_correlation_evaluation_result_cosine_correlation_lambda0.05.csv",
                       help="Path to CSV results file")
    parser.add_argument("--json", type=str,
                       default="first_eng_cosine_correlation_evaluation_result_cosine_correlation_lambda0.05_analysis.json", 
                       help="Path to JSON analysis file")
    parser.add_argument("--output_dir", type=str, default="./first_post_steering_analysis/",
                       help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    df, analysis_data = load_data(args.csv, args.json)
    print(f"   Loaded {len(df)} samples")
    
    # Print summary statistics
    print_summary_statistics(df, analysis_data)
    
    # Generate plots
    print(f"\n📊 Generating visualizations...")
    
    # 1. Accuracy overview
    print("   Creating accuracy overview...")
    fig1 = plot_accuracy_overview(df, analysis_data)
    fig1.savefig(output_dir / "01_accuracy_overview.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir / '01_accuracy_overview.png'}")
    
    # 2. Layer-wise analysis
    print("   Creating layer-wise analysis...")
    fig2 = plot_layer_wise_analysis(analysis_data)
    if fig2:
        fig2.savefig(output_dir / "02_layer_wise_analysis.png", dpi=300, bbox_inches='tight')
        print(f"   Saved: {output_dir / '02_layer_wise_analysis.png'}")
    
    # 3. Cosine similarity patterns
    print("   Creating cosine similarity patterns...")
    fig3 = plot_cosine_similarity_patterns(df)
    fig3.savefig(output_dir / "03_cosine_patterns.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir / '03_cosine_patterns.png'}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()