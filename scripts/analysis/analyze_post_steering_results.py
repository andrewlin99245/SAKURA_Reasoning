#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

def load_data(csv_path, json_path):
    """Load both CSV and JSON data"""
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Load JSON analysis data
    with open(json_path, 'r') as f:
        analysis_data = json.load(f)
    
    return df, analysis_data

def analyze_results(df, analysis_data):
    """Analyze post-steering cosine correlation results"""
    
    print("=" * 80)
    print("POST-STEERING COSINE CORRELATION ANALYSIS")
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
    print(f"   25th percentile: {cosine_stats['25%']:.4f}")
    print(f"   75th percentile: {cosine_stats['75%']:.4f}")
    
    # Correct vs Incorrect comparison
    correct_cosines = df[df['correct'] == True]['mean_cosine'].dropna()
    incorrect_cosines = df[df['correct'] == False]['mean_cosine'].dropna()
    
    if len(correct_cosines) > 0 and len(incorrect_cosines) > 0:
        print(f"\n📈 CORRECT vs INCORRECT COMPARISON:")
        print(f"   Correct predictions:")
        print(f"     Count: {len(correct_cosines)}")
        print(f"     Mean cosine: {correct_cosines.mean():.4f}")
        print(f"     Std cosine:  {correct_cosines.std():.4f}")
        print(f"   Incorrect predictions:")
        print(f"     Count: {len(incorrect_cosines)}")
        print(f"     Mean cosine: {incorrect_cosines.mean():.4f}")
        print(f"     Std cosine:  {incorrect_cosines.std():.4f}")
        print(f"   Difference (Correct - Incorrect): {correct_cosines.mean() - incorrect_cosines.mean():.4f}")
        
        # Simple statistical test
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(correct_cosines, incorrect_cosines)
        print(f"   Statistical test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"   Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(correct_cosines) - 1) * correct_cosines.std()**2 + 
                             (len(incorrect_cosines) - 1) * incorrect_cosines.std()**2) / 
                            (len(correct_cosines) + len(incorrect_cosines) - 2))
        cohens_d = (correct_cosines.mean() - incorrect_cosines.mean()) / pooled_std
        print(f"   Effect size (Cohen's d): {cohens_d:.3f}")
        
        effect_interpretation = (
            'Large' if abs(cohens_d) >= 0.8 else
            'Medium' if abs(cohens_d) >= 0.5 else
            'Small' if abs(cohens_d) >= 0.2 else
            'Negligible'
        )
        print(f"   Effect size interpretation: {effect_interpretation}")
    
    # Layer-wise analysis summary
    layer_analysis = analysis_data.get('layer_wise_analysis', {})
    if layer_analysis:
        print(f"\n🔍 LAYER-WISE ANALYSIS:")
        print(f"   Layers analyzed: {len(layer_analysis)}")
        
        # Find significant layers
        significant_layers = []
        layer_stats = []
        
        for layer_id, stats in layer_analysis.items():
            is_significant = stats['comparison']['significant']
            if is_significant:
                significant_layers.append(int(layer_id))
            
            layer_stats.append({
                'layer': int(layer_id),
                'correct_mean': stats['correct_predictions']['mean_cosine'],
                'incorrect_mean': stats['incorrect_predictions']['mean_cosine'],
                'difference': stats['comparison']['mean_difference'],
                'p_value': stats['comparison']['p_value'],
                'cohens_d': stats['comparison']['cohens_d'],
                'significant': is_significant
            })
        
        # Sort by layer number
        layer_stats.sort(key=lambda x: x['layer'])
        
        print(f"   Significant layers (p < 0.05): {len(significant_layers)}")
        
        if significant_layers:
            print(f"   Significant layer IDs: {sorted(significant_layers)}")
            
            # Find layers with largest effects
            large_effects = [s['layer'] for s in layer_stats if s['significant'] and abs(s['cohens_d']) >= 0.8]
            medium_effects = [s['layer'] for s in layer_stats if s['significant'] and 0.5 <= abs(s['cohens_d']) < 0.8]
            small_effects = [s['layer'] for s in layer_stats if s['significant'] and 0.2 <= abs(s['cohens_d']) < 0.5]
            
            if large_effects:
                print(f"   Large effect sizes (|d| ≥ 0.8): {large_effects}")
            if medium_effects:
                print(f"   Medium effect sizes (0.5 ≤ |d| < 0.8): {medium_effects}")
            if small_effects:
                print(f"   Small effect sizes (0.2 ≤ |d| < 0.5): {small_effects}")
            
            # Show top 10 most significant layers
            print(f"\n   📋 TOP 10 MOST SIGNIFICANT LAYERS:")
            significant_sorted = [s for s in layer_stats if s['significant']]
            significant_sorted.sort(key=lambda x: x['p_value'])
            
            print(f"   {'Layer':<6} {'Correct':<10} {'Incorrect':<12} {'Difference':<12} {'Effect Size':<12} {'P-value':<10}")
            print(f"   {'-'*70}")
            
            for stats in significant_sorted[:10]:
                print(f"   {stats['layer']:<6} {stats['correct_mean']:<10.4f} {stats['incorrect_mean']:<12.4f} "
                      f"{stats['difference']:<12.4f} {stats['cohens_d']:<12.3f} {stats['p_value']:<10.4f}")
    
    # Sampling method analysis (if available)
    if 'sampling_method' in df.columns:
        print(f"\n📊 SAMPLING METHOD ANALYSIS:")
        sampling_stats = df.groupby('sampling_method').agg({
            'correct': ['count', 'sum', 'mean'],
            'mean_cosine': ['mean', 'std']
        }).round(4)
        
        sampling_stats.columns = ['Count', 'Correct', 'Accuracy', 'Mean_Cosine', 'Std_Cosine']
        sampling_stats['Accuracy'] = sampling_stats['Accuracy'] * 100
        
        print(f"   {'Method':<15} {'Count':<8} {'Accuracy':<10} {'Mean_Cosine':<12} {'Std_Cosine':<12}")
        print(f"   {'-'*65}")
        
        for method, row in sampling_stats.iterrows():
            print(f"   {method:<15} {row['Count']:<8.0f} {row['Accuracy']:<10.1f}% "
                  f"{row['Mean_Cosine']:<12.4f} {row['Std_Cosine']:<12.4f}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    if len(correct_cosines) > 0 and len(incorrect_cosines) > 0:
        difference = correct_cosines.mean() - incorrect_cosines.mean()
        if difference > 0:
            print(f"   ✅ Higher post-steering cosine similarity correlates with CORRECT predictions")
            print(f"      → Steering is working as intended: successfully aligning hidden states")
            print(f"      → Difference: +{difference:.4f} (correct predictions have higher similarity)")
        else:
            print(f"   ⚠️  Higher post-steering cosine similarity correlates with INCORRECT predictions")
            print(f"      → Steering may be counterproductive or needs adjustment")
            print(f"      → Difference: {difference:.4f} (incorrect predictions have higher similarity)")
    
    if significant_layers:
        print(f"   📍 {len(significant_layers)} layers show significant correlation patterns")
        print(f"   🔧 These layers are most important for steering effectiveness")
        
        # Identify problematic layers (negative differences)
        problematic_layers = [s['layer'] for s in layer_stats if s['significant'] and s['difference'] < 0]
        if problematic_layers:
            print(f"   🚨 Problematic layers (negative correlation): {problematic_layers}")
            print(f"      → Consider negative lambda for these layers")
    
    # Cosine similarity range analysis
    high_cosine = df[df['mean_cosine'] > 0.3]
    low_cosine = df[df['mean_cosine'] < 0.1]
    
    if len(high_cosine) > 0:
        high_accuracy = high_cosine['correct'].mean() * 100
        print(f"   📈 High cosine similarity (>0.3): {len(high_cosine)} samples, {high_accuracy:.1f}% accuracy")
    
    if len(low_cosine) > 0:
        low_accuracy = low_cosine['correct'].mean() * 100
        print(f"   📉 Low cosine similarity (<0.1): {len(low_cosine)} samples, {low_accuracy:.1f}% accuracy")
    
    print("=" * 80)
    
    return layer_stats

def export_summary(layer_stats, output_path):
    """Export layer statistics to CSV"""
    layer_df = pd.DataFrame(layer_stats)
    layer_df.to_csv(output_path, index=False)
    print(f"\n💾 Layer statistics exported to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze post-steering cosine correlation results")
    parser.add_argument("--csv", type=str, 
                       default="cosine_correlation_evaluation_result_cosine_correlation_lambda0.05.csv",
                       help="Path to CSV results file")
    parser.add_argument("--json", type=str,
                       default="cosine_correlation_evaluation_result_cosine_correlation_lambda0.05_analysis.json", 
                       help="Path to JSON analysis file")
    parser.add_argument("--output", type=str, default="layer_statistics_post_steering.csv",
                       help="Output CSV file for layer statistics")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.csv).exists():
        print(f"❌ CSV file not found: {args.csv}")
        return
    
    if not Path(args.json).exists():
        print(f"❌ JSON file not found: {args.json}")
        return
    
    # Load data
    print("📂 Loading data...")
    df, analysis_data = load_data(args.csv, args.json)
    print(f"   Loaded {len(df)} samples")
    
    # Analyze results
    layer_stats = analyze_results(df, analysis_data)
    
    # Export summary
    if layer_stats:
        export_summary(layer_stats, args.output)

if __name__ == "__main__":
    main()