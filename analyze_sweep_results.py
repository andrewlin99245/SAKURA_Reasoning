#!/usr/bin/env python3
"""
Analyze parameter sweep results and find optimal settings.
"""
import pandas as pd
import numpy as np
import os

def analyze_beta_results():
    """Analyze beta weight sweep results."""
    if not os.path.exists('parameter_sweep_results.csv'):
        print("❌ Beta sweep results not found. Run parameter_sweep_beta.sh first.")
        return None
    
    print("🔍 ANALYZING BETA WEIGHT RESULTS")
    print("=" * 50)
    
    df = pd.read_csv('parameter_sweep_results.csv')
    
    # Find best results
    best_accuracy = df.loc[df['accuracy'].idxmax()]
    best_improvement = df.loc[df['improvement'].idxmax()]
    
    print(f"📊 Total combinations tested: {len(df)}")
    print(f"📈 Accuracy range: {df['accuracy'].min():.2f}% - {df['accuracy'].max():.2f}%")
    print(f"📈 Improvement range: {df['improvement'].min():.2f} - {df['improvement'].max():.2f} pts")
    print()
    
    print("🏆 BEST ACCURACY:")
    print(f"  β1={best_accuracy['beta1']}, β2={best_accuracy['beta2']}, β3={best_accuracy['beta3']}")
    print(f"  Accuracy: {best_accuracy['accuracy']:.2f}% (+{best_accuracy['improvement']:.2f} pts)")
    print(f"  Flips: {best_accuracy['flips']} ({best_accuracy['flipped_pct']:.1f}%)")
    print()
    
    if best_accuracy.name != best_improvement.name:
        print("🏆 BEST IMPROVEMENT:")
        print(f"  β1={best_improvement['beta1']}, β2={best_improvement['beta2']}, β3={best_improvement['beta3']}")
        print(f"  Accuracy: {best_improvement['accuracy']:.2f}% (+{best_improvement['improvement']:.2f} pts)")
        print(f"  Flips: {best_improvement['flips']} ({best_improvement['flipped_pct']:.1f}%)")
        print()
    
    # Top 5 results
    print("🥇 TOP 5 COMBINATIONS:")
    top5 = df.nlargest(5, 'improvement')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. β=({row['beta1']}, {row['beta2']}, {row['beta3']}) → "
              f"{row['accuracy']:.2f}% (+{row['improvement']:.2f} pts)")
    print()
    
    # Feature importance analysis
    print("📊 FEATURE IMPORTANCE (Average improvement by beta value):")
    for beta_col in ['beta1', 'beta2', 'beta3']:
        feature_name = {'beta1': 'Frobenius', 'beta2': 'Band', 'beta3': 'Eigenvalue'}[beta_col]
        avg_by_beta = df.groupby(beta_col)['improvement'].mean().sort_values(ascending=False)
        print(f"  {feature_name} ({beta_col}):")
        for val, avg_imp in avg_by_beta.items():
            print(f"    {val}: {avg_imp:.3f} pts")
        print()
    
    return best_improvement

def analyze_threshold_results():
    """Analyze threshold sweep results."""
    if not os.path.exists('threshold_sweep_results.csv'):
        print("❌ Threshold sweep results not found. Run parameter_sweep_threshold.sh first.")
        return None
    
    print("🎯 ANALYZING THRESHOLD RESULTS")
    print("=" * 50)
    
    df = pd.read_csv('threshold_sweep_results.csv')
    
    best_accuracy = df.loc[df['accuracy'].idxmax()]
    best_improvement = df.loc[df['improvement'].idxmax()]
    best_precision = df.loc[df['precision_estimate'].idxmax()]
    
    print(f"📊 Thresholds tested: {len(df)}")
    print(f"📈 Accuracy range: {df['accuracy'].min():.2f}% - {df['accuracy'].max():.2f}%")
    print(f"🎯 Threshold range: {df['threshold'].min()} - {df['threshold'].max()}")
    print()
    
    print("🏆 BEST RESULTS:")
    print(f"  Best Accuracy: threshold={best_accuracy['threshold']} → {best_accuracy['accuracy']:.2f}%")
    print(f"  Best Improvement: threshold={best_improvement['threshold']} → +{best_improvement['improvement']:.2f} pts")  
    print(f"  Best Precision: threshold={best_precision['threshold']} → {best_precision['precision_estimate']:.2f}")
    print()
    
    # Show all results
    print("📊 ALL THRESHOLD RESULTS:")
    print("Thresh | Accuracy | Improve | Flips | Precision")
    print("-------|----------|---------|-------|----------")
    for _, row in df.iterrows():
        print(f" {row['threshold']:5.2f} | {row['accuracy']:7.2f}% | {row['improvement']:6.2f} | "
              f"{row['flips']:5.0f} | {row['precision_estimate']:8.2f}")
    
    return best_improvement

def analyze_layer_results():
    """Analyze layer count sweep results."""
    if not os.path.exists('layer_sweep_results.csv'):
        print("❌ Layer sweep results not found. Run parameter_sweep_layers.sh first.")
        return None
    
    print("📊 ANALYZING LAYER COUNT RESULTS")
    print("=" * 50)
    
    df = pd.read_csv('layer_sweep_results.csv')
    
    best_result = df.loc[df['improvement'].idxmax()]
    
    print(f"📊 Layer counts tested: {list(df['layers'].values)}")
    print(f"📈 Improvement range: {df['improvement'].min():.2f} - {df['improvement'].max():.2f} pts")
    print()
    
    print("🏆 BEST LAYER COUNT:")
    print(f"  Layers: {best_result['layers']}")
    print(f"  Accuracy: {best_result['accuracy']:.2f}% (+{best_result['improvement']:.2f} pts)")
    print(f"  Flips: {best_result['flips']} ({best_result['flipped_pct']:.1f}%)")
    print()
    
    print("📊 ALL LAYER RESULTS:")
    print("Layers | Accuracy | Improve | Flips")
    print("-------|----------|---------|------")
    for _, row in df.iterrows():
        print(f"   {row['layers']:3.0f} | {row['accuracy']:7.2f}% | {row['improvement']:6.2f} | {row['flips']:5.0f}")
    
    return best_result

def main():
    print("🔍 PARAMETER SWEEP ANALYSIS")
    print("=" * 60)
    print()
    
    # Analyze each sweep type
    best_beta = analyze_beta_results()
    print()
    
    best_threshold = analyze_threshold_results()  
    print()
    
    best_layer = analyze_layer_results()
    print()
    
    # Final recommendations
    print("🏆 FINAL RECOMMENDATIONS")
    print("=" * 50)
    
    if best_beta is not None:
        print("🎛️  OPTIMAL BETA WEIGHTS:")
        print(f"   --beta1 {best_beta['beta1']} --beta2 {best_beta['beta2']} --beta3 {best_beta['beta3']}")
    
    if best_threshold is not None:
        print(f"🎯 OPTIMAL THRESHOLD: --flip_threshold {best_threshold['threshold']}")
        
    if best_layer is not None:
        print(f"📊 OPTIMAL LAYER COUNT: --template_last_k {int(best_layer['layers'])}")
    
    if best_beta is not None and best_threshold is not None:
        expected_accuracy = max(best_beta.get('accuracy', 0), best_threshold.get('accuracy', 0))
        expected_improvement = max(best_beta.get('improvement', 0), best_threshold.get('improvement', 0))
        print()
        print(f"🚀 EXPECTED PERFORMANCE:")
        print(f"   Accuracy: {expected_accuracy:.2f}% (+{expected_improvement:.2f} pts)")

if __name__ == "__main__":
    main()