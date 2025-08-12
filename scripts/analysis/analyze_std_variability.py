#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

# Import the orthogonal layer functions to access stored angle data
sys.path.insert(0, os.path.abspath("."))
from llm_layer_orthogonal import get_layer_by_layer_statistics

def analyze_std_deviation_variability():
    """Analyze and visualize the variability of standard deviations across layers"""
    
    # Get the layer statistics
    layer_stats = get_layer_by_layer_statistics()
    
    if not layer_stats:
        print("No layer statistics available! Run angle analysis first.")
        return
    
    # Extract standard deviations for each layer
    layers = sorted(layer_stats.keys())
    layer_stds = [layer_stats[l]['std'] for l in layers]
    layer_means = [layer_stats[l]['mean'] for l in layers]
    
    # Calculate statistics about the standard deviations
    std_of_stds = np.std(layer_stds)
    mean_of_stds = np.mean(layer_stds)
    min_std = np.min(layer_stds)
    max_std = np.max(layer_stds)
    range_of_stds = max_std - min_std
    
    # Find layers with highest and lowest variability
    max_std_layer = layers[layer_stds.index(max_std)]
    min_std_layer = layers[layer_stds.index(min_std)]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Standard deviations across layers with trend
    plt.subplot(2, 3, 1)
    plt.plot(layers, layer_stds, 'bo-', markersize=5, linewidth=2)
    plt.axhline(mean_of_stds, color='red', linestyle='--', 
                label=f'Mean of StdDevs: {mean_of_stds:.3f}°')
    plt.axhline(mean_of_stds + std_of_stds, color='orange', linestyle=':', 
                label=f'+1σ: {mean_of_stds + std_of_stds:.3f}°')
    plt.axhline(mean_of_stds - std_of_stds, color='orange', linestyle=':', 
                label=f'-1σ: {mean_of_stds - std_of_stds:.3f}°')
    plt.fill_between(layers, mean_of_stds - std_of_stds, mean_of_stds + std_of_stds, 
                     alpha=0.2, color='orange', label='StdDev Range')
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Standard Deviation Across Layers\n(Variability of Variability)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of standard deviations
    plt.subplot(2, 3, 2)
    plt.hist(layer_stds, bins=min(15, len(layers)//2), alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(mean_of_stds, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_of_stds:.3f}°')
    plt.xlabel('Standard Deviation (degrees)')
    plt.ylabel('Number of Layers')
    plt.title('Distribution of Standard Deviations\nAcross All Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Relationship between layer mean angles and their std devs
    plt.subplot(2, 3, 3)
    plt.scatter(layer_means, layer_stds, c=layers, cmap='viridis', s=50)
    plt.colorbar(label='Layer Index')
    
    # Add trend line
    if len(layers) > 2:
        z = np.polyfit(layer_means, layer_stds, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(layer_means), max(layer_means), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
                label=f'Trend: slope={z[0]:.4f}')
        plt.legend()
    
    plt.xlabel('Mean Angle (degrees)')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Mean Angle vs. Standard Deviation\n(Colored by Layer)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Deviation from mean standard deviation
    plt.subplot(2, 3, 4)
    std_deviations_from_mean = np.array(layer_stds) - mean_of_stds
    plt.bar(layers, std_deviations_from_mean, 
            color=['red' if x > 0 else 'blue' for x in std_deviations_from_mean],
            alpha=0.7)
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Layer Index')
    plt.ylabel('Deviation from Mean StdDev (degrees)')
    plt.title('Each Layer\'s StdDev Relative to Average\n(Red=Above Avg, Blue=Below Avg)')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Moving average of standard deviations
    plt.subplot(2, 3, 5)
    window_size = min(5, len(layers)//4)  # Adaptive window size
    if window_size >= 2:
        moving_avg = np.convolve(layer_stds, np.ones(window_size), 'valid') / window_size
        moving_layers = layers[window_size-1:]
        plt.plot(layers, layer_stds, 'b-', alpha=0.5, label='Original StdDev')
        plt.plot(moving_layers, moving_avg, 'r-', linewidth=3, 
                label=f'{window_size}-Layer Moving Average')
        plt.legend()
    else:
        plt.plot(layers, layer_stds, 'b-', linewidth=2)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Smoothed Standard Deviation Trend')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Statistics summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate coefficient of variation for std devs
    coeff_var = std_of_stds / mean_of_stds * 100 if mean_of_stds > 0 else 0
    
    stats_text = f"""
STANDARD DEVIATION VARIABILITY ANALYSIS

Key Metrics:
• Mean of StdDevs: {mean_of_stds:.3f}°
• StdDev of StdDevs: {std_of_stds:.3f}°
• Coefficient of Variation: {coeff_var:.1f}%
• Range of StdDevs: {range_of_stds:.3f}°

Layer Analysis:
• Most variable layer: {max_std_layer} ({max_std:.3f}°)
• Most consistent layer: {min_std_layer} ({min_std:.3f}°)
• Total layers analyzed: {len(layers)}

Interpretation:
• Low StdDev of StdDevs ({std_of_stds:.3f}°) = 
  Consistent variability across layers
• High StdDev of StdDevs = 
  Some layers much more variable than others

Coefficient of Variation:
• <15%: Very consistent variability
• 15-30%: Moderate variation in variability  
• >30%: High variation in variability

Your result: {coeff_var:.1f}% = 
{"Very consistent" if coeff_var < 15 else "Moderate variation" if coeff_var < 30 else "High variation"}
"""
    
    plt.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('std_deviation_variability_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('std_deviation_variability_analysis.pdf', bbox_inches='tight')
    
    print("Standard deviation variability analysis saved as:")
    print("  • std_deviation_variability_analysis.png")
    print("  • std_deviation_variability_analysis.pdf")
    
    plt.show()
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("STANDARD DEVIATION VARIABILITY SUMMARY")  
    print(f"{'='*60}")
    print(f"Standard deviation of standard deviations: {std_of_stds:.3f}°")
    print(f"Mean of standard deviations: {mean_of_stds:.3f}°") 
    print(f"Coefficient of variation: {coeff_var:.1f}%")
    print(f"Range of standard deviations: {min_std:.3f}° to {max_std:.3f}°")
    print(f"Most variable layer: {max_std_layer} (σ = {max_std:.3f}°)")
    print(f"Most consistent layer: {min_std_layer} (σ = {min_std:.3f}°)")
    
    print(f"\nInterpretation:")
    if coeff_var < 15:
        print("• Very consistent variability across layers")
        print("• All layers have similar steering consistency")
    elif coeff_var < 30:
        print("• Moderate variation in layer consistency") 
        print("• Some layers more variable than others")
    else:
        print("• High variation in layer consistency")
        print("• Some layers much more erratic than others")
    
    # Save detailed statistics
    variability_stats = {
        'std_of_stds': std_of_stds,
        'mean_of_stds': mean_of_stds,
        'coefficient_of_variation_percent': coeff_var,
        'min_std': min_std,
        'max_std': max_std,
        'range_of_stds': range_of_stds,
        'most_variable_layer': max_std_layer,
        'most_consistent_layer': min_std_layer,
        'layer_std_deviations': dict(zip(layers, layer_stds)),
        'interpretation': "Very consistent" if coeff_var < 15 else "Moderate variation" if coeff_var < 30 else "High variation"
    }
    
    with open('std_variability_analysis.json', 'w') as f:
        json.dump(variability_stats, f, indent=2)
    
    print(f"Detailed statistics saved to 'std_variability_analysis.json'")
    
    return variability_stats

if __name__ == "__main__":
    analyze_std_deviation_variability()