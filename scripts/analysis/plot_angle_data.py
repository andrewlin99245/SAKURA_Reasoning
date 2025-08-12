#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

# Import the orthogonal layer functions to access stored angle data
sys.path.insert(0, os.path.abspath("."))
from llm_layer_orthogonal import get_layer_by_layer_statistics, get_angle_statistics

def plot_existing_angle_data():
    """Plot angle statistics from currently stored data"""
    
    # Get stored angle data
    layer_stats = get_layer_by_layer_statistics() 
    overall_stats = get_angle_statistics()
    
    if not layer_stats or not overall_stats:
        print("No angle data currently stored!")
        print("Run your orthogonal steering model first to collect angle measurements.")
        return False
    
    print(f"Found angle data for {len(layer_stats)} layers with {overall_stats['count']} total measurements")
    
    # Create visualization
    create_angle_plots(layer_stats, overall_stats)
    return True

def create_angle_plots(layer_stats, overall_stats, save_path="stored_angle_analysis"):
    """Create plots from angle statistics"""
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Extract data
    layers = sorted(layer_stats.keys())
    layer_means = [layer_stats[l]['mean'] for l in layers]
    layer_stds = [layer_stats[l]['std'] for l in layers]
    layer_mins = [layer_stats[l]['min'] for l in layers]
    layer_maxs = [layer_stats[l]['max'] for l in layers]
    
    # 1. Mean angles per layer with error bars
    plt.subplot(2, 3, 1)
    plt.errorbar(layers, layer_means, yerr=layer_stds, marker='o', capsize=5, capthick=2, markersize=4)
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Mean Steering Vector Angles per Layer')
    plt.grid(True, alpha=0.3)
    
    # 2. Min/Max range visualization
    plt.subplot(2, 3, 2)
    plt.fill_between(layers, layer_mins, layer_maxs, alpha=0.3, color='lightblue', label='Min-Max Range')
    plt.plot(layers, layer_means, 'ro-', markersize=4, label='Mean')
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Angle Range per Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Overall distribution histogram
    plt.subplot(2, 3, 3)
    all_angles = overall_stats['all_angles']
    plt.hist(all_angles, bins=min(30, len(all_angles)//2), alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(overall_stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {overall_stats["mean"]:.1f}°')
    plt.axvline(overall_stats['mean'] + overall_stats['std'], color='orange', linestyle=':', 
                label=f'+1σ: {overall_stats["mean"] + overall_stats["std"]:.1f}°')
    plt.axvline(overall_stats['mean'] - overall_stats['std'], color='orange', linestyle=':', 
                label=f'-1σ: {overall_stats["mean"] - overall_stats["std"]:.1f}°')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Angles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Layer progression (trend analysis)
    plt.subplot(2, 3, 4)
    plt.plot(layers, layer_means, 'bo-', markersize=4)
    # Add trend line
    if len(layers) > 1:
        z = np.polyfit(layers, layer_means, 1)
        p = np.poly1d(z)
        plt.plot(layers, p(layers), "r--", alpha=0.8, 
                label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
        plt.legend()
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Angle (degrees)')
    plt.title('Angle Progression Across Layers')
    plt.grid(True, alpha=0.3)
    
    # 5. Standard deviation per layer
    plt.subplot(2, 3, 5)
    plt.bar(layers, layer_stds, alpha=0.7, color='lightcoral')
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Angle Variability per Layer')
    plt.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Find interesting layers
    max_mean_layer = layers[layer_means.index(max(layer_means))]
    min_mean_layer = layers[layer_means.index(min(layer_means))]
    max_std_layer = layers[layer_stds.index(max(layer_stds))]
    
    stats_text = f"""
Angle Statistics Summary:

Overall:
• Total measurements: {overall_stats['count']}
• Mean angle: {overall_stats['mean']:.2f}°
• Std deviation: {overall_stats['std']:.2f}°
• Range: {overall_stats['min']:.1f}° - {overall_stats['max']:.1f}°

Layer Analysis:
• Number of layers: {len(layers)}
• Highest mean: {max(layer_means):.2f}° (Layer {max_mean_layer})
• Lowest mean: {min(layer_means):.2f}° (Layer {min_mean_layer})
• Most variable: {max(layer_stds):.2f}° std (Layer {max_std_layer})

Interpretation:
• Angles > 90° indicate opposing directions
• Angles < 90° indicate similar directions  
• High std indicates inconsistent alignment
"""
    
    plt.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    
    print(f"\nAngle analysis plots saved as:")
    print(f"  • {save_path}.png")
    print(f"  • {save_path}.pdf")
    
    plt.show()
    
    # Print summary to console
    print(f"\n{'='*50}")
    print("ANGLE ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Total angle measurements: {overall_stats['count']}")
    print(f"Mean angle: {overall_stats['mean']:.2f}° ± {overall_stats['std']:.2f}°")
    print(f"Angle range: {overall_stats['min']:.1f}° to {overall_stats['max']:.1f}°")
    print(f"Number of layers analyzed: {len(layers)}")
    print(f"Layer with highest mean angle: {max_mean_layer} ({max(layer_means):.2f}°)")
    print(f"Layer with lowest mean angle: {min_mean_layer} ({min(layer_means):.2f}°)")
    print(f"Layer with most variability: {max_std_layer} (std={max(layer_stds):.2f}°)")

def create_simple_angle_plot():
    """Create a simple plot if you just want the basics"""
    
    layer_stats = get_layer_by_layer_statistics()
    overall_stats = get_angle_statistics()
    
    if not layer_stats or not overall_stats:
        print("No angle data available!")
        return
    
    # Simple layer-wise plot
    layers = sorted(layer_stats.keys())
    layer_means = [layer_stats[l]['mean'] for l in layers]
    layer_stds = [layer_stats[l]['std'] for l in layers]
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Mean angles with error bars
    plt.subplot(1, 2, 1)
    plt.errorbar(layers, layer_means, yerr=layer_stds, marker='o', capsize=5)
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Steering Vector Angles per Layer')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution
    plt.subplot(1, 2, 2)
    all_angles = overall_stats['all_angles']
    plt.hist(all_angles, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(overall_stats['mean'], color='red', linestyle='--', 
                label=f'Mean: {overall_stats["mean"]:.1f}°')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Angle Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_angle_analysis.png', dpi=200, bbox_inches='tight')
    print("Simple angle plot saved as 'simple_angle_analysis.png'")
    plt.show()

if __name__ == "__main__":
    print("Checking for stored angle data...")
    
    if not plot_existing_angle_data():
        print("\nTo collect angle data, run your SLAandSteering_orthogonal.py script first.")
        print("The angle measurements will be automatically collected during model inference.")
        print("\nAlternatively, run visualize_angle_statistics.py for a complete demo.")