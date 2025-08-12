#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import json

def analyze_std_variability_from_json():
    """Analyze standard deviation variability using the saved JSON data"""
    
    # Load the angle analysis data
    try:
        with open('quick_angle_analysis_data.json', 'r') as f:
            data = json.load(f)
        layer_stats = data['layer_statistics']
    except FileNotFoundError:
        print("Error: quick_angle_analysis_data.json not found!")
        print("Please run the angle analysis first.")
        return
    
    # Extract standard deviations for each layer
    layers = sorted([int(k) for k in layer_stats.keys()])
    layer_stds = [layer_stats[str(l)]['std'] for l in layers]
    layer_means = [layer_stats[str(l)]['mean'] for l in layers]
    
    # Calculate statistics about the standard deviations
    std_of_stds = np.std(layer_stds)
    mean_of_stds = np.mean(layer_stds)
    min_std = np.min(layer_stds)
    max_std = np.max(layer_stds)
    range_of_stds = max_std - min_std
    
    # Find layers with highest and lowest variability
    max_std_layer = layers[layer_stds.index(max_std)]
    min_std_layer = layers[layer_stds.index(min_std)]
    
    print(f"{'='*60}")
    print("STANDARD DEVIATION OF STANDARD DEVIATIONS ANALYSIS")
    print(f"{'='*60}")
    print(f"Standard deviation of standard deviations: {std_of_stds:.4f}°")
    print(f"Mean of standard deviations: {mean_of_stds:.4f}°") 
    print(f"Coefficient of variation: {(std_of_stds/mean_of_stds)*100:.2f}%")
    print(f"Range of standard deviations: {min_std:.4f}° to {max_std:.4f}°")
    print(f"Most variable layer: {max_std_layer} (σ = {max_std:.4f}°)")
    print(f"Most consistent layer: {min_std_layer} (σ = {min_std:.4f}°)")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Standard deviations across layers with statistics
    plt.subplot(2, 3, 1)
    plt.plot(layers, layer_stds, 'bo-', markersize=5, linewidth=2)
    plt.axhline(mean_of_stds, color='red', linestyle='--', linewidth=2,
                label=f'Mean of StdDevs: {mean_of_stds:.3f}°')
    plt.axhline(mean_of_stds + std_of_stds, color='orange', linestyle=':', linewidth=2,
                label=f'+1σ: {mean_of_stds + std_of_stds:.3f}°')
    plt.axhline(mean_of_stds - std_of_stds, color='orange', linestyle=':', linewidth=2,
                label=f'-1σ: {mean_of_stds - std_of_stds:.3f}°')
    plt.fill_between(layers, mean_of_stds - std_of_stds, mean_of_stds + std_of_stds, 
                     alpha=0.2, color='orange')
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Standard Deviation Across Layers\n(with Mean ± StdDev of StdDevs)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Highlight extreme layers
    plt.scatter([max_std_layer], [max_std], color='red', s=100, zorder=5, 
                label=f'Max: Layer {max_std_layer}')
    plt.scatter([min_std_layer], [min_std], color='green', s=100, zorder=5,
                label=f'Min: Layer {min_std_layer}')
    
    # Plot 2: Histogram of standard deviations
    plt.subplot(2, 3, 2)
    plt.hist(layer_stds, bins=12, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(mean_of_stds, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_of_stds:.3f}°')
    plt.axvline(mean_of_stds + std_of_stds, color='orange', linestyle=':', 
                label=f'+1σ: {mean_of_stds + std_of_stds:.3f}°')
    plt.axvline(mean_of_stds - std_of_stds, color='orange', linestyle=':', 
                label=f'-1σ: {mean_of_stds - std_of_stds:.3f}°')
    plt.xlabel('Standard Deviation (degrees)')
    plt.ylabel('Number of Layers')
    plt.title('Distribution of Standard Deviations')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Deviation from mean standard deviation (bar chart)
    plt.subplot(2, 3, 3)
    std_deviations_from_mean = np.array(layer_stds) - mean_of_stds
    colors = ['red' if x > std_of_stds else 'orange' if x > 0 else 'lightblue' if x > -std_of_stds else 'blue' 
              for x in std_deviations_from_mean]
    
    plt.bar(layers, std_deviations_from_mean, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axhline(std_of_stds, color='red', linestyle='--', alpha=0.7, 
                label=f'StdDev of StdDevs: ±{std_of_stds:.3f}°')
    plt.axhline(-std_of_stds, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Layer Index')
    plt.ylabel('Deviation from Mean StdDev (degrees)')
    plt.title('Each Layer\'s StdDev Relative to Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Relationship between mean angle and standard deviation
    plt.subplot(2, 3, 4)
    plt.scatter(layer_means, layer_stds, c=layers, cmap='viridis', s=60, alpha=0.8)
    plt.colorbar(label='Layer Index')
    
    # Add correlation analysis
    correlation = np.corrcoef(layer_means, layer_stds)[0, 1]
    
    # Add trend line
    z = np.polyfit(layer_means, layer_stds, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(layer_means), max(layer_means), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
            label=f'Trend (r={correlation:.3f})')
    
    plt.xlabel('Mean Angle (degrees)')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Mean Angle vs. Standard Deviation\n(Correlation Analysis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative variability
    plt.subplot(2, 3, 5)
    cumulative_variance = np.cumsum(np.array(layer_stds)**2)
    plt.plot(layers, cumulative_variance, 'g-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Layer Index')
    plt.ylabel('Cumulative Variance (degrees²)')
    plt.title('Cumulative Variability Across Layers')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Detailed statistics summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    coeff_var = (std_of_stds / mean_of_stds) * 100
    
    # Classify consistency
    if coeff_var < 15:
        consistency = "Very Consistent"
        color = 'lightgreen'
    elif coeff_var < 30:
        consistency = "Moderately Variable" 
        color = 'lightyellow'
    else:
        consistency = "Highly Variable"
        color = 'lightcoral'
    
    stats_text = f"""
VARIABILITY OF VARIABILITY ANALYSIS

Core Statistics:
• StdDev of StdDevs: {std_of_stds:.4f}°
• Mean of StdDevs: {mean_of_stds:.4f}°
• Coefficient of Variation: {coeff_var:.2f}%
• Range: {min_std:.4f}° - {max_std:.4f}°

Layer Extremes:
• Most variable: Layer {max_std_layer} ({max_std:.4f}°)
• Most consistent: Layer {min_std_layer} ({min_std:.4f}°)
• Difference: {max_std - min_std:.4f}°

Correlation Analysis:
• Mean-StdDev correlation: {correlation:.3f}
{'• Positive = higher angles → more variable' if correlation > 0.1 else '• Negative = higher angles → less variable' if correlation < -0.1 else '• No clear relationship'}

Overall Assessment: {consistency}

Interpretation:
• Your layers show {consistency.lower()} variability
• StdDev varies by only {std_of_stds:.4f}° across layers
• This indicates {('stable' if coeff_var < 20 else 'moderate' if coeff_var < 35 else 'unstable')} steering behavior
"""
    
    plt.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('std_of_stds_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('std_of_stds_analysis.pdf', bbox_inches='tight')
    
    print(f"\n{'='*60}")
    print("VISUALIZATION SAVED")
    print(f"{'='*60}")
    print("Files created:")
    print("  • std_of_stds_analysis.png")
    print("  • std_of_stds_analysis.pdf")
    
    plt.show()
    
    # Additional insights
    print(f"\n{'='*60}")
    print("DETAILED INSIGHTS")
    print(f"{'='*60}")
    
    print(f"1. CONSISTENCY METRIC:")
    print(f"   Coefficient of Variation = {coeff_var:.2f}%")
    if coeff_var < 15:
        print("   → EXCELLENT: Very consistent steering across layers")
    elif coeff_var < 30:
        print("   → GOOD: Reasonably consistent with some variation")
    else:
        print("   → POOR: High variability indicates unstable steering")
    
    print(f"\n2. LAYER ANALYSIS:")
    outlier_threshold = mean_of_stds + std_of_stds
    outlier_layers = [layers[i] for i, std in enumerate(layer_stds) if std > outlier_threshold]
    
    if outlier_layers:
        print(f"   Outlier layers (StdDev > {outlier_threshold:.3f}°): {outlier_layers}")
        print("   → These layers may need attention")
    else:
        print("   No outlier layers detected")
        print("   → All layers within normal variability range")
    
    print(f"\n3. CORRELATION INSIGHT:")
    if abs(correlation) > 0.3:
        direction = "positive" if correlation > 0 else "negative"
        print(f"   Strong {direction} correlation ({correlation:.3f})")
        if correlation > 0:
            print("   → Layers with higher mean angles are MORE variable")
        else:
            print("   → Layers with higher mean angles are LESS variable")
    else:
        print(f"   Weak correlation ({correlation:.3f})")
        print("   → Mean angle doesn't predict variability")
    
    return {
        'std_of_stds': std_of_stds,
        'mean_of_stds': mean_of_stds,
        'coefficient_of_variation': coeff_var,
        'consistency_rating': consistency,
        'correlation': correlation
    }

if __name__ == "__main__":
    analyze_std_variability_from_json()