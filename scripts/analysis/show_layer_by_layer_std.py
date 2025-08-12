#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np

def show_layer_by_layer_std():
    """Display standard deviation for each layer individually"""
    
    # Load the angle analysis data
    try:
        with open('quick_angle_analysis_data.json', 'r') as f:
            data = json.load(f)
        layer_stats = data['layer_statistics']
    except FileNotFoundError:
        print("Error: quick_angle_analysis_data.json not found!")
        return
    
    # Extract data
    layers = sorted([int(k) for k in layer_stats.keys()])
    
    print("="*60)
    print("STANDARD DEVIATION BY LAYER")
    print("="*60)
    print(f"{'Layer':<6} {'StdDev (°)':<12} {'Mean (°)':<12} {'Count':<8} {'Status'}")
    print("-"*60)
    
    layer_stds = []
    for layer in layers:
        stats = layer_stats[str(layer)]
        std_val = stats['std']
        mean_val = stats['mean'] 
        count = stats['count']
        
        layer_stds.append(std_val)
        
        # Classify based on std dev magnitude
        if std_val < 2.0:
            status = "Very Stable"
        elif std_val < 4.0:
            status = "Stable"
        elif std_val < 6.0:
            status = "Variable"
        else:
            status = "Highly Variable"
        
        print(f"{layer:<6} {std_val:<12.4f} {mean_val:<12.2f} {count:<8} {status}")
    
    # Summary statistics
    mean_std = np.mean(layer_stds)
    std_of_stds = np.std(layer_stds)
    min_std = np.min(layer_stds)
    max_std = np.max(layer_stds)
    
    print("-"*60)
    print(f"{'SUMMARY':<6} {'Value':<12} {'Layer':<12}")
    print("-"*60)
    print(f"{'Mean':<6} {mean_std:<12.4f}")
    print(f"{'StdDev':<6} {std_of_stds:<12.4f}")
    print(f"{'Min':<6} {min_std:<12.4f} Layer {layers[layer_stds.index(min_std)]}")
    print(f"{'Max':<6} {max_std:<12.4f} Layer {layers[layer_stds.index(max_std)]}")
    print(f"{'Range':<6} {max_std - min_std:<12.4f}")
    
    # Create a simple bar chart
    plt.figure(figsize=(14, 6))
    bars = plt.bar(layers, layer_stds, alpha=0.7, edgecolor='black')
    
    # Color bars based on std dev level
    colors = []
    for std_val in layer_stds:
        if std_val < 2.0:
            colors.append('green')      # Very stable
        elif std_val < 4.0:
            colors.append('lightgreen') # Stable  
        elif std_val < 6.0:
            colors.append('orange')     # Variable
        else:
            colors.append('red')        # Highly variable
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add horizontal lines for reference
    plt.axhline(mean_std, color='blue', linestyle='--', linewidth=2, 
                label=f'Mean StdDev: {mean_std:.3f}°')
    plt.axhline(mean_std + std_of_stds, color='red', linestyle=':', alpha=0.7,
                label=f'Mean + 1σ: {mean_std + std_of_stds:.3f}°')
    plt.axhline(mean_std - std_of_stds, color='red', linestyle=':', alpha=0.7,
                label=f'Mean - 1σ: {mean_std - std_of_stds:.3f}°')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Standard Deviation by Layer\n(Green=Very Stable, Light Green=Stable, Orange=Variable, Red=Highly Variable)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)
    
    # Rotate x-axis labels if too many layers
    if len(layers) > 20:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('layer_by_layer_std.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'layer_by_layer_std.png'")
    
    # Identify problematic layers
    print(f"\n{'='*60}")
    print("LAYER CLASSIFICATION")
    print(f"{'='*60}")
    
    very_stable = [layer for layer, std in zip(layers, layer_stds) if std < 2.0]
    stable = [layer for layer, std in zip(layers, layer_stds) if 2.0 <= std < 4.0]
    variable = [layer for layer, std in zip(layers, layer_stds) if 4.0 <= std < 6.0]
    highly_variable = [layer for layer, std in zip(layers, layer_stds) if std >= 6.0]
    
    print(f"Very Stable (σ < 2.0°):      {very_stable}")
    print(f"Stable (2.0° ≤ σ < 4.0°):    {stable}")
    print(f"Variable (4.0° ≤ σ < 6.0°):  {variable}")
    print(f"Highly Variable (σ ≥ 6.0°):  {highly_variable}")
    
    if highly_variable:
        print(f"\n⚠️  ATTENTION: Layers {highly_variable} need tuning!")
    if variable:
        print(f"⚡ CONSIDER: Layers {variable} may benefit from adjustment")
    if very_stable:
        print(f"✅ EXCELLENT: Layers {very_stable} have very consistent steering")

if __name__ == "__main__":
    show_layer_by_layer_std()