#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor

# Add path to cache config and model files
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up 3 levels from scripts/analysis/
sys.path.append(os.path.join(project_root, 'src', 'utils'))
sys.path.append(os.path.join(project_root, 'src', 'models'))
sys.path.append(os.path.join(project_root, 'src', 'layers', 'variants'))

from cache_config import set_hf_cache_env
from qwen_omni_utils import process_mm_info

# Configure cache to use local directory to avoid permission issues
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)

# Import Qwen2.5Omni specific modules
from Qwen2_5Omni_patch import Qwen2_5OmniSLAForCausalLM
from steering_vector_qwen2_5omni import obtain_vsv
from llm_layer_orthogonal import (
    add_vsv_layers, remove_vsv_layers, clear_angle_storage, 
    get_angle_statistics, get_layer_by_layer_statistics
)

def setup_model():
    """Setup Qwen2.5Omni model for angle analysis"""
    MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
    
    model = Qwen2_5OmniSLAForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    
    # Enable SLA
    model.enable_sla(gamma=0.0, w=4)
    
    return model, processor

def build_messages(include_audio: bool, audio_path: str, prompt: str):
    """Build message structure for Qwen2.5Omni"""
    base = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
    ]
    if include_audio:
        base.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        base.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        })
    return base

def build_inputs(messages, processor, model, audio=None):
    """Build inputs for Qwen2.5Omni model"""
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if audio is None:
        # For negative case - no multimodal inputs
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True
        )
    else:
        # For positive case - use the provided audio data
        inputs = processor(
            text=prompt,
            audio=[audio],  # Use the actual audio data passed as parameter
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
    
    # Move tensors to model device
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs

def run_sample_with_angle_collection(model, processor, audio_path, prompt, lam=0.05):
    """Run a single sample and collect angle statistics for Qwen2.5Omni"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Build positive and negative inputs for VSV computation
        vsv_prompt = "Describe the audio in detail."
        messages_pos = build_messages(include_audio=True,  audio_path=audio_path, prompt=vsv_prompt)
        messages_neg = build_messages(include_audio=False, audio_path=audio_path, prompt=vsv_prompt)
        
        pos_inputs = build_inputs(messages_pos, processor, model, audio=audio)
        neg_inputs = build_inputs(messages_neg, processor, model, audio=None)
        
        # Compute VSV
        with torch.no_grad():
            kwargs_list = [[neg_inputs, pos_inputs]]
            vsv = obtain_vsv(model, kwargs_list)
            vsv = vsv.to(model.device)
        
        # Add VSV layers (this will start collecting angle measurements)
        add_vsv_layers(model, vsv=vsv, lam=lam, which_stack="decoder")
        
        # Perform inference (this triggers the angle measurements)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, audio=[audio], return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():
            output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=64, thinker_do_sample=False)  # Shorter for speed
        
        # Remove VSV layers
        remove_vsv_layers(model, which_stack="decoder")
        
        return True
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Make sure to remove VSV layers even on error
        try:
            remove_vsv_layers(model, which_stack="decoder")
        except:
            pass
        return False

def create_angle_visualization(layer_stats, overall_stats, save_path="angle_analysis_qwen2_5omni"):
    """Create comprehensive visualizations of angle statistics for Qwen2.5Omni"""
    
    if not layer_stats or not overall_stats:
        print("No angle statistics to visualize!")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Extract data
    layers = sorted(layer_stats.keys())
    layer_means = [layer_stats[l]['mean'] for l in layers]
    layer_stds = [layer_stats[l]['std'] for l in layers]
    layer_mins = [layer_stats[l]['min'] for l in layers]
    layer_maxs = [layer_stats[l]['max'] for l in layers]
    
    # 1. Layer-wise mean angles with error bars
    plt.subplot(2, 3, 1)
    plt.errorbar(layers, layer_means, yerr=layer_stds, marker='o', capsize=5, capthick=2, markersize=3)
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Mean Steering Vector Angles per Layer (Qwen2.5Omni)')
    plt.grid(True, alpha=0.3)
    
    # 2. Layer-wise min/max range
    plt.subplot(2, 3, 2)
    plt.fill_between(layers, layer_mins, layer_maxs, alpha=0.3, label='Min-Max Range')
    plt.plot(layers, layer_means, 'ro-', markersize=3, label='Mean')
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Angle Range per Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Overall angle distribution (histogram)
    plt.subplot(2, 3, 3)
    all_angles = overall_stats['all_angles']
    plt.hist(all_angles, bins=min(30, len(all_angles)//3), alpha=0.7, edgecolor='black')
    plt.axvline(overall_stats['mean'], color='red', linestyle='--', 
                label=f'Mean: {overall_stats["mean"]:.2f}°')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Overall Angle Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Angle progression with trend
    plt.subplot(2, 3, 4)
    plt.plot(layers, layer_means, 'bo-', markersize=4)
    # Add trend line if enough data points
    if len(layers) > 2:
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
    
    max_mean_layer = layers[layer_means.index(max(layer_means))]
    min_mean_layer = layers[layer_means.index(min(layer_means))]
    
    stats_text = f"""
Qwen2.5Omni-3B Angle Analysis:

Total measurements: {overall_stats['count']}
Mean angle: {overall_stats['mean']:.2f}°
Std deviation: {overall_stats['std']:.2f}°
Range: {overall_stats['min']:.1f}° - {overall_stats['max']:.1f}°

Layers analyzed: {len(layers)}
Highest mean: {max(layer_means):.2f}° (Layer {max_mean_layer})
Lowest mean: {min(layer_means):.2f}° (Layer {min_mean_layer})

Interpretation:
• < 90°: Steering vector aligns with hidden states
• > 90°: Steering vector opposes hidden states  
• High std: Inconsistent alignment across samples

Note: Qwen2.5Omni appears to have {len(layers)} 
transformer layers in the language model.
"""
    
    plt.text(0.1, 0.9, stats_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    print(f"Qwen2.5Omni angle visualization saved as {save_path}.png")
    plt.show()

def run_quick_angle_analysis():
    """Run angle analysis on 3 examples from each dataset type for Qwen2.5Omni"""
    
    SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/data"
    
    print("Setting up Qwen2.5Omni model...")
    model, processor = setup_model()
    
    # Clear any existing angle measurements
    clear_angle_storage()
    
    samples_processed = 0
    total_samples = 0
    
    # Process 3 examples from each dataset
    for subset in ["Animal", "Emotion", "Gender", "Language"]:
        metadata_path = f"{SAKURA_DATA_DIR}/{subset}/metadata.csv"
        
        if not os.path.exists(metadata_path):
            print(f"Skipping {subset}: metadata not found")
            continue
            
        print(f"\nProcessing {subset} dataset...")
        df = pd.read_csv(metadata_path)
        
        # Take first 3 examples
        for i in range(min(3, len(df))):
            audio_file = df.iloc[i]["file"]
            prompt = df.iloc[i]["single_instruction"]  # Use single-hop for speed
            audio_path = f"{SAKURA_DATA_DIR}/{subset}/audio/{audio_file}"
            
            if not os.path.exists(audio_path):
                print(f"  Skipping {audio_file}: audio file not found")
                continue
            
            print(f"  Processing sample {i+1}: {audio_file}")
            total_samples += 1
            
            success = run_sample_with_angle_collection(model, processor, audio_path, prompt, lam=0.05)
            if success:
                samples_processed += 1
                print(f"    ✓ Collected angle data")
            else:
                print(f"    ✗ Failed to collect angle data")
    
    print(f"\n{'='*50}")
    print(f"QWEN2.5OMNI ANGLE COLLECTION COMPLETE")
    print(f"{'='*50}")
    print(f"Samples processed successfully: {samples_processed}/{total_samples}")
    
    # Get collected angle statistics
    layer_stats = get_layer_by_layer_statistics()
    overall_stats = get_angle_statistics()
    
    if layer_stats and overall_stats:
        print(f"Total angle measurements collected: {overall_stats['count']}")
        print(f"Layers analyzed: {len(layer_stats)}")
        
        # Create visualizations
        print("\nGenerating Qwen2.5Omni angle visualizations...")
        create_angle_visualization(layer_stats, overall_stats, 
                                 save_path="quick_qwen2_5omni_orthogonal_angle_analysis")
        
        # Print summary statistics
        print(f"\n{'='*50}")
        print("QWEN2.5OMNI ANGLE STATISTICS SUMMARY")
        print(f"{'='*50}")
        print(f"Mean angle: {overall_stats['mean']:.2f}° ± {overall_stats['std']:.2f}°")
        print(f"Angle range: {overall_stats['min']:.1f}° to {overall_stats['max']:.1f}°")
        
        # Layer analysis
        layers = sorted(layer_stats.keys())
        layer_means = [layer_stats[l]['mean'] for l in layers]
        max_layer = layers[layer_means.index(max(layer_means))]
        min_layer = layers[layer_means.index(min(layer_means))]
        
        print(f"Layers analyzed: {len(layers)} (Layer {min(layers)} to {max(layers)})")
        print(f"Highest mean angle: {max(layer_means):.2f}° at Layer {max_layer}")
        print(f"Lowest mean angle: {min(layer_means):.2f}° at Layer {min_layer}")
        
        # Save detailed data
        analysis_data = {
            'model': 'Qwen2.5Omni-3B',
            'samples_processed': samples_processed,
            'total_samples': total_samples,
            'layer_statistics': layer_stats,
            'overall_statistics': overall_stats,
        }
        
        with open('quick_qwen2_5omni_angle_analysis_data.json', 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"Detailed Qwen2.5Omni data saved to 'quick_qwen2_5omni_angle_analysis_data.json'")
        
    else:
        print("No angle statistics were collected!")

if __name__ == "__main__":
    run_quick_angle_analysis()