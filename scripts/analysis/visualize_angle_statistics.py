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
from transformers import AutoProcessor

# Add path to cache config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up 3 levels from scripts/analysis/
sys.path.append(os.path.join(project_root, 'src', 'utils'))
from Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig
sys.path.append(os.path.join(project_root, 'src', 'utils'))
from cache_config import set_hf_cache_env

# Configure shared cache
set_hf_cache_env()

# Import orthogonal layer functions
sys.path.insert(0, project_root)
from src.models.steering_vector import obtain_vsv
from src.layers.variants.llm_layer_orthogonal import (
    add_vsv_layers, remove_vsv_layers, clear_angle_storage, 
    get_angle_statistics, get_layer_by_layer_statistics
)

def setup_model():
    """Setup model for angle analysis"""
    MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
    
    config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
    model = Qwen2AudioSLAForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Enable SLA
    model.enable_sla(gamma=0.0, w=4)
    
    return model, processor

def build_messages(include_audio: bool, wav_path: str, prompt: str):
    base = [{"role": "system", "content": "You are a helpful assistant."}]
    if include_audio:
        base.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": wav_path},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        base.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })
    return base

def build_inputs(messages, processor, model, audio=None, sr=16000):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if audio is None:
        inputs = processor(text=prompt, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
    
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs

def run_sample_with_angle_collection(model, processor, audio_path, prompt, lam=0.05):
    """Run a single sample and collect angle statistics"""
    # Clear previous measurements
    clear_angle_storage()
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Build positive and negative inputs for VSV computation
    messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=prompt)
    messages_neg = build_messages(include_audio=False, wav_path=audio_path, prompt=prompt)
    
    pos_inputs = build_inputs(messages_pos, processor, model, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, processor, model, audio=None, sr=16000)
    
    # Compute VSV
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    # Add VSV layers (this will start collecting angle measurements)
    add_vsv_layers(model, vsv=vsv, lam=lam, which_stack="decoder")
    
    # Perform inference (this triggers the angle measurements)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audios=[audio], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)  # Shorter for angle collection
    
    # Remove VSV layers
    remove_vsv_layers(model, which_stack="decoder")
    
    # Get the collected angle statistics
    layer_stats = get_layer_by_layer_statistics()
    overall_stats = get_angle_statistics()
    
    return layer_stats, overall_stats

def visualize_angle_statistics(layer_stats, overall_stats, save_path="angle_analysis"):
    """Create comprehensive visualizations of angle statistics"""
    
    if not layer_stats or not overall_stats:
        print("No angle statistics to visualize!")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Extract data
    layers = sorted(layer_stats.keys())
    layer_means = [layer_stats[l]['mean'] for l in layers]
    layer_stds = [layer_stats[l]['std'] for l in layers]
    layer_mins = [layer_stats[l]['min'] for l in layers]
    layer_maxs = [layer_stats[l]['max'] for l in layers]
    
    # 1. Layer-wise mean angles with error bars
    plt.subplot(2, 3, 1)
    plt.errorbar(layers, layer_means, yerr=layer_stds, marker='o', capsize=5, capthick=2)
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Mean Angles per Layer (with std dev)')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers[::max(1, len(layers)//10)])  # Show every 10th layer if many layers
    
    # 2. Layer-wise min/max range
    plt.subplot(2, 3, 2)
    plt.fill_between(layers, layer_mins, layer_maxs, alpha=0.3, label='Min-Max Range')
    plt.plot(layers, layer_means, 'ro-', label='Mean')
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Angle Range per Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(layers[::max(1, len(layers)//10)])
    
    # 3. Overall angle distribution (histogram)
    plt.subplot(2, 3, 3)
    all_angles = overall_stats['all_angles']
    plt.hist(all_angles, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(overall_stats['mean'], color='red', linestyle='--', 
                label=f'Mean: {overall_stats["mean"]:.2f}°')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Overall Angle Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Box plot per layer (showing distribution)
    plt.subplot(2, 3, 4)
    layer_angle_data = [layer_stats[l]['angles'] for l in layers]
    plt.boxplot(layer_angle_data, labels=layers)
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Angle Distribution per Layer (Box Plot)')
    plt.xticks(rotation=45)
    if len(layers) > 15:  # If too many layers, show fewer labels
        plt.xticks(range(1, len(layers)+1, max(1, len(layers)//10)))
    plt.grid(True, alpha=0.3)
    
    # 5. Standard deviation of angles per layer
    plt.subplot(2, 3, 5)
    plt.plot(layers, layer_stds, 'go-', marker='s')
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Standard Deviation of Angles per Layer')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers[::max(1, len(layers)//10)])
    
    # 6. Statistics summary (text)
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = f"""
    Overall Statistics:
    • Total measurements: {overall_stats['count']}
    • Mean angle: {overall_stats['mean']:.2f}°
    • Std deviation: {overall_stats['std']:.2f}°
    • Min angle: {overall_stats['min']:.2f}°
    • Max angle: {overall_stats['max']:.2f}°
    
    Layer Statistics:
    • Number of layers: {len(layers)}
    • Layer range: {min(layers)} - {max(layers)}
    • Highest mean angle: {max(layer_means):.2f}° (Layer {layers[layer_means.index(max(layer_means))]})
    • Lowest mean angle: {min(layer_means):.2f}° (Layer {layers[layer_means.index(min(layer_means))]})
    """
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')  # Also save as PDF
    
    print(f"Angle visualization saved as {save_path}.png and {save_path}.pdf")
    plt.show()
    
    # Save raw data as JSON
    analysis_data = {
        'layer_statistics': layer_stats,
        'overall_statistics': overall_stats,
        'summary': {
            'total_measurements': overall_stats['count'],
            'mean_angle': overall_stats['mean'],
            'std_angle': overall_stats['std'],
            'min_angle': overall_stats['min'],
            'max_angle': overall_stats['max'],
            'num_layers': len(layers),
            'layer_range': [min(layers), max(layers)]
        }
    }
    
    with open(f'{save_path}_data.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"Raw angle data saved as {save_path}_data.json")

def run_angle_analysis_demo():
    """Run a demo analysis on a sample audio file"""
    
    # Check if we have sample data
    SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/data"
    
    # Try to find a sample audio file
    sample_audio = None
    sample_prompt = "What animal sound do you hear in this audio?"
    
    for subset in ["Animal", "Emotion", "Gender", "Language"]:
        metadata_path = f"{SAKURA_DATA_DIR}/{subset}/metadata.csv"
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            if len(df) > 0:
                audio_file = df.iloc[0]["file"]
                audio_path = f"{SAKURA_DATA_DIR}/{subset}/audio/{audio_file}"
                if os.path.exists(audio_path):
                    sample_audio = audio_path
                    sample_prompt = df.iloc[0]["single_instruction"]
                    print(f"Using sample audio: {audio_path}")
                    print(f"Using prompt: {sample_prompt}")
                    break
    
    if sample_audio is None:
        print("No sample audio file found. Please provide a path to an audio file.")
        return
    
    print("Setting up model...")
    model, processor = setup_model()
    
    print("Running inference with angle collection...")
    layer_stats, overall_stats = run_sample_with_angle_collection(
        model, processor, sample_audio, sample_prompt, lam=0.05
    )
    
    if layer_stats and overall_stats:
        print("\nCreating visualizations...")
        visualize_angle_statistics(layer_stats, overall_stats, 
                                 save_path="orthogonal_steering_angle_analysis")
    else:
        print("No angle statistics were collected!")

if __name__ == "__main__":
    run_angle_analysis_demo()