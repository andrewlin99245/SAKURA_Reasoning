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
    # Clear previous measurements
    clear_angle_storage()
    
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
        output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=128, thinker_do_sample=False)  # Shorter for angle collection
    
    # Remove VSV layers
    remove_vsv_layers(model, which_stack="decoder")
    
    # Get the collected angle statistics
    layer_stats = get_layer_by_layer_statistics()
    overall_stats = get_angle_statistics()
    
    return layer_stats, overall_stats

def visualize_angle_statistics(layer_stats, overall_stats, save_path="angle_analysis_qwen2_5omni"):
    """Create comprehensive visualizations of angle statistics for Qwen2.5Omni"""
    
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
    
    # 5. Cumulative angle progression
    plt.subplot(2, 3, 5)
    cumulative_angles = []
    for i, layer in enumerate(layers):
        if i == 0:
            cumulative_angles.append(layer_means[i])
        else:
            cumulative_angles.append(cumulative_angles[-1] + layer_means[i])
    
    plt.plot(layers, cumulative_angles, 'bo-')
    plt.xlabel('Layer Index')
    plt.ylabel('Cumulative Angle (degrees)')
    plt.title('Cumulative Angle Progression')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers[::max(1, len(layers)//10)])
    
    # 6. Statistics summary (text)
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = f"""
    Overall Statistics (Qwen2.5Omni-3B):
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
    
    print(f"Qwen2.5Omni angle visualization saved as {save_path}.png and {save_path}.pdf")
    plt.show()
    
    # Save raw data as JSON
    analysis_data = {
        'model': 'Qwen2.5Omni-3B',
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
    
    print(f"Raw Qwen2.5Omni angle data saved as {save_path}_data.json")

def run_angle_analysis_demo():
    """Run a demo analysis on a sample audio file using Qwen2.5Omni"""
    
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
    
    print("Setting up Qwen2.5Omni model...")
    model, processor = setup_model()
    
    print("Running Qwen2.5Omni inference with angle collection...")
    layer_stats, overall_stats = run_sample_with_angle_collection(
        model, processor, sample_audio, sample_prompt, lam=0.05
    )
    
    if layer_stats and overall_stats:
        print("\nCreating Qwen2.5Omni visualizations...")
        visualize_angle_statistics(layer_stats, overall_stats, 
                                 save_path="qwen2_5omni_orthogonal_steering_angle_analysis")
    else:
        print("No angle statistics were collected!")

if __name__ == "__main__":
    run_angle_analysis_demo()