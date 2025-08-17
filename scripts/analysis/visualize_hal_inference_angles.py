#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import torch
import librosa
import pandas as pd
import argparse
import csv
import time
import random
from tqdm import tqdm
from datasets import load_dataset
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

# Import steering and orthogonal layer functions
sys.path.insert(0, project_root)
from src.models.steering_vector import obtain_vsv

# Try to import orthogonal layer functions
try:
    from src.layers.variants.llm_layer_orthogonal import (
        add_vsv_layers, remove_vsv_layers, clear_angle_storage, 
        get_angle_statistics, get_layer_by_layer_statistics
    )
    ORTHOGONAL_LAYERS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to regular layers if orthogonal not available
        from src.layers.llm_layer import add_vsv_layers, remove_vsv_layers
        ORTHOGONAL_LAYERS_AVAILABLE = False
        print("Warning: Orthogonal layers not available. Angle measurements will not work.")
    except ImportError:
        print("Error: Neither orthogonal nor regular layers available!")
        ORTHOGONAL_LAYERS_AVAILABLE = False

# Global variables
model = None
processor = None
verbose_progress = False

def initialize_model():
    """Initialize the model and processor globally"""
    global model, processor
    
    MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
    
    print("🚀 Initializing model...")
    
    print("  📦 Loading configuration...")
    config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
    
    print("  🤖 Loading model (this may take a few minutes)...")
    model = Qwen2AudioSLAForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    
    print("  🔧 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    print("  ⚡ Enabling SLA...")
    # Enable SLA (as used in the existing codebase)
    model.enable_sla(gamma=0.0, w=4)
    
    print("✅ Model initialization complete!")

def build_messages(include_audio: bool, wav_path: str, prompt: str):
    """Build messages for VSV computation"""
    base = []
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
            "content": [
                {"type": "text", "text": prompt},
                # No audio content here (this is the 'neg' case)
            ],
        })
    return base

def build_inputs(messages, audio=None, sr=16000):
    """Build model inputs from messages"""
    global processor, model
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if audio is None:
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=prompt,
            audios=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
    # Move tensors to model device
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs

def compute_vsv_for_audio(audio_path, prompt):
    """Compute VSV for a specific audio file"""
    global model, processor, verbose_progress
    
    if verbose_progress:
        print("    🎯 Computing vector steering vector...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Create soundless audio with same length as original for negative instance
    soundless_audio = np.zeros_like(audio)
    
    # Use the prompt for VSV computation
    vsv_prompt = f"{prompt} Answer just yes or no."
    
    # Build positive and negative inputs for VSV computation
    messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    messages_neg = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    
    pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, audio=soundless_audio, sr=16000)
    
    # Compute VSV specific to this input
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    if verbose_progress:
        print(f"    ✅ VSV computed with shape: {vsv.shape}")
    
    return vsv

def hal_inference_with_angle_collection(audio_path, prompt_text, lam=0.05):
    """
    Perform HAL inference with angle collection enabled.
    This uses the orthogonal steering layers to measure angles between 
    steering vectors and original hidden states during inference.
    
    The angles measured are independent of lambda - they show the geometric
    relationship between the steering vector and hidden states.
    """
    global model, processor, verbose_progress
    
    if not ORTHOGONAL_LAYERS_AVAILABLE:
        print("Error: Orthogonal layers not available for angle measurement!")
        return "No", {}, {}
    
    if model is None or processor is None:
        initialize_model()
    
    # Clear previous measurements
    clear_angle_storage()
    
    if verbose_progress:
        print(f"  🎵 Processing: {os.path.basename(audio_path)}")
    
    vsv_applied = False
    try:
        if verbose_progress:
            print("    🎯 Computing vector steering vector...")
        
        # Compute VSV for this specific audio using the actual prompt
        vsv = compute_vsv_for_audio(audio_path, prompt_text)
        
        if verbose_progress:
            print("    📐 Applying orthogonal steering layers (angle measurement enabled)...")
        
        # Apply orthogonal VSV layers - this will measure angles during forward pass
        # The angle measurement happens regardless of lambda value
        add_vsv_layers(model, vsv=vsv, lam=lam, which_stack="decoder")
        vsv_applied = True
        
        if verbose_progress:
            print(f"    ✅ Orthogonal steering applied with λ={lam} (angles will be measured)")
        
        # Build messages in the expected format
        modified_prompt = f"{prompt_text} Answer just yes or no."
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": modified_prompt},
            ]},
        ]

        if verbose_progress:
            print("    📝 Applying chat template...")
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if verbose_progress:
            print("    🎧 Loading and processing audio...")
        # Process audio
        audios = []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio, sr = librosa.load(ele["audio_url"])
                        if sr != 16000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        audios.append(audio)

        if verbose_progress:
            print("    🔧 Preparing model inputs...")
        # Prepare inputs
        inputs = processor(text=text, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        if verbose_progress:
            print("    🧠 Generating response (measuring angles between steering vector and hidden states)...")
        
        # Generate response - the orthogonal layers will measure angles during forward pass
        # These angles show the geometric relationship between VSV and hidden states
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)

        if verbose_progress:
            print("    📤 Decoding output...")
        # Decode output
        output = output[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Clean and normalize response to Yes/No
        response = response.strip().lower()
        
        # Extract Yes/No from response
        if 'yes' in response:
            result = "Yes"
        elif 'no' in response:
            result = "No"
        else:
            # Default to "No" if unclear
            result = "No"
        
        if verbose_progress:
            print(f"    ✅ Response: {result}")
        
        # Get the collected angle statistics (independent of lambda)
        layer_stats = get_layer_by_layer_statistics()
        overall_stats = get_angle_statistics()
        
        if verbose_progress and overall_stats:
            print(f"    📐 Collected {overall_stats['count']} angle measurements")
            print(f"    📊 Mean angle: {overall_stats['mean']:.2f}° (independent of λ={lam})")
        
        return result, layer_stats, overall_stats
            
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {e}"
        if verbose_progress:
            print(f"    ❌ {error_msg}")
        else:
            print(error_msg)
        return "No", {}, {}
    
    finally:
        # Always remove VSV hooks after inference
        if vsv_applied:
            remove_vsv_layers(model, which_stack="decoder")
            if verbose_progress:
                print("    🔄 Orthogonal steering hooks removed")

def visualize_angle_statistics(layer_stats, overall_stats, save_path="hal_inference_angle_analysis"):
    """Create comprehensive visualizations of angle statistics from HAL inference"""
    
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
    plt.title('VSV-Hidden State Angles per Layer\n(λ-independent geometric relationship)')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers[::max(1, len(layers)//10)])  # Show every 10th layer if many layers
    
    # 2. Layer-wise min/max range
    plt.subplot(2, 3, 2)
    plt.fill_between(layers, layer_mins, layer_maxs, alpha=0.3, label='Min-Max Range')
    plt.plot(layers, layer_means, 'ro-', label='Mean')
    plt.xlabel('Layer Index')
    plt.ylabel('Angle (degrees)')
    plt.title('VSV-Hidden State Angle Range per Layer')
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
    plt.title('Overall VSV-Hidden State Angle Distribution\n(measures orthogonality relationship)')
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
    VSV-Hidden State Angle Analysis:
    
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
    
    About these measurements:
    • Angles between steering vector & hidden states
    • Geometric relationship (λ-independent)
    • 90° = perfectly orthogonal vectors
    • 0°/180° = aligned/anti-aligned vectors
    """
    plt.text(0.1, 0.9, stats_text, fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')  # Also save as PDF
    
    print(f"HAL inference angle visualization saved as {save_path}.png and {save_path}.pdf")
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
            'layer_range': [min(layers), max(layers)],
            'analysis_type': 'hal_inference_angles'
        }
    }
    
    with open(f'{save_path}_data.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"Raw HAL inference angle data saved as {save_path}_data.json")

def load_local_dataset(file_path):
    """Load dataset from local TSV file"""
    print(f"📂 Loading local dataset from: {file_path}")
    
    # Read the TSV file, skipping comment lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (first line without #)
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    header = data_lines[0].split('\t')
    
    # Parse data
    data = []
    for line in data_lines[1:]:
        if line:  # Skip empty lines
            fields = line.split('\t')
            if len(fields) >= 6:  # Ensure we have all required fields
                data.append({
                    'entry_id': fields[0],
                    'audio_index': fields[1], 
                    'prompt_text': fields[2],
                    'object': fields[3],
                    'attribute': fields[4],
                    'label': fields[5],
                    'sampling': fields[6] if len(fields) > 6 else 'unknown'
                })
    
    print(f"📊 Loaded {len(data)} samples")
    return data

def run_hal_inference_angle_analysis(args):
    """Run HAL inference with angle analysis on dataset samples"""
    global verbose_progress
    verbose_progress = args.verbose

    # Check if using local dataset file or HuggingFace dataset
    if hasattr(args, 'dataset_file') and args.dataset_file:
        print("📊 Loading local dataset...")
        dataset_samples = load_local_dataset(args.dataset_file)
        
        # Randomly shuffle the dataset
        print("🔀 Randomly shuffling dataset...")
        random.shuffle(dataset_samples)
        
        # Limit samples if specified (for angle analysis, we typically use fewer samples)
        max_samples = args.max_samples if hasattr(args, 'max_samples') and args.max_samples else 5
        if max_samples < len(dataset_samples):
            dataset_samples = dataset_samples[:max_samples]
            print(f"🔢 Limited to {max_samples} samples for angle analysis")
        
        use_local_dataset = True
    else:
        print("📊 Loading HuggingFace dataset...")
        dataset = load_dataset(args.dataset_name)
        dataset_samples = list(dataset['test'])
        
        # Randomly shuffle and limit for angle analysis
        print("🔀 Randomly shuffling dataset...")
        random.shuffle(dataset_samples)
        max_samples = args.max_samples if hasattr(args, 'max_samples') and args.max_samples else 5
        dataset_samples = dataset_samples[:max_samples]
        use_local_dataset = False
        print(f"📝 Dataset loaded: {len(dataset_samples)} samples for angle analysis")

    # Initialize model
    if model is None:
        initialize_model()

    print(f"🎯 Starting HAL inference with angle analysis on {len(dataset_samples)} samples...")
    start_time = time.time()
    
    # Collect all angle statistics across samples
    all_layer_stats = {}
    all_angle_measurements = []
    evaluation_results = []
    
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Processing samples with angle analysis", unit="sample")):
        # Entry ID for the dataset
        entry_id = sample["entry_id"]
        audio_index = sample["audio_index"]
        audio_path = f"{args.audio_root_dir}/{audio_index}.wav"
        prompt_text = sample["prompt_text"]
        label = sample["label"]
        
        print(f"\n🎵 Sample {idx+1}/{len(dataset_samples)}: {entry_id}")
        print(f"   Audio: {audio_index}.wav")
        print(f"   Prompt: {prompt_text}")
        print(f"   Ground truth: {label}")
        
        # Perform inference with angle collection
        response, layer_stats, overall_stats = hal_inference_with_angle_collection(
            audio_path=audio_path, 
            prompt_text=prompt_text, 
            lam=args.vsv_lambda
        )
        
        print(f"   Model response: {response}")
        print(f"   Correct: {'✅' if response == label else '❌'}")
        
        # Accumulate statistics
        if layer_stats and overall_stats:
            print(f"   Angle measurements collected: {overall_stats['count']}")
            all_angle_measurements.extend(overall_stats['all_angles'])
            
            # Accumulate layer statistics
            for layer_id, stats in layer_stats.items():
                if layer_id not in all_layer_stats:
                    all_layer_stats[layer_id] = {
                        'angles': [],
                        'counts': 0
                    }
                all_layer_stats[layer_id]['angles'].extend(stats['angles'])
                all_layer_stats[layer_id]['counts'] += len(stats['angles'])
        
        # Record evaluation result
        evaluation_results.append({
            'entry_id': entry_id,
            'audio_index': audio_index,
            'label': label,
            'response': response,
            'correct': response == label,
            'angle_measurements': overall_stats['count'] if overall_stats else 0
        })
    
    # Compute aggregated statistics
    print("\n📊 Computing aggregated angle statistics...")
    
    # Compute overall aggregated statistics
    aggregated_overall_stats = {
        'all_angles': all_angle_measurements,
        'count': len(all_angle_measurements),
        'mean': np.mean(all_angle_measurements) if all_angle_measurements else 0,
        'std': np.std(all_angle_measurements) if all_angle_measurements else 0,
        'min': np.min(all_angle_measurements) if all_angle_measurements else 0,
        'max': np.max(all_angle_measurements) if all_angle_measurements else 0
    }
    
    # Compute aggregated layer statistics
    aggregated_layer_stats = {}
    for layer_id, layer_data in all_layer_stats.items():
        angles = layer_data['angles']
        aggregated_layer_stats[layer_id] = {
            'angles': angles,
            'count': len(angles),
            'mean': np.mean(angles),
            'std': np.std(angles),
            'min': np.min(angles),
            'max': np.max(angles)
        }
    
    # Calculate final evaluation statistics
    total_time = time.time() - start_time
    correct = sum(1 for result in evaluation_results if result['correct'])
    final_accuracy = correct / len(evaluation_results) * 100
    
    print(f"\n🏁 HAL inference angle analysis completed!")
    print(f"  📊 Final accuracy: {final_accuracy:.2f}% ({correct}/{len(evaluation_results)})")
    print(f"  ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"  🎯 Total angle measurements: {aggregated_overall_stats['count']}")
    print(f"  📈 Mean angle: {aggregated_overall_stats['mean']:.2f}°")
    print(f"  📉 Std angle: {aggregated_overall_stats['std']:.2f}°")
    
    # Create visualizations
    if aggregated_layer_stats and aggregated_overall_stats['count'] > 0:
        print("\n📊 Creating angle visualizations...")
        save_path = f"hal_inference_angle_analysis_{len(dataset_samples)}samples_lambda{args.vsv_lambda}"
        visualize_angle_statistics(
            aggregated_layer_stats, 
            aggregated_overall_stats, 
            save_path=save_path
        )
        
        # Save detailed results
        detailed_results = {
            'evaluation_results': evaluation_results,
            'angle_analysis': {
                'layer_statistics': aggregated_layer_stats,
                'overall_statistics': aggregated_overall_stats
            },
            'experiment_config': {
                'num_samples': len(dataset_samples),
                'vsv_lambda': args.vsv_lambda,
                'dataset_file': args.dataset_file if hasattr(args, 'dataset_file') else args.dataset_name,
                'audio_root_dir': args.audio_root_dir
            },
            'summary': {
                'accuracy': final_accuracy,
                'total_time_minutes': total_time/60,
                'total_angle_measurements': aggregated_overall_stats['count'],
                'mean_angle': aggregated_overall_stats['mean'],
                'std_angle': aggregated_overall_stats['std']
            }
        }
        
        with open(f'{save_path}_detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"💾 Detailed results saved as {save_path}_detailed_results.json")
        
    else:
        print("❌ No angle statistics were collected!")

def main():
    parser = argparse.ArgumentParser(description="HAL inference with angle analysis visualization")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name.", default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, help="Path to local dataset TSV file (alternative to --dataset_name)", default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, help="Audio root directory", default="./understanding_sound_data/audio")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output for individual inference steps")
    
    # Vector steering options
    parser.add_argument("--vsv_lambda", type=float, default=0.05, help="Vector steering strength (lambda). Higher values = stronger steering. Default: 0.05")
    
    # Analysis options
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to process for angle analysis (default: 5)")
    
    args = parser.parse_args()
    run_hal_inference_angle_analysis(args)

if __name__ == "__main__":
    main()