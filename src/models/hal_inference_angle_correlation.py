import os
import sys

# CRITICAL: Set up cache environment variables BEFORE any other imports
# This ensures consistent cache usage across all modules
SHARED_CACHE_DIR = os.path.expanduser("~/.cache/sakura_reasoning")
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)

# Set all relevant HuggingFace cache environment variables
os.environ["HF_HOME"] = SHARED_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = SHARED_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["HF_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.cache")
os.environ["TORCH_HOME"] = SHARED_CACHE_DIR

# Unset conflicting variables that could cause cache issues
if "PYTORCH_CACHE_HOME" in os.environ:
    del os.environ["PYTORCH_CACHE_HOME"]

print(f"Cache configured: {SHARED_CACHE_DIR}")

# Now import everything else
import csv
import argparse
import torch
import librosa
import time
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from scipy.stats import pearsonr, spearmanr, ttest_ind

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

from utils.Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig

# Import vector steering modules
try:
    from steering_vector import obtain_vsv
    from ..layers.llm_layer import add_vsv_layers, remove_vsv_layers
    VSV_AVAILABLE = True
except ImportError as e:
    try:
        # Try absolute imports from project root
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.insert(0, project_dir)
        from src.models.steering_vector import obtain_vsv
        from src.layers.llm_layer import add_vsv_layers, remove_vsv_layers
        VSV_AVAILABLE = True
    except ImportError as e2:
        print(f"Warning: Vector steering modules not available: {e2}")
        print("Vector steering will be disabled.")
        VSV_AVAILABLE = False

# Try to import orthogonal layer functions for angle measurement
try:
    from ..layers.variants.llm_layer_orthogonal import (
        add_vsv_layers as add_orthogonal_vsv_layers, 
        remove_vsv_layers as remove_orthogonal_vsv_layers,
        clear_angle_storage, get_angle_statistics, get_layer_by_layer_statistics
    )
    ORTHOGONAL_LAYERS_AVAILABLE = True
    print("✅ Orthogonal layers available - angle measurements enabled")
except ImportError:
    try:
        # Try absolute import
        from src.layers.variants.llm_layer_orthogonal import (
            add_vsv_layers as add_orthogonal_vsv_layers,
            remove_vsv_layers as remove_orthogonal_vsv_layers, 
            clear_angle_storage, get_angle_statistics, get_layer_by_layer_statistics
        )
        ORTHOGONAL_LAYERS_AVAILABLE = True
        print("✅ Orthogonal layers available - angle measurements enabled")
    except ImportError:
        ORTHOGONAL_LAYERS_AVAILABLE = False
        print("❌ Warning: Orthogonal layers not available. Angle measurements will not work.")

# Global variables for model and processor
model = None
processor = None
verbose_progress = False
vsv_enabled = False
vsv_lambda = 1.0

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
    """Compute VSV for a specific audio file using the data_prompt as input for positive and negative instances"""
    global model, processor, verbose_progress
    
    if verbose_progress:
        print("    🎯 Computing vector steering vector...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Create soundless audio with same length as original for negative instance
    soundless_audio = np.zeros_like(audio)
    
    # Use the data_prompt (prompt parameter) for VSV computation
    vsv_prompt = f"{prompt} Answer just yes or no."
    
    # Build positive and negative inputs for VSV computation using the data_prompt
    messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    messages_neg = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)  # Changed to True to include audio
    
    pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, audio=soundless_audio, sr=16000)  # Use soundless audio instead of None
    
    # Compute VSV specific to this input
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    if verbose_progress:
        print(f"    ✅ VSV computed with shape: {vsv.shape}")
    
    return vsv

def inference_with_angle_measurement(audio_path, prompt_text, vsv_lambda=0.05):
    """
    Perform inference with angle measurement between steering vectors and hidden states.
    
    Returns:
        result: 'Yes' or 'No' prediction
        layer_stats: Per-layer angle statistics 
        overall_stats: Overall angle statistics
    """
    global model, processor, verbose_progress
    
    if not ORTHOGONAL_LAYERS_AVAILABLE:
        print("❌ Error: Orthogonal layers not available for angle measurement!")
        return "No", {}, {}
    
    if model is None or processor is None:
        initialize_model()
    
    # Clear previous angle measurements
    clear_angle_storage()
    
    if verbose_progress:
        print(f"  🎵 Processing: {os.path.basename(audio_path)}")
    
    vsv_applied = False
    try:
        # Apply vector steering with angle measurement enabled
        if verbose_progress:
            print("    🎯 Applying vector steering with angle measurement...")
        
        # Compute VSV for this specific audio using the actual prompt
        vsv = compute_vsv_for_audio(audio_path, prompt_text)
        
        # Apply orthogonal VSV layers - this will measure angles during forward pass
        add_orthogonal_vsv_layers(model, vsv=vsv, lam=vsv_lambda, which_stack="decoder")
        vsv_applied = True
        
        if verbose_progress:
            print(f"    ✅ Vector steering applied with λ={vsv_lambda} (angles will be measured)")
        
        # Build messages in the expected format
        # Append instruction to answer only yes or no
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
            print("    🧠 Generating response (measuring angles)...")
        # Generate response - angles are measured during forward pass
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)

        if verbose_progress:
            print("    📤 Decoding output...")
        # Decode output
        output = output[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Print the raw model generated response
        if verbose_progress:
            print(f"    Model Response: {response}")
        
        # Clean and normalize response to Yes/No
        response = response.strip().lower()
        
        # Extract Yes/No from response
        if 'yes' in response:
            result = "Yes"
        elif 'no' in response:
            result = "No"
        else:
            # Default to "No" if unclear (following paper's observation that models tend to give affirmative answers)
            result = "No"
        
        if verbose_progress:
            print(f"    ✅ Response: {result}")
        
        # Get the collected angle statistics
        layer_stats = get_layer_by_layer_statistics()
        overall_stats = get_angle_statistics()
        
        if verbose_progress and overall_stats:
            print(f"    📐 Collected {overall_stats['count']} angle measurements")
            print(f"    📊 Mean angle: {overall_stats['mean']:.2f}°")
        
        return result, layer_stats, overall_stats
            
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {e}"
        if verbose_progress:
            print(f"    ❌ {error_msg}")
        else:
            print(error_msg)
        return "No", {}, {}
    
    finally:
        # Always remove VSV hooks after inference to prevent interference
        if vsv_applied:
            remove_orthogonal_vsv_layers(model, which_stack="decoder")
            if verbose_progress:
                print("    🔄 Vector steering hooks removed")

def compute_layer_wise_angle_accuracy_analysis(evaluation_results):
    """
    Compute layer-by-layer analysis of angles vs prediction accuracy.
    
    Args:
        evaluation_results: List of dictionaries containing:
            - correct: bool, whether prediction was correct
            - layer_stats: dict, per-layer angle statistics
    
    Returns:
        layer_analysis: Dictionary with interpretable layer-wise statistics
    """
    if len(evaluation_results) < 2:
        print("❌ Need at least 2 samples for layer-wise analysis")
        return {}
    
    # Separate results by correctness
    correct_results = [r for r in evaluation_results if r['correct'] and r.get('layer_stats')]
    incorrect_results = [r for r in evaluation_results if not r['correct'] and r.get('layer_stats')]
    
    if not correct_results or not incorrect_results:
        print("❌ Need both correct and incorrect predictions for comparison")
        return {}
    
    # Find all layer IDs across all samples
    all_layer_ids = set()
    for result in evaluation_results:
        if result.get('layer_stats'):
            all_layer_ids.update(result['layer_stats'].keys())
    
    layer_analysis = {}
    
    for layer_id in sorted(all_layer_ids):
        # Collect angles for this layer from correct predictions
        correct_angles = []
        for result in correct_results:
            if layer_id in result['layer_stats']:
                correct_angles.extend(result['layer_stats'][layer_id].get('angles', []))
        
        # Collect angles for this layer from incorrect predictions
        incorrect_angles = []
        for result in incorrect_results:
            if layer_id in result['layer_stats']:
                incorrect_angles.extend(result['layer_stats'][layer_id].get('angles', []))
        
        if not correct_angles or not incorrect_angles:
            continue
            
        # Compute statistics for this layer
        correct_mean = np.mean(correct_angles)
        incorrect_mean = np.mean(incorrect_angles)
        correct_std = np.std(correct_angles)
        incorrect_std = np.std(incorrect_angles)
        correct_median = np.median(correct_angles)
        incorrect_median = np.median(incorrect_angles)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(correct_angles) - 1) * correct_std**2 + 
                             (len(incorrect_angles) - 1) * incorrect_std**2) / 
                            (len(correct_angles) + len(incorrect_angles) - 2))
        cohens_d = (correct_mean - incorrect_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test (Welch's t-test)
        t_stat, p_value = ttest_ind(correct_angles, incorrect_angles, equal_var=False)
        
        layer_analysis[layer_id] = {
            'correct_predictions': {
                'count': len(correct_angles),
                'mean_angle': correct_mean,
                'std_angle': correct_std,
                'median_angle': correct_median,
                'min_angle': np.min(correct_angles),
                'max_angle': np.max(correct_angles)
            },
            'incorrect_predictions': {
                'count': len(incorrect_angles),
                'mean_angle': incorrect_mean,
                'std_angle': incorrect_std,
                'median_angle': incorrect_median,
                'min_angle': np.min(incorrect_angles),
                'max_angle': np.max(incorrect_angles)
            },
            'comparison': {
                'mean_difference': correct_mean - incorrect_mean,
                'median_difference': correct_median - incorrect_median,
                'cohens_d': cohens_d,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size_interpretation': (
                    'Large' if abs(cohens_d) >= 0.8 else
                    'Medium' if abs(cohens_d) >= 0.5 else
                    'Small' if abs(cohens_d) >= 0.2 else
                    'Negligible'
                )
            }
        }
    
    return layer_analysis

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

def main(args):
    global verbose_progress, vsv_enabled, vsv_lambda
    verbose_progress = args.verbose
    vsv_enabled = True  # Always enable VSV for angle measurement
    vsv_lambda = args.vsv_lambda

    if not ORTHOGONAL_LAYERS_AVAILABLE:
        print("❌ Error: Orthogonal layers required for angle measurement!")
        print("Please ensure orthogonal layer variants are available.")
        return

    # Check if using local dataset file or HuggingFace dataset
    if hasattr(args, 'dataset_file') and args.dataset_file:
        print("📊 Loading local dataset...")
        dataset_samples = load_local_dataset(args.dataset_file)
        
        # Randomly shuffle the dataset
        print("🔀 Randomly shuffling dataset...")
        random.shuffle(dataset_samples)
        
        # Limit samples if specified
        if hasattr(args, 'max_samples') and args.max_samples and args.max_samples < len(dataset_samples):
            dataset_samples = dataset_samples[:args.max_samples]
            print(f"🔢 Limited to {args.max_samples} samples for testing")
        
        total_samples = len(dataset_samples)
        use_local_dataset = True
    else:
        print("📊 Loading HuggingFace dataset...")
        # Load the dataset.
        dataset = load_dataset(args.dataset_name)
        dataset_samples = list(dataset['test'])
        
        # Randomly shuffle the dataset
        print("🔀 Randomly shuffling dataset...")
        random.shuffle(dataset_samples)
        
        # Limit samples for angle analysis
        if hasattr(args, 'max_samples') and args.max_samples:
            dataset_samples = dataset_samples[:args.max_samples]
        
        total_samples = len(dataset_samples)
        use_local_dataset = False
        print(f"📝 Dataset loaded: {total_samples} samples to process")

    # Evaluation results.
    evaluation_results = []
    
    # Initialize model before processing (if not already initialized)
    if model is None:
        initialize_model()

    print(f"🎯 Vector steering ENABLED with λ={vsv_lambda} for angle measurement")
    print(f"🎯 Starting inference with angle-accuracy correlation analysis on {total_samples} samples...")
    start_time = time.time()
    
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Processing samples", unit="sample")):

        # Entry ID for the dataset.
        entry_id = sample["entry_id"]

        # The ID in AudioCaps, e.g., Y7fmOlUlwoNg corresponds to Y7fmOlUlwoNg.wav
        audio_index = sample["audio_index"]

        # The absolute path of audio.
        audio_path = f"{args.audio_root_dir}/{audio_index}.wav"

        # The input text prompt.
        prompt_text = sample["prompt_text"]

        # The correct answer corresponding to the prompt_text.
        label = sample["label"]

        # Get sampling method if available (for local datasets)
        sampling_method = sample.get("sampling", "unknown") if use_local_dataset else "unknown"

        # Inference model and get response with angle measurement.
        response, layer_stats, overall_stats = inference_with_angle_measurement(
            audio_path=audio_path, 
            prompt_text=prompt_text,
            vsv_lambda=vsv_lambda
        )

        # Determine if prediction was correct
        correct = (response == label)

        # Extract angle statistics for correlation analysis
        mean_angle = overall_stats.get('mean', None) if overall_stats else None
        std_angle = overall_stats.get('std', None) if overall_stats else None
        angle_count = overall_stats.get('count', 0) if overall_stats else 0

        # Record evaluation result with angle data
        result_data = {
            'entry_id': entry_id,
            'audio_index': audio_index,
            'label': label,
            'response': response,
            'correct': correct,
            'mean_angle': mean_angle,
            'std_angle': std_angle,
            'angle_count': angle_count,
            'layer_stats': layer_stats,  # Store layer stats for layer-wise analysis
            'vsv_lambda': vsv_lambda
        }
        
        if use_local_dataset:
            result_data['sampling_method'] = sampling_method
            
        evaluation_results.append(result_data)
        
        # Show progress every 10 samples or at the end
        if (idx + 1) % 10 == 0 or (idx + 1) == total_samples:
            correct_count = sum(1 for result in evaluation_results if result['correct'])
            accuracy = correct_count / len(evaluation_results) * 100
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / (idx + 1)
            estimated_total_time = avg_time_per_sample * total_samples
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"  📈 Progress: {idx + 1}/{total_samples} | Current accuracy: {accuracy:.1f}% | "
                  f"Avg time/sample: {avg_time_per_sample:.1f}s | ETA: {remaining_time/60:.1f}m")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    correct = sum(1 for result in evaluation_results if result['correct'])
    final_accuracy = correct / len(evaluation_results) * 100
    
    print(f"\n🏁 Inference completed!")
    print(f"  📊 Final accuracy: {final_accuracy:.2f}% ({correct}/{total_samples})")
    print(f"  ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"  ⚡ Average time per sample: {total_time/total_samples:.1f}s")
    
    # Compute layer-wise angle analysis
    print(f"\n📐 Computing layer-wise angle analysis...")
    layer_analysis = compute_layer_wise_angle_accuracy_analysis(evaluation_results)
    
    # Print interpretable layer-wise results
    if layer_analysis:
        print(f"\n📊 Layer-wise Analysis Results:")
        print(f"{'='*80}")
        print(f"{'Layer':<6} {'Correct Mean':<12} {'Incorrect Mean':<14} {'Difference':<12} {'Effect Size':<12} {'P-value':<10} {'Significant':<12}")
        print(f"{'='*80}")
        
        for layer_id in sorted(layer_analysis.keys()):
            stats = layer_analysis[layer_id]
            correct_mean = stats['correct_predictions']['mean_angle']
            incorrect_mean = stats['incorrect_predictions']['mean_angle']
            difference = stats['comparison']['mean_difference']
            cohens_d = stats['comparison']['cohens_d']
            p_value = stats['comparison']['p_value']
            significant = "Yes" if stats['comparison']['significant'] else "No"
            
            print(f"{layer_id:<6} {correct_mean:<12.2f} {incorrect_mean:<14.2f} {difference:<12.2f} {cohens_d:<12.3f} {p_value:<10.3f} {significant:<12}")
        
        print(f"{'='*80}")
        
        # Summary of significant layers
        significant_layers = [layer_id for layer_id, stats in layer_analysis.items() if stats['comparison']['significant']]
        if significant_layers:
            print(f"\n🎯 Significant layers (p < 0.05): {len(significant_layers)}/{len(layer_analysis)}")
            print(f"   Layers with significant differences: {', '.join(map(str, significant_layers))}")
            
            # Find layers with largest effects
            large_effect_layers = [layer_id for layer_id, stats in layer_analysis.items() 
                                 if abs(stats['comparison']['cohens_d']) >= 0.8]
            medium_effect_layers = [layer_id for layer_id, stats in layer_analysis.items() 
                                  if 0.5 <= abs(stats['comparison']['cohens_d']) < 0.8]
            
            if large_effect_layers:
                print(f"   Layers with large effect sizes (|d| ≥ 0.8): {', '.join(map(str, large_effect_layers))}")
            if medium_effect_layers:
                print(f"   Layers with medium effect sizes (0.5 ≤ |d| < 0.8): {', '.join(map(str, medium_effect_layers))}")
        else:
            print(f"\n⚠️  No layers show statistically significant differences between correct and incorrect predictions")
    
    # Analyze by sampling method if using local dataset
    if use_local_dataset:
        sampling_stats = {}
        for result in evaluation_results:
            sampling = result['sampling_method']
            if sampling not in sampling_stats:
                sampling_stats[sampling] = {'correct': 0, 'total': 0}
            sampling_stats[sampling]['total'] += 1
            if result['correct']:
                sampling_stats[sampling]['correct'] += 1
        
        print(f"\n📊 Results by sampling method:")
        for sampling, stats in sampling_stats.items():
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"  {sampling}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Update output filename to include steering info
    output_path = args.output_path
    # Insert angle correlation info before file extension
    name_parts = output_path.rsplit('.', 1)
    if len(name_parts) == 2:
        output_path = f"{name_parts[0]}_angle_correlation_lambda{vsv_lambda}.{name_parts[1]}"
    else:
        output_path = f"{output_path}_angle_correlation_lambda{vsv_lambda}"
    
    # Writing the data to CSV using csv module
    print(f"💾 Saving results to {output_path}...")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if use_local_dataset:
            writer.writerow(["entry_id", "audio_index", "label", "response", "correct", "mean_angle", "std_angle", "angle_count", "sampling_method", "vsv_lambda"])
        else:
            writer.writerow(["entry_id", "audio_index", "label", "response", "correct", "mean_angle", "std_angle", "angle_count", "vsv_lambda"])
        
        for result in evaluation_results:
            if use_local_dataset:
                writer.writerow([
                    result['entry_id'], result['audio_index'], result['label'], result['response'], 
                    result['correct'], result['mean_angle'], result['std_angle'], result['angle_count'],
                    result['sampling_method'], result['vsv_lambda']
                ])
            else:
                writer.writerow([
                    result['entry_id'], result['audio_index'], result['label'], result['response'],
                    result['correct'], result['mean_angle'], result['std_angle'], result['angle_count'],
                    result['vsv_lambda']
                ])
    
    # Save detailed analysis results
    analysis_results = {
        'experiment_config': {
            'num_samples': total_samples,
            'vsv_lambda': vsv_lambda,
            'dataset_file': args.dataset_file if hasattr(args, 'dataset_file') else args.dataset_name,
            'audio_root_dir': args.audio_root_dir
        },
        'evaluation_results': evaluation_results,
        'layer_wise_analysis': layer_analysis,
        'summary': {
            'accuracy': final_accuracy,
            'total_time_minutes': total_time/60,
            'samples_with_angle_data': len([r for r in evaluation_results if r['mean_angle'] is not None]),
            'layers_analyzed': len(layer_analysis) if layer_analysis else 0,
            'significant_layers': len([l for l, s in layer_analysis.items() if s['comparison']['significant']]) if layer_analysis else 0
        }
    }
    
    analysis_output_path = output_path.replace('.csv', '_analysis.json')
    with open(analysis_output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✅ Inference results are saved to {output_path}")
    print(f"📊 Detailed analysis saved to {analysis_output_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio hallucination evaluation with angle-accuracy correlation analysis")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name.", default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, help="Path to local dataset TSV file (alternative to --dataset_name)", default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, help="Audio root directory", default="./understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, help="Output path of csv file.", default="./angle_correlation_evaluation_result.csv")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output for individual inference steps")
    
    # Vector steering options
    parser.add_argument("--vsv_lambda", type=float, default=0.05, help="Vector steering strength (lambda). Higher values = stronger steering. Default: 0.05")
    
    # Testing options
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    main(args)