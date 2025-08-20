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
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from scipy.stats import pearsonr, spearmanr, ttest_ind
import torch.nn.functional as F

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
    from ..layers.llm_layer import add_vsv_layers, remove_vsv_layers, get_layers
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
        from src.layers.llm_layer import add_vsv_layers, remove_vsv_layers, get_layers
        VSV_AVAILABLE = True
    except ImportError as e2:
        print(f"Warning: Vector steering modules not available: {e2}")
        print("Vector steering will be disabled.")
        VSV_AVAILABLE = False

# Global variables for model and processor
model = None
processor = None
verbose_progress = False
vsv_enabled = False
vsv_lambda = 1.0

# Global storage for cosine similarity measurements
cosine_measurements = []

class CosineHookWithSteering:
    """Hook to capture per-token cosine similarity between hidden states and steering vectors after steering is applied, averaged by layer"""
    
    def __init__(self, layer_idx: int, steering_vector: torch.Tensor, lam: float):
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector.clone()
        self.lam = lam
    
    def __call__(self, module, input, output):
        """Hook function called during forward pass - applies steering then measures cosine similarity"""
        global cosine_measurements
        
        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
            rest_output = output[1:]
        else:
            hidden_states = output
            rest_output = ()
        
        if hidden_states.dim() != 3:
            return output
        
        orig_dtype = hidden_states.dtype
        x = hidden_states.float()
        batch_size, seq_len, hidden_dim = x.shape
        
        # Ensure steering vector is on the same device and dtype
        steering_vec = self.steering_vector.to(x.device, x.dtype)
        
        # Apply steering: h' = h + λ * v (same as llm_layer.py)
        v = steering_vec.view(1, 1, -1)  # [1, 1, hidden_dim] for broadcasting
        x_steered = x + self.lam * v
        
        # NOW measure cosine similarity on the STEERED hidden states (same method as llm_layer.py)
        # Normalize steered hidden states per token: [batch_size, seq_len, hidden_dim]
        x_normalized = F.normalize(x_steered, p=2, dim=-1)
        
        # Normalize steering vector: [1, 1, hidden_dim]
        v_normalized = F.normalize(v, p=2, dim=-1)
        
        # Compute per-token cosine similarity: [batch_size, seq_len]
        per_token_cos_sim = torch.sum(x_normalized * v_normalized, dim=-1)
        
        # Average cosine similarity across tokens: [batch_size]
        #cos_sim = per_token_cos_sim.mean(dim=1)  # [batch_size]
        cos_sim =per_token_cos_sim[:,0]
        # Store measurement (using first batch item since batch_size=1 during generation)
        layer_mean_cosine = cos_sim[0].item()
        measurement = {
            'layer': self.layer_idx,
            'cosine_similarity': layer_mean_cosine
        }
        cosine_measurements.append(measurement)
        
        # Return the steered hidden states (applying the actual steering effect)
        x_steered = x_steered.to(orig_dtype)
        if rest_output:
            return (x_steered,) + rest_output
        else:
            return x_steered

def clear_cosine_storage():
    """Clear stored cosine measurements"""
    global cosine_measurements
    cosine_measurements = []

def get_cosine_statistics():
    """Get overall cosine similarity statistics"""
    global cosine_measurements
    
    if not cosine_measurements:
        return {}
    
    similarities = [m['cosine_similarity'] for m in cosine_measurements]
    
    return {
        'count': len(similarities),
        'mean': np.mean(similarities),
        'std': np.std(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'median': np.median(similarities)
    }

def get_layer_by_layer_cosine_statistics():
    """Get per-layer cosine similarity statistics"""
    global cosine_measurements
    
    if not cosine_measurements:
        return {}
    
    layer_stats = {}
    
    # Group by layer
    for measurement in cosine_measurements:
        layer_id = measurement['layer']
        if layer_id not in layer_stats:
            layer_stats[layer_id] = []
        layer_stats[layer_id].append(measurement['cosine_similarity'])
    
    # Compute statistics per layer
    result = {}
    for layer_id, similarities in layer_stats.items():
        result[layer_id] = {
            'count': len(similarities),
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'min': np.min(similarities),
            'max': np.max(similarities),
            'median': np.median(similarities),
            'cosine_similarities': similarities  # Store for layer-wise analysis
        }
    
    return result

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
            audio=[audio],
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
    vsv_prompt = f"Focus on the given audio and answer the following question. {prompt} Answer just yes or no."
    
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

def inference_with_cosine_measurement(audio_path, prompt_text, vsv_lambda=0.05):
    """
    Perform inference with per-token cosine similarity measurement between steering vectors and hidden states AFTER steering is applied.
    This measures the post-steering cosine similarity using the same method as llm_layer.py.
    
    Returns:
        result: 'Yes' or 'No' prediction
        layer_stats: Per-layer cosine similarity statistics (post-steering)
        overall_stats: Overall cosine similarity statistics (post-steering)
    """
    global model, processor, verbose_progress
    
    if model is None or processor is None:
        initialize_model()
    
    # Clear previous cosine measurements
    clear_cosine_storage()
    
    if verbose_progress:
        print(f"  🎵 Processing: {os.path.basename(audio_path)}")
    
    hooks = []
    try:
        # Compute VSV for this specific audio using the actual prompt
        vsv = compute_vsv_for_audio(audio_path, prompt_text)
        
        if verbose_progress:
            print("    🎯 Setting up post-steering cosine similarity measurement hooks...")
        
        # Get model layers
        layers = get_layers(model, which_stack="decoder")
        
        # Register hooks for cosine similarity measurement with steering
        for layer_idx, layer in enumerate(layers):
            if layer_idx < len(vsv):  # Only if we have steering vector for this layer
                hook = CosineHookWithSteering(layer_idx, vsv[layer_idx], vsv_lambda)
                handle = layer.register_forward_hook(hook)
                hooks.append(handle)
        
        if verbose_progress:
            print(f"    ✅ Post-steering cosine similarity hooks registered for {len(hooks)} layers")
        
        # Build messages in the expected format
        # Append instruction to answer only yes or no
        modified_prompt = f"Focus on the given audio and answer the following question. {prompt_text} Answer just yes or no."
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
        inputs = processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        if verbose_progress:
            print("    🧠 Generating response (measuring post-steering cosine similarities)...")
        # Generate response - cosine similarities are measured during forward pass
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
        
        # Get the collected cosine similarity statistics
        layer_stats = get_layer_by_layer_cosine_statistics()
        overall_stats = get_cosine_statistics()
        
        if verbose_progress and overall_stats:
            print(f"    📐 Collected {overall_stats['count']} cosine similarity measurements")
            print(f"    📊 Mean cosine similarity: {overall_stats['mean']:.4f}")
        
        return result, layer_stats, overall_stats
            
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {e}"
        if verbose_progress:
            print(f"    ❌ {error_msg}")
        else:
            print(error_msg)
        return "No", {}, {}
    
    finally:
        # Always remove hooks after inference to prevent interference
        for handle in hooks:
            handle.remove()
        if verbose_progress and hooks:
            print("    🔄 Post-steering cosine similarity hooks removed")

def compute_inter_layer_cosine_correlations(evaluation_results):
    """
    Compute Pearson correlations between cosine similarities across layers,
    comparing correct vs incorrect predictions.
    
    Args:
        evaluation_results: List of dictionaries containing:
            - correct: bool, whether prediction was correct
            - layer_stats: dict, per-layer cosine similarity statistics
    
    Returns:
        correlation_analysis: Dictionary with correlation matrices and comparisons
    """
    if len(evaluation_results) < 2:
        print("❌ Need at least 2 samples for inter-layer correlation analysis")
        return {}
    
    # Separate results by correctness
    correct_results = [r for r in evaluation_results if r['correct'] and r.get('layer_stats')]
    incorrect_results = [r for r in evaluation_results if not r['correct'] and r.get('layer_stats')]
    
    if len(correct_results) < 2 or len(incorrect_results) < 2:
        print("❌ Need at least 2 samples in each category for correlation analysis")
        return {}
    
    def build_layer_cosine_matrix(results):
        """Build matrix of layer cosine similarities for correlation analysis"""
        layer_cosine_matrix = {}  # layer_id -> list of mean cosine values
        
        for result in results:
            if result.get('layer_stats'):
                for layer_id, stats in result['layer_stats'].items():
                    if layer_id not in layer_cosine_matrix:
                        layer_cosine_matrix[layer_id] = []
                    layer_cosine_matrix[layer_id].append(stats['mean'])
        
        # Only keep layers that have data for all samples
        min_samples = min(len(values) for values in layer_cosine_matrix.values()) if layer_cosine_matrix else 0
        filtered_matrix = {layer_id: values[:min_samples] 
                          for layer_id, values in layer_cosine_matrix.items() 
                          if len(values) >= min_samples}
        
        return filtered_matrix
    
    def compute_correlation_matrix(layer_matrix):
        """Compute pairwise Pearson correlations between layers"""
        correlations = {}
        layer_ids = sorted(layer_matrix.keys())
        
        for i, layer1 in enumerate(layer_ids):
            for j, layer2 in enumerate(layer_ids):
                if i < j:  # Only compute upper triangle to avoid duplicates
                    try:
                        corr, p_val = pearsonr(layer_matrix[layer1], layer_matrix[layer2])
                        correlations[f"{layer1}-{layer2}"] = {
                            'correlation': corr,
                            'p_value': p_val,
                            'significant': p_val < 0.05,
                            'layer1': layer1,
                            'layer2': layer2,
                            'abs_correlation': abs(corr)
                        }
                    except Exception as e:
                        print(f"⚠️  Could not compute correlation between layers {layer1}-{layer2}: {e}")
                        
        return correlations
    
    # Build matrices for correct and incorrect predictions
    correct_matrix = build_layer_cosine_matrix(correct_results)
    incorrect_matrix = build_layer_cosine_matrix(incorrect_results)
    
    if not correct_matrix or not incorrect_matrix:
        print("❌ Insufficient layer data for correlation analysis")
        return {}
    
    # Compute correlation matrices
    correct_correlations = compute_correlation_matrix(correct_matrix)
    incorrect_correlations = compute_correlation_matrix(incorrect_matrix)
    
    # Find common layer pairs for comparison
    common_pairs = set(correct_correlations.keys()) & set(incorrect_correlations.keys())
    
    if not common_pairs:
        print("❌ No common layer pairs between correct and incorrect predictions")
        return {}
    
    # Compare correlations between correct and incorrect predictions
    correlation_comparison = {}
    for pair in common_pairs:
        correct_corr = correct_correlations[pair]['correlation']
        incorrect_corr = incorrect_correlations[pair]['correlation']
        
        correlation_comparison[pair] = {
            'correct_correlation': correct_corr,
            'incorrect_correlation': incorrect_corr,
            'correlation_difference': correct_corr - incorrect_corr,
            'correct_p_value': correct_correlations[pair]['p_value'],
            'incorrect_p_value': incorrect_correlations[pair]['p_value'],
            'correct_significant': correct_correlations[pair]['significant'],
            'incorrect_significant': incorrect_correlations[pair]['significant'],
            'layer1': correct_correlations[pair]['layer1'],
            'layer2': correct_correlations[pair]['layer2']
        }
    
    # Summary statistics
    correlation_diffs = [comp['correlation_difference'] for comp in correlation_comparison.values()]
    
    summary = {
        'num_layer_pairs': len(correlation_comparison),
        'num_correct_samples': len(correct_results),
        'num_incorrect_samples': len(incorrect_results),
        'mean_correlation_difference': np.mean(correlation_diffs) if correlation_diffs else 0,
        'std_correlation_difference': np.std(correlation_diffs) if correlation_diffs else 0,
        'max_correlation_difference': np.max(correlation_diffs) if correlation_diffs else 0,
        'min_correlation_difference': np.min(correlation_diffs) if correlation_diffs else 0
    }
    
    return {
        'correlation_comparison': correlation_comparison,
        'correct_correlations': correct_correlations,
        'incorrect_correlations': incorrect_correlations,
        'summary': summary
    }

def compute_layer_wise_cosine_accuracy_analysis(evaluation_results):
    """
    Compute layer-by-layer analysis of cosine similarities vs prediction accuracy.
    
    Args:
        evaluation_results: List of dictionaries containing:
            - correct: bool, whether prediction was correct
            - layer_stats: dict, per-layer cosine similarity statistics
    
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
        # Collect cosine similarities for this layer from correct predictions
        correct_similarities = []
        for result in correct_results:
            if layer_id in result['layer_stats']:
                correct_similarities.extend(result['layer_stats'][layer_id].get('cosine_similarities', []))
        
        # Collect cosine similarities for this layer from incorrect predictions
        incorrect_similarities = []
        for result in incorrect_results:
            if layer_id in result['layer_stats']:
                incorrect_similarities.extend(result['layer_stats'][layer_id].get('cosine_similarities', []))
        
        if not correct_similarities or not incorrect_similarities:
            continue
            
        # Compute statistics for this layer
        correct_mean = np.mean(correct_similarities)
        incorrect_mean = np.mean(incorrect_similarities)
        correct_std = np.std(correct_similarities)
        incorrect_std = np.std(incorrect_similarities)
        correct_median = np.median(correct_similarities)
        incorrect_median = np.median(incorrect_similarities)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(correct_similarities) - 1) * correct_std**2 + 
                             (len(incorrect_similarities) - 1) * incorrect_std**2) / 
                            (len(correct_similarities) + len(incorrect_similarities) - 2))
        cohens_d = (correct_mean - incorrect_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test (Welch's t-test)
        t_stat, p_value = ttest_ind(correct_similarities, incorrect_similarities, equal_var=False)
        
        layer_analysis[layer_id] = {
            'correct_predictions': {
                'count': len(correct_similarities),
                'mean_cosine': correct_mean,
                'std_cosine': correct_std,
                'median_cosine': correct_median,
                'min_cosine': np.min(correct_similarities),
                'max_cosine': np.max(correct_similarities)
            },
            'incorrect_predictions': {
                'count': len(incorrect_similarities),
                'mean_cosine': incorrect_mean,
                'std_cosine': incorrect_std,
                'median_cosine': incorrect_median,
                'min_cosine': np.min(incorrect_similarities),
                'max_cosine': np.max(incorrect_similarities)
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
    vsv_enabled = True  # Always enable VSV for cosine similarity measurement
    vsv_lambda = args.vsv_lambda

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
        
        # Limit samples for cosine similarity analysis
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

    print(f"🎯 Vector steering ENABLED with λ={vsv_lambda} for cosine similarity measurement")
    print(f"🎯 Starting inference with cosine similarity-accuracy correlation analysis on {total_samples} samples...")
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

        # Inference model and get response with cosine similarity measurement.
        response, layer_stats, overall_stats = inference_with_cosine_measurement(
            audio_path=audio_path, 
            prompt_text=prompt_text,
            vsv_lambda=vsv_lambda
        )

        # Determine if prediction was correct
        correct = (response == label)

        # Extract cosine similarity statistics for correlation analysis
        mean_cosine = overall_stats.get('mean', None) if overall_stats else None
        std_cosine = overall_stats.get('std', None) if overall_stats else None
        cosine_count = overall_stats.get('count', 0) if overall_stats else 0

        # Record evaluation result with cosine similarity data (only essential data)
        result_data = {
            'entry_id': entry_id,
            'audio_index': audio_index,
            'label': label,
            'response': response,
            'correct': correct,
            'mean_cosine': mean_cosine,
            'std_cosine': std_cosine,
            'cosine_count': cosine_count,
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
    
    # Compute layer-wise cosine similarity analysis
    print(f"\n📐 Computing layer-wise cosine similarity analysis...")
    layer_analysis = compute_layer_wise_cosine_accuracy_analysis(evaluation_results)
    
    # Compute inter-layer correlation analysis
    print(f"\n🔗 Computing inter-layer cosine correlation analysis...")
    correlation_analysis = compute_inter_layer_cosine_correlations(evaluation_results)
    
    # Print interpretable layer-wise results
    if layer_analysis:
        print(f"\n📊 Layer-wise Analysis Results:")
        print(f"{'='*90}")
        print(f"{'Layer':<6} {'Correct Mean':<12} {'Incorrect Mean':<14} {'Difference':<12} {'Effect Size':<12} {'P-value':<10} {'Significant':<12}")
        print(f"{'='*90}")
        
        for layer_id in sorted(layer_analysis.keys()):
            stats = layer_analysis[layer_id]
            correct_mean = stats['correct_predictions']['mean_cosine']
            incorrect_mean = stats['incorrect_predictions']['mean_cosine']
            difference = stats['comparison']['mean_difference']
            cohens_d = stats['comparison']['cohens_d']
            p_value = stats['comparison']['p_value']
            significant = "Yes" if stats['comparison']['significant'] else "No"
            
            print(f"{layer_id:<6} {correct_mean:<12.4f} {incorrect_mean:<14.4f} {difference:<12.4f} {cohens_d:<12.3f} {p_value:<10.3f} {significant:<12}")
        
        print(f"{'='*90}")
        
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
    
    # Print inter-layer correlation analysis results
    if correlation_analysis and correlation_analysis.get('correlation_comparison'):
        print(f"\n🔗 Inter-layer Correlation Analysis Results:")
        print(f"{'='*110}")
        print(f"{'Layer Pair':<12} {'Correct Corr':<12} {'Incorrect Corr':<14} {'Difference':<12} {'Correct Sig':<12} {'Incorrect Sig':<14} {'Interpretation':<20}")
        print(f"{'='*110}")
        
        # Sort by absolute correlation difference for most interesting pairs first
        sorted_pairs = sorted(correlation_analysis['correlation_comparison'].items(), 
                             key=lambda x: abs(x[1]['correlation_difference']), reverse=True)
        
        for pair_name, stats in sorted_pairs:
            correct_corr = stats['correct_correlation']
            incorrect_corr = stats['incorrect_correlation']
            difference = stats['correlation_difference']
            correct_sig = "Yes" if stats['correct_significant'] else "No"
            incorrect_sig = "Yes" if stats['incorrect_significant'] else "No"
            
            # Interpretation
            if difference > 0.2:
                interpretation = "Higher when correct"
            elif difference < -0.2:
                interpretation = "Higher when incorrect"
            elif abs(difference) < 0.1:
                interpretation = "Similar"
            else:
                interpretation = "Moderate difference"
            
            print(f"{pair_name:<12} {correct_corr:<12.3f} {incorrect_corr:<14.3f} {difference:<12.3f} {correct_sig:<12} {incorrect_sig:<14} {interpretation:<20}")
        
        print(f"{'='*110}")
        
        # Summary of correlation analysis
        summary = correlation_analysis['summary']
        print(f"\n🎯 Inter-layer Correlation Summary:")
        print(f"   Layer pairs analyzed: {summary['num_layer_pairs']}")
        print(f"   Correct prediction samples: {summary['num_correct_samples']}")
        print(f"   Incorrect prediction samples: {summary['num_incorrect_samples']}")
        print(f"   Mean correlation difference: {summary['mean_correlation_difference']:.3f}")
        print(f"   Std correlation difference: {summary['std_correlation_difference']:.3f}")
        
        # Find most interesting pairs
        large_diff_pairs = [pair for pair, stats in correlation_analysis['correlation_comparison'].items() 
                           if abs(stats['correlation_difference']) > 0.3]
        if large_diff_pairs:
            print(f"   Pairs with large differences (|diff| > 0.3): {', '.join(large_diff_pairs)}")
        
        # Steering coherence insight
        mean_diff = summary['mean_correlation_difference']
        if mean_diff > 0.1:
            print(f"\n💡 Insight: Inter-layer correlations are generally HIGHER for correct predictions")
            print(f"   → Suggests steering vectors work more coherently across layers when predictions are accurate")
        elif mean_diff < -0.1:
            print(f"\n💡 Insight: Inter-layer correlations are generally HIGHER for incorrect predictions")
            print(f"   → May indicate over-coordination or getting stuck in wrong patterns")
        else:
            print(f"\n💡 Insight: Inter-layer correlations show similar patterns for correct and incorrect predictions")
    else:
        print(f"\n⚠️  Could not compute inter-layer correlation analysis (insufficient data)")
    
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
    # Insert cosine correlation info before file extension
    name_parts = output_path.rsplit('.', 1)
    if len(name_parts) == 2:
        output_path = f"{name_parts[0]}_cosine_correlation_lambda{vsv_lambda}.{name_parts[1]}"
    else:
        output_path = f"{output_path}_cosine_correlation_lambda{vsv_lambda}"
    
    # Writing the data to CSV using csv module (only essential data)
    print(f"💾 Saving results to {output_path}...")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if use_local_dataset:
            writer.writerow(["entry_id", "audio_index", "label", "response", "correct", "mean_cosine", "std_cosine", "cosine_count", "sampling_method", "vsv_lambda"])
        else:
            writer.writerow(["entry_id", "audio_index", "label", "response", "correct", "mean_cosine", "std_cosine", "cosine_count", "vsv_lambda"])
        
        for result in evaluation_results:
            if use_local_dataset:
                writer.writerow([
                    result['entry_id'], result['audio_index'], result['label'], result['response'], 
                    result['correct'], result['mean_cosine'], result['std_cosine'], result['cosine_count'],
                    result['sampling_method'], result['vsv_lambda']
                ])
            else:
                writer.writerow([
                    result['entry_id'], result['audio_index'], result['label'], result['response'],
                    result['correct'], result['mean_cosine'], result['std_cosine'], result['cosine_count'],
                    result['vsv_lambda']
                ])
    
    # Save detailed analysis results (only essential analysis data)
    analysis_results = {
        'experiment_config': {
            'num_samples': total_samples,
            'vsv_lambda': vsv_lambda,
            'dataset_file': args.dataset_file if hasattr(args, 'dataset_file') else args.dataset_name,
            'audio_root_dir': args.audio_root_dir
        },
        'layer_wise_analysis': layer_analysis,
        'inter_layer_correlation_analysis': correlation_analysis,
        'summary': {
            'accuracy': final_accuracy,
            'total_time_minutes': total_time/60,
            'samples_with_cosine_data': len([r for r in evaluation_results if r['mean_cosine'] is not None]),
            'layers_analyzed': len(layer_analysis) if layer_analysis else 0,
            'significant_layers': len([l for l, s in layer_analysis.items() if s['comparison']['significant']]) if layer_analysis else 0,
            'layer_pairs_analyzed': len(correlation_analysis.get('correlation_comparison', {})) if correlation_analysis else 0,
            'mean_correlation_difference': correlation_analysis.get('summary', {}).get('mean_correlation_difference', 0) if correlation_analysis else 0
        }
    }
    
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    analysis_output_path = output_path.replace('.csv', '_analysis.json')
    with open(analysis_output_path, 'w') as f:
        json.dump(convert_numpy_types(analysis_results), f, indent=2)
    
    print(f"✅ Inference results are saved to {output_path}")
    print(f"📊 Detailed analysis saved to {analysis_output_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio hallucination evaluation with cosine similarity-accuracy correlation analysis")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name.", default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, help="Path to local dataset TSV file (alternative to --dataset_name)", default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, help="Audio root directory", default="./understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, help="Output path of csv file.", default="./no_steer_first_eng_cosine_correlation_evaluation_result.csv")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output for individual inference steps")
    
    # Vector steering options
    parser.add_argument("--vsv_lambda", type=float, default=0.0, help="Vector steering strength (lambda). Higher values = stronger steering. Default: 0.05")
    
    # Testing options
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    main(args)