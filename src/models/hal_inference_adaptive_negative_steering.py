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
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

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

# Layer-specific steering configuration based on cosine correlation analysis
# Layers with negative differences need negative lambda to flip steering direction
LAYER_STEERING_CONFIG = {
    # Format: layer_id: (lambda_multiplier, description)
    0: (0.0, "neutral"),       # -0.0001, not significant
    1: (0.0, "positive"),      # +0.0010, significant  
    2: (0.0, "positive"),      # +0.0002, not significant
    3: (0.0, "positive"),      # +0.0015, significant
    4: (0.0, "positive"),      # +0.0025, significant
    5: (0.0, "positive"),      # +0.0020, significant
    6: (0.0, "positive"),      # +0.0012, not significant
    7: (0.0, "positive"),      # +0.0009, not significant
    8: (0.0, "positive"),      # +0.0004, not significant
    9: (0.0, "positive"),      # +0.0013, significant
    10: (0.0, "positive"),     # +0.0016, significant
    11: (0.0, "positive"),     # +0.0014, significant
    12: (0.0, "positive"),     # +0.0003, not significant
    13: (0.0, "negative"),    # -0.0009, significant - FLIP
    14: (0.0, "negative"),    # -0.0010, significant - FLIP
    15: (1.0, "neutral"),      # -0.0000, not significant
    16: (1.0, "neutral"),      # -0.0003, not significant
    17: (1.0, "negative"),    # -0.0018, significant - FLIP
    18: (1.0, "neutral"),      # +0.0000, not significant
    19: (1.0, "neutral"),      # -0.0009, not significant
    20: (1.0, "neutral"),      # -0.0010, not significant
    21: (1.0, "positive"),     # +0.0008, not significant
    22: (1.0, "positive"),     # +0.0028, significant
    23: (1.0, "positive"),     # +0.0036, significant
    24: (1.0, "positive"),     # +0.0026, significant
    25: (1.0, "positive"),     # +0.0022, significant
    26: (1.0, "positive"),     # +0.0012, not significant
    27: (1.0, "neutral"),      # -0.0000, not significant
    28: (1.0, "positive"),     # +0.0019, significant
    29: (1.0, "positive"),     # +0.0009, not significant
    30: (1.0, "negative"),    # -0.0052, significant - FLIP
    31: (1.0, "negative"),    # -0.0225, highly significant - FLIP
}

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

def apply_adaptive_vsv_layers(model, vsv, base_lambda, which_stack="decoder"):
    """
    Apply VSV layers with adaptive lambda values based on layer-wise cosine correlation analysis.
    
    Args:
        model: The model to apply steering to
        vsv: Steering vectors [num_layers, hidden_dim]  
        base_lambda: Base lambda value (will be multiplied by layer-specific multipliers)
        which_stack: Which stack to apply to
    """
    global verbose_progress
    
    # get_layers is already imported at the top
    
    layers = get_layers(model, which_stack=which_stack)
    
    # Ensure we don't exceed available layers or VSV dimensions
    num_layers = min(len(layers), len(vsv), len(LAYER_STEERING_CONFIG))
    
    handles = []
    positive_layers = []
    negative_layers = []
    
    for i in range(num_layers):
        layer = layers[i]
        steering_vector = vsv[i]
        
        # Get layer-specific configuration
        if i in LAYER_STEERING_CONFIG:
            multiplier, description = LAYER_STEERING_CONFIG[i]
        else:
            multiplier, description = (1.0, "default")  # Default to positive
        
        # Calculate effective lambda for this layer
        effective_lambda = base_lambda * multiplier
        
        # Track which layers get positive vs negative steering
        if multiplier > 0:
            positive_layers.append(i)
        else:
            negative_layers.append(i)
        
        # Create and register the hook
        def make_vsv_hook(v_l, lam):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                orig_dtype = h.dtype
                x = h.float()
                
                # Apply steering: h' = h + λ * v
                v = v_l.view(1, 1, -1).to(x.device, x.dtype)
                x = x + lam * v
                
                x = x.to(orig_dtype)
                return (x,) + out[1:] if isinstance(out, tuple) else x
            return hook
        
        handle = layer.register_forward_hook(make_vsv_hook(steering_vector, effective_lambda))
        handles.append(handle)
    
    # Store handles for later removal
    if not hasattr(model, "_adaptive_vsv_handles"):
        model._adaptive_vsv_handles = []
    model._adaptive_vsv_handles.extend(handles)
    
    if verbose_progress:
        print(f"    ✅ Adaptive steering applied:")
        print(f"      Positive steering (λ={base_lambda}): layers {positive_layers}")
        print(f"      Negative steering (λ={-base_lambda}): layers {negative_layers}")
        print(f"      Total layers steered: {num_layers}")
    
    return len(positive_layers), len(negative_layers)

def remove_adaptive_vsv_layers(model):
    """Remove adaptive VSV layers"""
    if hasattr(model, "_adaptive_vsv_handles"):
        for handle in model._adaptive_vsv_handles:
            try:
                handle.remove()
            except:
                pass
        model._adaptive_vsv_handles = []

def inference(audio_path, prompt_text):
    """
    Perform inference on audio with the given prompt text using adaptive negative steering.
    Returns 'Yes' or 'No' for discriminative tasks.
    Applies positive or negative steering per layer based on cosine correlation analysis.
    """
    global model, processor, verbose_progress, vsv_enabled, vsv_lambda
    
    if model is None or processor is None:
        initialize_model()
    
    if verbose_progress:
        print(f"  🎵 Processing: {os.path.basename(audio_path)}")
    
    vsv_applied = False
    try:
        # Apply adaptive vector steering if enabled
        if vsv_enabled:
            if verbose_progress:
                print("    🎯 Applying adaptive vector steering...")
            
            # Compute VSV for this specific audio using the actual prompt
            vsv = compute_vsv_for_audio(audio_path, prompt_text)
            
            # Apply adaptive VSV with layer-specific lambda values
            num_positive, num_negative = apply_adaptive_vsv_layers(
                model, vsv=vsv, base_lambda=vsv_lambda, which_stack="decoder"
            )
            vsv_applied = True
            
            if verbose_progress:
                print(f"    ✅ Adaptive steering applied: {num_positive} positive + {num_negative} negative layers")
        
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
            print("    🧠 Generating response...")
        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)

        if verbose_progress:
            print("    📤 Decoding output...")
        # Decode output
        output = output[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Print the raw model generated response
        print(f"Model Response: {response}")
        
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
        
        return result
            
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {e}"
        if verbose_progress:
            print(f"    ❌ {error_msg}")
        else:
            print(error_msg)
        return "No"
    
    finally:
        # Always remove VSV hooks after inference to prevent interference
        if vsv_applied:
            remove_adaptive_vsv_layers(model)
            if verbose_progress:
                print("    🔄 Adaptive steering hooks removed")

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
    vsv_enabled = args.enable_vsv
    vsv_lambda = args.vsv_lambda

    # Check if using local dataset file or HuggingFace dataset
    if hasattr(args, 'dataset_file') and args.dataset_file:
        print("📊 Loading local dataset...")
        dataset_samples = load_local_dataset(args.dataset_file)
        
        # Randomly shuffle the dataset
        print("🔀 Randomly shuffling dataset...")
        #random.shuffle(dataset_samples)
        
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
        
        total_samples = len(dataset_samples)
        use_local_dataset = False
        print(f"📝 Dataset loaded: {total_samples} samples to process")

    # Evaluation results.
    evaluation_results = []
    
    # Initialize model before processing (if not already initialized)
    if model is None:
        initialize_model()

    # Print adaptive steering configuration
    if vsv_enabled:
        print(f"🎯 Adaptive vector steering ENABLED with base λ={vsv_lambda}")
        
        # Count positive vs negative layers
        positive_count = sum(1 for mult, desc in LAYER_STEERING_CONFIG.values() if mult > 0)
        negative_count = sum(1 for mult, desc in LAYER_STEERING_CONFIG.values() if mult < 0)
        
        print(f"📊 Steering configuration:")
        print(f"   Positive steering layers: {positive_count}")
        print(f"   Negative steering layers: {negative_count}")
        print(f"   Negative layers: {[i for i, (mult, desc) in LAYER_STEERING_CONFIG.items() if mult < 0]}")
    else:
        print("🎯 Vector steering DISABLED")

    print(f"🎯 Starting inference on {total_samples} samples...")
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

        # Inference model and get response.
        response = inference(audio_path=audio_path, prompt_text=prompt_text)

        # Record evaluation result.
        if use_local_dataset:
            evaluation_result = [entry_id, audio_index, label, response, sampling_method, vsv_enabled, vsv_lambda if vsv_enabled else 0.0]
        else:
            evaluation_result = [entry_id, audio_index, label, response, vsv_enabled, vsv_lambda if vsv_enabled else 0.0]
        evaluation_results.append(evaluation_result)
        
        # Show progress every 50 samples or at the end
        if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
            correct = sum(1 for result in evaluation_results if result[2] == result[3])
            accuracy = correct / len(evaluation_results) * 100
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / (idx + 1)
            estimated_total_time = avg_time_per_sample * total_samples
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"  📈 Progress: {idx + 1}/{total_samples} | Current accuracy: {accuracy:.1f}% | "
                  f"Avg time/sample: {avg_time_per_sample:.1f}s | ETA: {remaining_time/60:.1f}m")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    correct = sum(1 for result in evaluation_results if result[2] == result[3])
    final_accuracy = correct / len(evaluation_results) * 100
    
    print(f"\n🏁 Inference completed!")
    print(f"  📊 Final accuracy: {final_accuracy:.2f}% ({correct}/{total_samples})")
    print(f"  ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"  ⚡ Average time per sample: {total_time/total_samples:.1f}s")
    
    # Analyze by sampling method if using local dataset
    if use_local_dataset:
        sampling_stats = {}
        for result in evaluation_results:
            sampling = result[4]  # sampling_method is at index 4 for local datasets
            if sampling not in sampling_stats:
                sampling_stats[sampling] = {'correct': 0, 'total': 0}
            sampling_stats[sampling]['total'] += 1
            if result[2] == result[3]:  # label == response
                sampling_stats[sampling]['correct'] += 1
        
        print(f"\n📊 Results by sampling method:")
        for sampling, stats in sampling_stats.items():
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"  {sampling}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Update output filename to include adaptive steering info
    output_path = args.output_path
    if vsv_enabled:
        # Insert adaptive steering info before file extension
        name_parts = output_path.rsplit('.', 1)
        if len(name_parts) == 2:
            output_path = f"{name_parts[0]}_adaptive_negative_lambda{vsv_lambda}.{name_parts[1]}"
        else:
            output_path = f"{output_path}_adaptive_negative_lambda{vsv_lambda}"
    
    # Writing the data to CSV using csv module
    print(f"💾 Saving results to {output_path}...")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if use_local_dataset:
            writer.writerow(["entry_id", "audio_index", "label", "response", "sampling_method", "vsv_enabled", "vsv_lambda"])
        else:
            writer.writerow(["entry_id", "audio_index", "label", "response", "vsv_enabled", "vsv_lambda"])
        writer.writerows(evaluation_results)
    
    print(f"✅ Inference results are saved to {output_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio hallucination evaluation with adaptive negative vector steering")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name.", default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, help="Path to local dataset TSV file (alternative to --dataset_name)", default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, help="Audio root directory", default="./understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, help="Output path of csv file.", default="./adaptive_negative_evaluation_result.csv")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output for individual inference steps")
    
    # Vector steering options
    parser.add_argument("--enable_vsv", action="store_true", help="Enable adaptive vector steering for audio hallucination mitigation")
    parser.add_argument("--vsv_lambda", type=float, default=0.05, help="Base vector steering strength (lambda). Adaptive multipliers will be applied per layer. Default: 0.05")
    
    # Testing options
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    main(args)