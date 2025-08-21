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
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

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

# Global variables for model and processor
model = None
processor = None
verbose_progress = False
vsv_enabled = False
vsv_lambda = 1.0
vsv_beta = 0.0
vsv_prepared = False
vsv_tensor = None

def initialize_model():
    """Initialize the model and processor globally"""
    global model, processor
    
    MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
    
    print("🚀 Initializing model...")
    
    print("  📦 Loading configuration...")
    config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
    
    print("  🤖 Loading model (this may take a few minutes)...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    
    print("  🔧 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
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
    # Answer just yes or no.
    # Use the data_prompt (prompt parameter) for VSV computation
    vsv_prompt = f"Focus on the given audio and answer the following question. {prompt} Answer just yes or no."
    #print(vsv_prompt)
    #vsv_prompt = 'Describe the audio content in detail.'
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

def add_vsv_layers_with_beta(model, vsv, base_lambda, beta, which_stack="decoder"):
    """
    Add VSV layers with layer-specific steering strength based on beta parameter.
    For Qwen2Audio layers 0-31:
    - Layers 22-29: multiplier = (1 + 3*beta)
    - Other layers: multiplier = (1 - beta)
    Final steering strength = base_lambda * multiplier
    """
    from transformers import PreTrainedModel
    try:
        from layers.llm_layer import get_layers, _slice_layers_and_vsv, _make_vsv_hook
    except ImportError:
        try:
            from src.layers.llm_layer import get_layers, _slice_layers_and_vsv, _make_vsv_hook
        except ImportError:
            # Fallback: use the same approach as the top-level imports
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)
            project_dir = os.path.dirname(src_dir)
            sys.path.insert(0, src_dir)
            sys.path.insert(0, project_dir)
            from src.layers.llm_layer import get_layers, _slice_layers_and_vsv, _make_vsv_hook
    
    layers = get_layers(model, which_stack=which_stack)
    layers, vsv = _slice_layers_and_vsv(layers, vsv, tar_layers=None)
    
    handles = []
    for i, blk in enumerate(layers):
        # Calculate layer-specific multiplier
        if 22 <= i <= 29:  # Layers 22-29
            multiplier = 1 + 3 * beta
        else:  # Other layers (0-21, 30-31)
            multiplier = 1 - beta
        
        layer_lambda = base_lambda * multiplier
        
        # Create hook with layer-specific lambda
        h = blk.register_forward_hook(_make_vsv_hook(vsv[i], layer_lambda, layer_idx=i))
        handles.append(h)
    
    # Store handles for later removal
    if not hasattr(model, "_vsv_handles"):
        model._vsv_handles = []
    model._vsv_handles.extend(handles)

def inference(audio_path, prompt_text):
    """
    Perform inference on audio with the given prompt text.
    Returns 'Yes' or 'No' for discriminative tasks.
    Supports vector steering if enabled.
    """
    global model, processor, verbose_progress, vsv_enabled, vsv_lambda, vsv_beta
    
    if model is None or processor is None:
        initialize_model()
    
    if verbose_progress:
        print(f"  🎵 Processing: {os.path.basename(audio_path)}")
    
    vsv_applied = False
    try:
        # Apply vector steering if enabled
        if vsv_enabled:
            if verbose_progress:
                print("    🎯 Applying vector steering...")
            
            # Compute VSV for this specific audio using the actual prompt
            vsv = compute_vsv_for_audio(audio_path, prompt_text)
            
            # Inject VSV with layer-specific steering strength
            add_vsv_layers_with_beta(model, vsv=vsv, base_lambda=vsv_lambda, beta=vsv_beta, which_stack="decoder")
            vsv_applied = True
            
            if verbose_progress:
                print(f"    ✅ Vector steering applied with λ={vsv_lambda}, β={vsv_beta}")
                print(f"        Layers 22-29: λ×(1+3β) = {vsv_lambda * (1 + 3 * vsv_beta):.4f}")
                print(f"        Other layers: λ×(1-β) = {vsv_lambda * (1 - vsv_beta):.4f}")
        
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
            output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1, top_p=0.9)

        if verbose_progress:
            print("    📤 Decoding output...")
        # Decode output
        output = output[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Print the raw model generated response
        print(f"Model Response: {response}")
        
        # Check if response is verbose (longer than just "Yes." or "No.")
        response_stripped = response.strip()
        is_verbose = len(response_stripped) > 4
        
        # Clean and normalize response to Yes/No
        response_clean = response.strip().lower()
        
        # Extract Yes/No from response
        if 'yes' in response_clean:
            result = "Yes"
        elif 'no' in response_clean:
            result = "No"
        else:
            # Default to "No" if unclear (following paper's observation that models tend to give affirmative answers)
            result = "No"
        
        if verbose_progress:
            print(f"    ✅ Response: {result} (Verbose: {is_verbose})")
        
        return result, is_verbose
            
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {e}"
        if verbose_progress:
            print(f"    ❌ {error_msg}")
        else:
            print(error_msg)
        return "No", False
    
    finally:
        # Always remove VSV hooks after inference to prevent interference
        if vsv_applied:
            remove_vsv_layers(model, which_stack="decoder")
            if verbose_progress:
                print("    🔄 Vector steering hooks removed")


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

def run_single_experiment(dataset_samples, use_local_dataset, args, lambda_val, beta_val):
    """Run a single experiment with given lambda and beta values"""
    global verbose_progress, vsv_enabled, vsv_lambda, vsv_beta
    
    # Set parameters for this experiment
    vsv_lambda = lambda_val
    vsv_beta = beta_val
    vsv_enabled = True  # Always enable VSV for grid search
    
    total_samples = len(dataset_samples)
    evaluation_results = []
    verbose_count = 0
    
    print(f"\n🔬 Running experiment: λ={lambda_val}, β={beta_val}")
    print(f"   Layer-specific multipliers:")
    print(f"     Layers 22-29: λ×(1+3β) = {lambda_val * (1 + 3 * beta_val):.4f}")
    print(f"     Other layers: λ×(1-β) = {lambda_val * (1 - beta_val):.4f}")
    
    start_time = time.time()
    
    for idx, sample in enumerate(tqdm(dataset_samples, desc=f"λ={lambda_val}, β={beta_val}", unit="sample")):
        # Entry ID for the dataset
        entry_id = sample["entry_id"]
        audio_index = sample["audio_index"]
        audio_path = f"{args.audio_root_dir}/{audio_index}.wav"
        prompt_text = sample["prompt_text"]
        label = sample["label"]
        sampling_method = sample.get("sampling", "unknown") if use_local_dataset else "unknown"

        # Inference model and get response with verbose flag
        response, is_verbose = inference(audio_path=audio_path, prompt_text=prompt_text)
        
        if is_verbose:
            verbose_count += 1

        # Record evaluation result
        if use_local_dataset:
            evaluation_result = [entry_id, audio_index, label, response, sampling_method, 
                               vsv_enabled, lambda_val, beta_val, is_verbose]
        else:
            evaluation_result = [entry_id, audio_index, label, response, 
                               vsv_enabled, lambda_val, beta_val, is_verbose]
        evaluation_results.append(evaluation_result)

    # Calculate statistics
    total_time = time.time() - start_time
    correct = sum(1 for result in evaluation_results if result[2] == result[3])
    accuracy = correct / len(evaluation_results) * 100
    verbose_percentage = verbose_count / len(evaluation_results) * 100
    
    print(f"   📊 Results: Accuracy: {accuracy:.2f}% ({correct}/{total_samples})")
    print(f"   💬 Verbose responses: {verbose_count}/{total_samples} ({verbose_percentage:.1f}%)")
    print(f"   ⏱️  Time: {total_time/60:.1f} minutes")
    
    # Save individual CSV file for this experiment
    base_path = args.output_path
    if not base_path.endswith('.csv'):
        base_path = base_path + '.csv'
    
    # For grid search, save to grid_search folder
    if hasattr(args, 'grid_search') and args.grid_search:
        # Get the project root directory (2 levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        grid_search_dir = os.path.join(project_root, "grid_search")
        os.makedirs(grid_search_dir, exist_ok=True)
        
        # Use just the filename from the original path
        base_filename = os.path.basename(base_path)
        base_name = base_filename.replace('.csv', '')
        individual_path = os.path.join(grid_search_dir, f"{base_name}_lambda{lambda_val}_beta{beta_val}.csv")
    else:
        # Regular experiment: use original path
        output_dir = os.path.dirname(os.path.abspath(base_path))
        os.makedirs(output_dir, exist_ok=True)
        base_name = base_path.replace('.csv', '')
        individual_path = f"{base_name}_lambda{lambda_val}_beta{beta_val}.csv"
    
    print(f"   💾 Saving results to {individual_path}...")
    try:
        with open(individual_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if use_local_dataset:
                writer.writerow(["entry_id", "audio_index", "label", "response", "sampling_method", 
                               "vsv_enabled", "vsv_lambda", "vsv_beta", "is_verbose"])
            else:
                writer.writerow(["entry_id", "audio_index", "label", "response", 
                               "vsv_enabled", "vsv_lambda", "vsv_beta", "is_verbose"])
            writer.writerows(evaluation_results)
        print(f"   ✅ Results saved to {individual_path}")
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")
    
    return {
        'lambda': lambda_val,
        'beta': beta_val,
        'accuracy': accuracy,
        'correct': correct,
        'total': total_samples,
        'verbose_count': verbose_count,
        'verbose_percentage': verbose_percentage,
        'time_minutes': total_time/60,
        'results': evaluation_results,
        'file_path': individual_path
    }

def main(args):
    global verbose_progress
    verbose_progress = args.verbose
    
    # Grid search parameters
    if args.grid_search:
        lambdas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        betas = [0.0, 0.2, 0.4, 0.6]
        print(f"🔬 Starting GRID SEARCH with {len(lambdas)} λ values and {len(betas)} β values")
        print(f"   λ values: {lambdas}")
        print(f"   β values: {betas}")
        print(f"   Total experiments: {len(lambdas) * len(betas)}")
    else:
        # Single experiment mode
        lambdas = [args.vsv_lambda]
        betas = [args.vsv_beta]
        print(f"🎯 Running single experiment with λ={args.vsv_lambda}, β={args.vsv_beta}")

    # Load and prepare dataset
    if hasattr(args, 'dataset_file') and args.dataset_file:
        print("📊 Loading local dataset...")
        dataset_samples = load_local_dataset(args.dataset_file)
        use_local_dataset = True
    else:
        print("📊 Loading HuggingFace dataset...")
        dataset = load_dataset(args.dataset_name)
        dataset_samples = list(dataset['test'])
        use_local_dataset = False

    # Shuffle and limit dataset
    print("🔀 Randomly shuffling dataset...")
    random.shuffle(dataset_samples)
    
    if hasattr(args, 'max_samples') and args.max_samples and args.max_samples < len(dataset_samples):
        dataset_samples = dataset_samples[:args.max_samples]
        print(f"🔢 Limited to {args.max_samples} samples for testing")
    
    print(f"📝 Dataset prepared: {len(dataset_samples)} samples to process")

    # Initialize model
    if model is None:
        initialize_model()

    # Run experiments
    all_experiment_results = []
    individual_files = []
    
    experiment_num = 0
    total_experiments = len(lambdas) * len(betas)
    overall_start_time = time.time()
    
    for lambda_val in lambdas:
        for beta_val in betas:
            experiment_num += 1
            print(f"\n{'='*80}")
            print(f"🧪 EXPERIMENT {experiment_num}/{total_experiments}")
            print(f"{'='*80}")
            
            # Run single experiment (this will save its own CSV file)
            experiment_result = run_single_experiment(
                dataset_samples, use_local_dataset, args, lambda_val, beta_val
            )
            all_experiment_results.append(experiment_result)
            individual_files.append(experiment_result['file_path'])
    
    # Calculate total time
    total_grid_time = time.time() - overall_start_time
    
    # Print summary results
    print(f"\n{'='*80}")
    print(f"📋 GRID SEARCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_grid_time/3600:.1f} hours ({total_grid_time/60:.1f} minutes)")
    print(f"\nResults sorted by accuracy (highest first):")
    
    # Sort by accuracy for summary
    sorted_results = sorted(all_experiment_results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'λ':<6} {'β':<6} {'Accuracy':<10} {'Verbose%':<9} {'Time(min)':<10}")
    print("-" * 60)
    
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank:<4} {result['lambda']:<6} {result['beta']:<6} "
              f"{result['accuracy']:.2f}%{'':<4} {result['verbose_percentage']:.1f}%{'':<4} "
              f"{result['time_minutes']:.1f}")
    
    # Find best result
    best_result = sorted_results[0]
    print(f"\n🏆 BEST RESULT:")
    print(f"   λ={best_result['lambda']}, β={best_result['beta']}")
    print(f"   Accuracy: {best_result['accuracy']:.2f}%")
    print(f"   Verbose responses: {best_result['verbose_percentage']:.1f}%")
    
    # Save summary results (only for grid search)
    if args.grid_search:
        # Get the project root directory and use grid_search folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        grid_search_dir = os.path.join(project_root, "grid_search")
        os.makedirs(grid_search_dir, exist_ok=True)
        
        base_path = args.output_path
        if not base_path.endswith('.csv'):
            base_path = base_path + '.csv'
        
        base_filename = os.path.basename(base_path)
        base_name = base_filename.replace('.csv', '')
        summary_path = os.path.join(grid_search_dir, f"{base_name}_grid_search_summary.csv")
        
        print(f"\n💾 Saving grid search summary to {summary_path}...")
        try:
            with open(summary_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["rank", "lambda", "beta", "accuracy", "correct", "total", 
                               "verbose_count", "verbose_percentage", "time_minutes", "individual_file"])
                for rank, result in enumerate(sorted_results, 1):
                    writer.writerow([rank, result['lambda'], result['beta'], result['accuracy'], 
                                   result['correct'], result['total'], result['verbose_count'], 
                                   result['verbose_percentage'], result['time_minutes'], result['file_path']])
            print(f"✅ Summary saved successfully to {summary_path}")
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
    
    # Print final completion message
    if args.grid_search:
        print(f"\n🎉 Grid search completed successfully!")
        print(f"📁 Individual result files ({len(individual_files)} total):")
        for i, file_path in enumerate(individual_files, 1):
            result = all_experiment_results[i-1]
            print(f"   {i:2d}. {os.path.basename(file_path)} (Acc: {result['accuracy']:.1f}%, Verbose: {result['verbose_percentage']:.1f}%)")
        if 'summary_path' in locals():
            print(f"\n📊 Summary file: {summary_path}")
    else:
        if len(individual_files) == 1:
            print(f"\n🎉 Experiment completed successfully!")
            print(f"📁 Results saved to: {individual_files[0]}")
        else:
            print(f"\n🎉 Experiments completed successfully!")
            print(f"📁 Results saved to {len(individual_files)} files")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio hallucination evaluation with progress tracking and vector steering")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name.", default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, help="Path to local dataset TSV file (alternative to --dataset_name)", default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, help="Audio root directory", default="./understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, help="Output path of csv file.", default="./prompt_eng_data_prompt_evaluation_result.csv")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output for individual inference steps")
    
    # Vector steering options
    parser.add_argument("--enable_vsv", action="store_true", help="Enable vector steering for audio hallucination mitigation (ignored in grid search mode)")
    parser.add_argument("--vsv_lambda", type=float, default=0.05, help="Vector steering strength (lambda). Higher values = stronger steering. Default: 0.05")
    parser.add_argument("--vsv_beta", type=float, default=0.0, help="Layer-specific steering multiplier (beta). Layers 22-29: (1+3β), others: (1-β). Default: 0.0")
    
    # Grid search option
    parser.add_argument("--grid_search", action="store_true", help="Enable grid search over lambda=[0.02,0.03,0.04,0.05,0.06,0.07] and beta=[0.0,0.25,0.5]")
    
    # Testing options
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    main(args)