import os
import sys
import warnings

# Suppress all warnings early - use regex patterns to catch variations
warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting.*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*image processor.*fast processor.*")
warnings.filterwarnings("ignore", message=".*video processor config.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Unrecognized keys.*rope_scaling.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
# Specifically target the exact EOS warning message format
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:8292 for open-end generation.")

# CRITICAL: Set up cache environment variables BEFORE any other imports
# This ensures consistent cache usage across all modules
SHARED_CACHE_DIR = os.path.expanduser("~/.cache/sakura_reasoning")
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)

# Set environment variable to suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set all relevant HuggingFace cache environment variables
os.environ["HF_HOME"] = SHARED_CACHE_DIR
# Remove deprecated TRANSFORMERS_CACHE
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
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
from transformers import Qwen2_5OmniProcessor

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

from utils.Qwen2_5Omni_patch import Qwen2_5OmniSLAForCausalLM

# Import vector steering modules
try:
    from steering_vector_qwen2_5omni import obtain_vsv
    from ..layers.llm_layer_qwen2_5omni import add_vsv_layers, remove_vsv_layers
    VSV_AVAILABLE = True
except ImportError as e:
    try:
        # Try absolute imports from project root
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.insert(0, project_dir)
        from src.models.steering_vector_qwen2_5omni import obtain_vsv
        from src.models.llm_layer_qwen2_5omni import add_vsv_layers, remove_vsv_layers
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
vsv_prepared = False
vsv_tensor = None

def initialize_model():
    """Initialize the model and processor globally"""
    global model, processor
    
    MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
    
    print("🚀 Initializing Qwen2.5Omni model...")
    
    print("  🤖 Loading model (this may take a few minutes)...")
    model = Qwen2_5OmniSLAForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    
    print("  🔧 Loading processor...")
    
    # Fix deprecated video processor config if it exists
    try:
        from transformers.utils import cached_file
        config_path = cached_file(MODEL_PATH, "preprocessor.json", _raise_exceptions_for_missing_entries=False)
        if config_path and os.path.exists(config_path):
            config_dir = os.path.dirname(config_path)
            new_path = os.path.join(config_dir, "video_preprocessor.json")
            if not os.path.exists(new_path):
                os.rename(config_path, new_path)
                print(f"  📁 Renamed deprecated video processor config")
    except Exception:
        pass
    
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    
    print("  ⚡ Enabling SLA...")
    # Enable SLA (as used in the existing codebase)
    model.enable_sla(gamma=0.0, w=4)
    
    print("✅ Qwen2.5Omni model initialization complete!")

def build_messages(include_audio: bool, wav_path: str, prompt: str):
    """Build messages for VSV computation"""
    # Use proper Qwen2.5Omni system prompt format to avoid warnings
    base = [{
        "role": "system", 
        "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
    }]
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
    try:
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
    except Exception as e:
        print(f"Error in build_inputs: {e}")
        print(f"Messages type: {type(messages)}, messages: {messages}")
        print(f"Audio type: {type(audio) if audio is not None else 'None'}")
        raise

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
    vsv_prompt = prompt
    
    # Build positive and negative inputs for VSV computation using the data_prompt
    messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    messages_neg = build_messages(include_audio=False, wav_path=audio_path, prompt=vsv_prompt)  # Text-only negative
    
    pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, audio=soundless_audio, sr=16000)
    
    # Compute VSV specific to this input
    with torch.no_grad():
        # Debug: Check the inputs format
        if verbose_progress:
            print(f"    🔍 pos_inputs type: {type(pos_inputs)}, keys: {list(pos_inputs.keys()) if isinstance(pos_inputs, dict) else 'N/A'}")
            print(f"    🔍 neg_inputs type: {type(neg_inputs)}, keys: {list(neg_inputs.keys()) if isinstance(neg_inputs, dict) else 'N/A'}")
        
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    if verbose_progress:
        print(f"    ✅ VSV computed with shape: {vsv.shape}")
    
    return vsv

def inference(audio_path, prompt_text):
    """
    Perform inference on audio with the given prompt text.
    Returns 'Yes' or 'No' for discriminative tasks.
    Supports vector steering if enabled.
    """
    global model, processor, verbose_progress, vsv_enabled, vsv_lambda
    
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
            
            # Inject VSV
            add_vsv_layers(model, vsv=vsv, lam=vsv_lambda, which_stack="decoder")
            vsv_applied = True
            
            if verbose_progress:
                print(f"    ✅ Vector steering applied with λ={vsv_lambda}")
        
        # Build messages in the expected format
        # Append instruction to answer only yes or no
        modified_prompt = prompt_text
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
            },
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
        # Move tensors to model device carefully
        device_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                device_inputs[k] = v.to(model.device)
                # Only convert float tensors to model dtype
                if v.dtype.is_floating_point:
                    device_inputs[k] = device_inputs[k].to(model.dtype)
            else:
                device_inputs[k] = v
        inputs = device_inputs

        if verbose_progress:
            print("    🧠 Generating response...")
        # Generate response
        with torch.no_grad():
            try:
                # Suppress warnings during generation specifically
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)
            except Exception as gen_error:
                if verbose_progress:
                    print(f"    ❌ Generation error: {gen_error}")
                raise gen_error

        if verbose_progress:
            print("    📤 Decoding output...")
        # Decode output - handle tuple from SLA model
        if isinstance(output, tuple):
            tokens = output[0]  # Get the tokens from the tuple
        else:
            tokens = output
        tokens = tokens[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
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
    print(f"📋 Header: {header}")
    
    # Parse data
    data = []
    skipped_lines = 0
    for line_idx, line in enumerate(data_lines[1:], 1):
        if line:  # Skip empty lines
            fields = line.split('\t')
            if len(fields) >= 6:  # Ensure we have all required fields
                sample_dict = {
                    'entry_id': fields[0],
                    'audio_index': fields[1], 
                    'prompt_text': fields[2],
                    'object': fields[3],
                    'attribute': fields[4],
                    'label': fields[5],
                    'sampling': fields[6] if len(fields) > 6 else 'unknown'
                }
                data.append(sample_dict)
            else:
                print(f"⚠️  Skipping line {line_idx}: insufficient fields ({len(fields)} < 6): {line[:100]}...")
                skipped_lines += 1
    
    print(f"📊 Loaded {len(data)} samples (skipped {skipped_lines} malformed lines)")
    if len(data) > 0:
        print(f"🔍 First sample: {data[0]}")
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
        
        total_samples = len(dataset_samples)
        use_local_dataset = False
        print(f"📝 Dataset loaded: {total_samples} samples to process")

    # Evaluation results.
    evaluation_results = []
    
    # Initialize model before processing (if not already initialized)
    if model is None:
        initialize_model()

    # Print vector steering configuration
    if vsv_enabled:
        print(f"🎯 Vector steering ENABLED with λ={vsv_lambda}")
    else:
        print("🎯 Vector steering DISABLED")

    print(f"🎯 Starting inference on {total_samples} samples...")
    start_time = time.time()
    
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Processing samples", unit="sample")):
        try:
            # Debug: Check if sample is the expected type
            if not isinstance(sample, dict):
                print(f"Warning: Sample {idx} is not a dictionary but {type(sample)}: {sample}")
                continue

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
        except KeyError as ke:
            print(f"Warning: Sample {idx} missing required key {ke}: {sample}")
            continue
        except Exception as e:
            print(f"Warning: Error parsing sample {idx}: {e}")
            print(f"Sample type: {type(sample)}, Sample: {sample}")
            continue

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
    
    # Update output filename to include steering info if enabled
    output_path = args.output_path
    if vsv_enabled:
        # Insert steering info before file extension
        name_parts = output_path.rsplit('.', 1)
        if len(name_parts) == 2:
            output_path = f"{name_parts[0]}_qwen2_5omni_vsv_lambda{vsv_lambda}.{name_parts[1]}"
        else:
            output_path = f"{output_path}_qwen2_5omni_vsv_lambda{vsv_lambda}"
    else:
        # Insert model info even when VSV is disabled
        name_parts = output_path.rsplit('.', 1)
        if len(name_parts) == 2:
            output_path = f"{name_parts[0]}_qwen2_5omni.{name_parts[1]}"
        else:
            output_path = f"{output_path}_qwen2_5omni"
    
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
    parser = argparse.ArgumentParser(description="Audio hallucination evaluation with progress tracking and vector steering using Qwen2.5Omni")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name.", default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, help="Path to local dataset TSV file (alternative to --dataset_name)", default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, help="Audio root directory", default="./understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, help="Output path of csv file.", default="./data_prompt_evaluation_result_qwen2_5omni.csv")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output for individual inference steps")
    
    # Vector steering options
    parser.add_argument("--enable_vsv", action="store_true", help="Enable vector steering for audio hallucination mitigation")
    parser.add_argument("--vsv_lambda", type=float, default=0.05, help="Vector steering strength (lambda). Higher values = stronger steering. Default: 0.05")
    
    # Testing options
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    main(args)