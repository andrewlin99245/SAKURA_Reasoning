import os
import sys
import torch
import librosa
import pandas as pd
import json
import csv
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig

# --- add path to the two helper files you saved earlier ---
# Make sure steering_vector_audio.py and llm_layers_audio.py are in this directory.
sys.path.insert(0, os.path.abspath("."))  # or the folder where the files are located

from steering_vector_fisher import obtain_vsv
from llm_layer_fisher import add_vsv_layers, remove_vsv_layers

# ---------------------
# Configuration
# ---------------------
MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
AUDIO_ROOT_DIR = "./audiocaps"  # Default audio directory
SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/data"

# Dataset names
DISCRIMINATIVE_DATASETS = [
    "kuanhuggingface/AudioHallucination_AudioCaps-Random",
    "kuanhuggingface/AudioHallucination_AudioCaps-Popular", 
    "kuanhuggingface/AudioHallucination_AudioCaps-Adversarial"
]

# ---------------------
# Load model + SLA on
# ---------------------
config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
model = Qwen2AudioSLAForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ---------------------
# Helpers to build inputs
# ---------------------
def build_messages(include_audio: bool, wav_path: str, prompt: str):
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
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

def inference(audio_path, prompt):
    """
    Inference function for discriminative tasks
    Returns the model's response to yes/no questions
    """
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios = []
    for message in messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audio, sr = librosa.load(ele["audio_url"])
                    if sr != 16000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    audios.append(audio)
    
    inputs = processor(text=text, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, max_new_tokens=32)  # Shorter for yes/no answers
    output = output[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text[0] if text else "No"

def run_discriminative_evaluation(gamma, w, lam, fisher_scale, model_suffix, dataset_name, audio_root_dir, max_samples=-1):
    """
    Run evaluation on discriminative datasets
    """
    print(f"\n=== Running discriminative evaluation on {dataset_name} ===")
    print(f"Parameters: gamma={gamma}, w={w}, lam={lam}, fisher_scale={fisher_scale}")
    
    # Configure SLA
    model.enable_sla(gamma=gamma, w=w)
    
    # Prepare VSV using a sample from Animal dataset (if available)
    vsv_wav_path = "/home/andrew99245/SAKURA_Reasoning/data/Animal/audio/rooster39.wav"
    if os.path.exists(vsv_wav_path):
        audio, _ = librosa.load(vsv_wav_path, sr=16000)
        
        sample_prompt = "What's the animal in the audio?"
        messages_pos = build_messages(include_audio=True,  wav_path=vsv_wav_path, prompt=sample_prompt)
        messages_neg = build_messages(include_audio=False, wav_path=vsv_wav_path, prompt=sample_prompt)
        
        pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
        neg_inputs = build_inputs(messages_neg, audio=None,  sr=16000)
        
        # Compute VSV and inject
        with torch.no_grad():
            kwargs_list = [[neg_inputs, pos_inputs]]
            vsv = obtain_vsv(model, kwargs_list)
            vsv = vsv.to(model.device)
        
        # Inject VSV
        add_vsv_layers(model, vsv=vsv, lam=lam, fisher_scale=fisher_scale, which_stack="decoder")
        print("VSV layers added successfully")
    else:
        print("Warning: VSV preparation audio not found, skipping VSV injection")

    # Load the discriminative dataset
    try:
        dataset = load_dataset(dataset_name)
        print(f"Dataset loaded successfully: {len(dataset['test'])} samples")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return

    # Prepare results directory
    os.makedirs("./discriminative_results", exist_ok=True)
    
    # Extract dataset type from name (Random, Popular, Adversarial)
    dataset_type = dataset_name.split('-')[-1]
    result_file = f"./discriminative_results/{dataset_type}_{model_suffix}_discriminative.csv"
    
    # Evaluation results
    evaluation_results = []
    
    # Determine number of samples to process
    total_samples = len(dataset['test'])
    num_samples = total_samples if max_samples == -1 else min(max_samples, total_samples)
    print(f"Processing {num_samples} out of {total_samples} samples")

    for i in tqdm(range(num_samples), desc=f"Evaluating {dataset_type}"):
        sample = dataset['test'][i]
        
        # Extract sample information
        entry_id = sample["entry_id"]
        audio_index = sample["audio_index"]
        prompt_text = sample["prompt_text"]
        object_name = sample["object"]
        label = sample["label"]
        sampling_method = sample["sampling"]
        
        # Construct audio path
        audio_path = f"{audio_root_dir}/{audio_index}.wav"
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            response = "No"  # Default response when audio is missing
        else:
            # Run inference
            try:
                response = inference(audio_path=audio_path, prompt=prompt_text)
            except Exception as e:
                print(f"Error during inference for {audio_index}: {e}")
                response = "No"  # Default response on error
        
        # Record evaluation result
        evaluation_result = [
            entry_id, audio_index, object_name, sampling_method, 
            label, response, prompt_text
        ]
        evaluation_results.append(evaluation_result)
        
        # Optional: print sample results for debugging
        if i < 5:  # Print first 5 results
            print(f"Sample {i}: {audio_index}, Label: {label}, Response: {response}")
    
    # Save results to CSV
    header = ["entry_id", "audio_index", "object", "sampling", "label", "response", "prompt_text"]
    with open(result_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(evaluation_results)
    
    print(f"Results saved to: {result_file}")
    
    # Calculate basic metrics
    correct = 0
    total = len(evaluation_results)
    yes_count = 0
    positive_labels = 0
    
    for result in evaluation_results:
        label = result[4]  # label column
        response = result[5]  # response column
        
        # Normalize responses for comparison
        normalized_response = "positive" if "yes" in response.lower() else "negative"
        if normalized_response == label:
            correct += 1
        
        if "yes" in response.lower():
            yes_count += 1
        if label == "positive":
            positive_labels += 1
    
    accuracy = correct / total if total > 0 else 0
    yes_rate = yes_count / total if total > 0 else 0
    
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Yes rate: {yes_rate:.3f} ({yes_count}/{total})")
    print(f"Positive labels: {positive_labels}/{total}")
    
    # Remove VSV hooks for next evaluation
    if os.path.exists(vsv_wav_path):
        remove_vsv_layers(model, which_stack="decoder")
        print("VSV layers removed")

def main():
    parser = argparse.ArgumentParser(description="Discriminative evaluation for audio hallucination datasets")
    parser.add_argument("--datasets", nargs='+', default=DISCRIMINATIVE_DATASETS,
                        help="List of discriminative datasets to evaluate")
    parser.add_argument("--audio_root_dir", type=str, default=AUDIO_ROOT_DIR,
                        help="Root directory containing audio files")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of samples to process (-1 for all)")
    parser.add_argument("--gamma", type=float, default=0.25,
                        help="SLA gamma parameter")
    parser.add_argument("--w", type=float, default=4,
                        help="SLA w parameter")
    parser.add_argument("--lam", type=float, default=0.05,
                        help="VSV lambda parameter")
    parser.add_argument("--fisher_scale", type=float, default=0.5,
                        help="Fisher scale parameter")
    parser.add_argument("--model_suffix", type=str, default="sla_vsv",
                        help="Suffix for output files")
    
    args = parser.parse_args()
    
    print("=== Discriminative Evaluation Script ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Audio root: {args.audio_root_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Parameters: gamma={args.gamma}, w={args.w}, lam={args.lam}, fisher_scale={args.fisher_scale}")
    
    # Run evaluation on each dataset
    for dataset_name in args.datasets:
        try:
            run_discriminative_evaluation(
                gamma=args.gamma,
                w=args.w,
                lam=args.lam,
                fisher_scale=args.fisher_scale,
                model_suffix=args.model_suffix,
                dataset_name=dataset_name,
                audio_root_dir=args.audio_root_dir,
                max_samples=args.max_samples
            )
        except Exception as e:
            print(f"Error evaluating dataset {dataset_name}: {e}")
            continue
    
    print("=== Evaluation Complete ===")

if __name__ == "__main__":
    main()