#!/usr/bin/env python3
"""
Comprehensive discriminative evaluation script for audio hallucination assessment.
Based on the "Understanding Sound" paper methodology.
"""

import argparse
import csv
import os
import sys
from tqdm import tqdm

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'runners'))

from hal_inference import inference
from discriminative_eval_config import (
    DISCRIMINATIVE_PROMPTS, 
    PROMPT_PREFIXES, 
    OUTPUT_CONFIG,
    SYSTEM_PROMPT
)

def load_local_dataset(metadata_file, audio_root_dir):
    """Load dataset from local metadata files."""
    samples = []
    
    with open(metadata_file, 'r') as f:
        lines = f.readlines()
        
    # Skip header and comment lines
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('entry_id') or not line:
            continue
            
        parts = line.split('\t')
        if len(parts) >= 6:
            entry_id, audio_index, prompt_text, obj, attribute, label = parts[:6]
            
            sample = {
                'entry_id': entry_id,
                'audio_index': audio_index,
                'prompt_text': prompt_text,
                'object': obj,
                'attribute': attribute,
                'label': label,
                'audio_path': os.path.join(audio_root_dir, f"{audio_index}.wav")
            }
            samples.append(sample)
    
    return samples

def run_evaluation(metadata_file, audio_root_dir, output_path, prompt_prefix=None, max_samples=None):
    """Run discriminative evaluation on the dataset."""
    
    print(f"Loading dataset from {metadata_file}")
    samples = load_local_dataset(metadata_file, audio_root_dir)
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    print(f"Processing {len(samples)} samples...")
    
    # Prepare results
    evaluation_results = []
    
    for sample in tqdm(samples, desc="Running inference"):
        entry_id = sample['entry_id']
        audio_index = sample['audio_index']
        audio_path = sample['audio_path']
        prompt_text = sample['prompt_text']
        label = sample['label']
        
        # Apply prompt prefix if specified
        if prompt_prefix and prompt_prefix in PROMPT_PREFIXES:
            prompt_text = f"{PROMPT_PREFIXES[prompt_prefix]} {prompt_text}"
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            response = "No"
        else:
            # Run inference
            try:
                response = inference(audio_path, prompt_text)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                response = "No"
        
        # Record result
        evaluation_result = [entry_id, audio_index, label, response]
        evaluation_results.append(evaluation_result)
    
    # Save results
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(OUTPUT_CONFIG["csv_columns"])
        writer.writerows(evaluation_results)
    
    print(f"Evaluation results saved to {output_path}")
    
    # Calculate basic accuracy
    correct = sum(1 for result in evaluation_results if result[2] == result[3])
    total = len(evaluation_results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Run discriminative evaluation for audio hallucination assessment")
    parser.add_argument("--metadata_file", type=str, required=True, 
                       help="Path to metadata file (e.g., understanding_sound_data/metadata/random_test.txt)")
    parser.add_argument("--audio_root_dir", type=str, default="./understanding_sound_data/audio",
                       help="Audio root directory")
    parser.add_argument("--output_path", type=str, default="./discriminative_evaluation_result.csv",
                       help="Output path for CSV results")
    parser.add_argument("--prompt_prefix", type=str, choices=list(PROMPT_PREFIXES.keys()),
                       help="Prompt engineering prefix to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.metadata_file):
        print(f"Error: Metadata file not found: {args.metadata_file}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_root_dir):
        print(f"Error: Audio directory not found: {args.audio_root_dir}")
        sys.exit(1)
    
    print(f"Starting discriminative evaluation...")
    print(f"Metadata file: {args.metadata_file}")
    print(f"Audio directory: {args.audio_root_dir}")
    print(f"Output file: {args.output_path}")
    if args.prompt_prefix:
        print(f"Prompt prefix: {args.prompt_prefix} - '{PROMPT_PREFIXES[args.prompt_prefix]}'")
    print()
    
    # Run evaluation
    results = run_evaluation(
        metadata_file=args.metadata_file,
        audio_root_dir=args.audio_root_dir,
        output_path=args.output_path,
        prompt_prefix=args.prompt_prefix,
        max_samples=args.max_samples
    )
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()