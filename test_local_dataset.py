#!/usr/bin/env python3

import csv
import argparse
import torch
import sys
import os
import librosa
import time
import pandas as pd

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
    # Import the hal_inference module
    sys.path.append('src')
    from src.models.hal_inference import main as hal_main, load_local_dataset as hal_load_dataset
    
    # Create a mock args object for hal_inference
    class MockArgs:
        def __init__(self):
            # Load the dataset to see how many samples we have
            dataset_samples = load_local_dataset(args.dataset_file)
            
            # Limit samples if specified
            if args.max_samples and args.max_samples < len(dataset_samples):
                dataset_samples = dataset_samples[:args.max_samples]
                print(f"🔢 Limited to {args.max_samples} samples for testing")
            
            # Store dataset samples for hal_inference to access
            self.dataset_samples = dataset_samples
            self.dataset_name = args.dataset_name
            self.audio_root_dir = args.audio_root_dir
            self.output_path = args.output_path
            self.verbose = args.verbose
            self.enable_vsv = args.enable_vsv
            self.vsv_lambda = args.vsv_lambda
    
    mock_args = MockArgs()
    
    # Modify hal_inference to work with local dataset
    # We'll need to patch the main function temporarily
    
    print("🚀 Starting hal_inference with local dataset...")
    hal_main(mock_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hal_inference with local dataset")
    parser.add_argument("--dataset_file", type=str, 
                       default="understanding_sound_data/metadata/balanced_subset_1500_test.txt")
    parser.add_argument("--dataset_name", type=str, 
                       default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--audio_root_dir", type=str, 
                       default="understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, 
                       default="test_vsv_results.csv")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--enable_vsv", action="store_true")
    parser.add_argument("--vsv_lambda", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    main(args)