#!/usr/bin/env python3
"""
Test script to work directly with local audio files and metadata
"""

import csv
import json
import os
import sys
from tqdm import tqdm
import pandas as pd


class MockAudioLanguageModel:
    """Mock model for testing purposes"""
    def __init__(self):
        print("Mock model initialized")
    
    def process_audio_and_generate(self, audio_path, prompt_text):
        """Mock inference that returns sample responses"""
        mock_responses = {
            "Describe the audio.": "The audio contains sounds of people talking and some background noise.",
            "What do you hear?": "I hear speech and ambient sounds in the background.",
            "What can be inferred from the audio?": "This appears to be a recording of human conversation.",
            "This is a sound of": "human speech with background activity.",
            "Generate audio caption:": "Audio clip featuring people speaking with environmental sounds."
        }
        return mock_responses.get(prompt_text, "I hear various sounds in this audio clip.")


def get_generative_task_prompt():
    """Get the generative task prompts as specified in the paper"""
    return [
        "Describe the audio.",
        "What do you hear?", 
        "What can be inferred from the audio?",
        "This is a sound of",
        "Generate audio caption:"
    ]


def inference(audio_path, prompt_text, model):
    """Perform inference on audio with given prompt"""
    try:
        if not os.path.exists(audio_path):
            return f"Error: Audio file not found - {audio_path}"
        
        # Use mock model
        response = model.process_audio_and_generate(audio_path, prompt_text)
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"


def test_with_local_metadata():
    """Test with local metadata and audio files"""
    
    # Paths
    metadata_file = "understanding_sound_data/metadata/generative_data_metadata.csv"
    audio_dir = "understanding_sound_data/audio"
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return
    
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(metadata_file)
    print(f"Loaded {len(df)} entries from metadata")
    
    # Initialize model
    model = MockAudioLanguageModel()
    
    # Get prompts
    prompts = get_generative_task_prompt()
    
    # Results storage
    all_results = {}
    
    # Process a few samples for testing
    max_samples = 3
    sample_count = 0
    
    for index, row in df.iterrows():
        if sample_count >= max_samples:
            break
            
        try:
            # Extract data
            youtube_id = row['youtube_id']
            caption = eval(row['caption']) if isinstance(row['caption'], str) else row['caption']
            label = eval(row['label']) if isinstance(row['label'], str) else row['label']
            
            # Look for audio file
            audio_filename = f"{youtube_id}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue
            
            print(f"\nProcessing {youtube_id}...")
            
            # Initialize results for this audio
            all_results[youtube_id] = {}
            
            # Test each prompt
            for prompt_text in prompts:
                print(f"  Testing prompt: {prompt_text}")
                
                # Perform inference
                response = inference(audio_path, prompt_text, model)
                
                # Store results
                all_results[youtube_id][prompt_text] = {
                    "prediction": response,
                    "caption": caption,
                    "label": label,
                    "task": "generative",
                    "audio_path": audio_path
                }
                
                print(f"    Response: {response}")
            
            sample_count += 1
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    # Save results
    output_file = "local_test_results.json"
    print(f"\nSaving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"Test completed! Processed {sample_count} audio samples")
    return all_results


if __name__ == "__main__":
    results = test_with_local_metadata()
    if results:
        print(f"\nSuccessfully processed {len(results)} audio samples")
        print("\nSample results structure:")
        first_key = list(results.keys())[0]
        first_prompt = list(results[first_key].keys())[0]
        print(f"Sample entry for {first_key}, prompt '{first_prompt}':")
        print(json.dumps(results[first_key][first_prompt], indent=2))
    else:
        print("No results generated")