#!/usr/bin/env python3
"""
Test script for discriminative inference.
This tests the inference function with a sample audio file and prompt.
"""

import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'runners'))

from hal_inference import inference

def test_inference():
    """Test the inference function with a sample audio file."""
    
    # Use a sample audio file from the understanding_sound_data directory
    audio_path = "/home/andrew99245/SAKURA_Reasoning/understanding_sound_data/audio/Y7fmOlUlwoNg.wav"
    
    # Test with a simple prompt based on the metadata structure
    prompt_text = "Is there a sound of speech in the audio?"
    
    print(f"Testing inference with:")
    print(f"Audio file: {os.path.basename(audio_path)}")
    print(f"Prompt: {prompt_text}")
    print(f"Loading model and running inference...")
    
    try:
        response = inference(audio_path, prompt_text)
        print(f"Response: {response}")
        
        # Test another prompt
        prompt_text2 = "Is there a sound of a dog barking in the audio?"
        print(f"\nTesting second prompt: {prompt_text2}")
        response2 = inference(audio_path, prompt_text2)
        print(f"Response: {response2}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_inference()