#!/usr/bin/env python3
"""Debug script to compare input handling between original and cosine implementations"""

import sys
import os
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src')

def test_original():
    print("=== Testing Original hal_inference.py ===")
    
    # Import original module
    import hal_inference
    
    hal_inference.initialize_model()
    
    # Get the initialized objects
    processor = hal_inference.processor
    model = hal_inference.model
    
    # Test the same audio file
    import librosa
    audio_path = './understanding_sound_data/audio/Y11SEBDuoqSk.wav'
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Build test prompt (same as in inference)
        prompt_text = "Is there a dog barking in this audio?"
        modified_prompt = f"{prompt_text} Answer just yes or no."
        
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": modified_prompt},
            ]},
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
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

        # Prepare inputs
        inputs = processor(text=text, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        
        print("Original inputs keys:", list(inputs.keys()))
        print("Model type:", type(model).__name__)
        print("Model has enable_sla:", hasattr(model, 'enable_sla'))
        print("Model sla_enable:", getattr(model, 'sla_enable', 'Not found'))
        print("Model sla_gamma:", getattr(model, 'sla_gamma', 'Not found'))
        
        # Try generation with explicit error catching
        print("Attempting generation...")
        import torch
        with torch.no_grad():
            try:
                output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
                print("✓ Generation successful, no warnings")
                return True
            except Exception as e:
                print(f"✗ Generation failed: {e}")
                return False
                
    except Exception as e:
        print(f"Error in original test: {e}")
        return False

def test_cosine():
    print("\n=== Testing Cosine-weighted Implementation ===")
    
    # Clean up any previous model
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Import cosine module
    import hal_inference_cosine_weighted
    
    hal_inference_cosine_weighted.initialize_model()
    
    # Get the initialized objects
    processor = hal_inference_cosine_weighted.processor
    model = hal_inference_cosine_weighted.model
    
    # Test the same audio file
    import librosa
    audio_path = './understanding_sound_data/audio/Y11SEBDuoqSk.wav'
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Build test prompt (same as in inference)
        prompt_text = "Is there a dog barking in this audio?"
        modified_prompt = f"{prompt_text} Answer just yes or no."
        
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": modified_prompt},
            ]},
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
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

        # Prepare inputs
        inputs = processor(text=text, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        
        print("Cosine inputs keys:", list(inputs.keys()))
        print("Model type:", type(model).__name__)
        print("Model has enable_sla:", hasattr(model, 'enable_sla'))
        print("Model sla_enable:", getattr(model, 'sla_enable', 'Not found'))
        print("Model sla_gamma:", getattr(model, 'sla_gamma', 'Not found'))
        
        # Try generation with explicit error catching
        print("Attempting generation...")
        import torch
        with torch.no_grad():
            try:
                output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
                print("✓ Generation successful, no warnings")
                return True
            except Exception as e:
                print(f"✗ Generation failed: {e}")
                return False
                
    except Exception as e:
        print(f"Error in cosine test: {e}")
        return False

if __name__ == "__main__":
    orig_success = test_original()
    cosine_success = test_cosine()
    
    print(f"\n=== Summary ===")
    print(f"Original: {'✓' if orig_success else '✗'}")
    print(f"Cosine: {'✓' if cosine_success else '✗'}")