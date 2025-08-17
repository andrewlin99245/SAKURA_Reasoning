#!/usr/bin/env python3
"""Debug script to compare original vs modified forward behavior"""

import sys
import os
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src')

def debug_forward_calls():
    print("=== Debugging Forward Method Calls ===")
    
    # Import cosine module
    import hal_inference_cosine_weighted
    from hal_inference_cosine_weighted import CosineWeightedSLAHook
    
    hal_inference_cosine_weighted.initialize_model()
    
    # Get the initialized objects
    processor = hal_inference_cosine_weighted.processor
    model = hal_inference_cosine_weighted.model
    
    # Test the same audio file
    import librosa
    import torch
    audio_path = './understanding_sound_data/audio/Y11SEBDuoqSk.wav'
    
    try:
        # Load audio and prepare inputs (same as before)
        audio, sr = librosa.load(audio_path, sr=16000)
        prompt_text = "Is there a dog barking in this audio?"
        modified_prompt = f"{prompt_text} Answer just yes or no."
        
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": modified_prompt},
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
        
        print("Inputs keys:", list(inputs.keys()))
        
        # Test: Compare what gets passed to forward in both cases
        print("\n--- Forward Call Comparison ---")
        
        # Store original forward
        original_forward = model.forward
        
        # Create a logging version of original forward
        def logging_original_forward(*args, **kwargs):
            print(f"ORIGINAL Forward - Args: {len(args)} args")
            if args:
                print(f"  arg[0] type: {type(args[0])}, shape: {getattr(args[0], 'shape', 'no shape')}")
            print(f"ORIGINAL Forward - Kwargs: {list(kwargs.keys())}")
            for k, v in kwargs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
            
            result = original_forward(*args, **kwargs)
            print(f"ORIGINAL Forward - Result type: {type(result)}")
            if hasattr(result, 'logits'):
                print(f"  logits shape: {result.logits.shape}")
            return result
        
        # Create a logging version of modified forward
        def logging_modified_forward(*args, **kwargs):
            print(f"MODIFIED Forward - Args: {len(args)} args")
            if args:
                print(f"  arg[0] type: {type(args[0])}, shape: {getattr(args[0], 'shape', 'no shape')}")
            print(f"MODIFIED Forward - Kwargs: {list(kwargs.keys())}")
            for k, v in kwargs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
            
            # Call original and log
            result = original_forward(*args, **kwargs)
            print(f"MODIFIED Forward - Result type: {type(result)}")
            if hasattr(result, 'logits'):
                print(f"  logits shape: {result.logits.shape}")
                # Here's where I'd modify logits in the real implementation
            return result
        
        print("\n=== Test 1: With Logging Original Forward ===")
        model.forward = logging_original_forward
        
        with torch.no_grad():
            try:
                print("Calling generate with logging original forward...")
                output = model.generate(**inputs, max_new_tokens=2, do_sample=False)  # Use greedy for consistency
                print("✓ Logging original: SUCCESS")
            except Exception as e:
                print(f"✗ Logging original: FAILED - {e}")
        
        print("\n=== Test 2: With Logging Modified Forward ===")
        model.forward = logging_modified_forward
        
        with torch.no_grad():
            try:
                print("Calling generate with logging modified forward...")
                output = model.generate(**inputs, max_new_tokens=2, do_sample=False)  # Use greedy for consistency
                print("✓ Logging modified: SUCCESS")
            except Exception as e:
                print(f"✗ Logging modified: FAILED - {e}")
        
        # Restore original
        model.forward = original_forward
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_forward_calls()