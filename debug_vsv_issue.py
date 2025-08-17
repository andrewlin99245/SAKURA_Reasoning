#!/usr/bin/env python3
"""Debug script to investigate VSV-related model_kwargs warning"""

import sys
import os
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src')

def debug_forward_modification():
    print("=== Debugging Forward Method Modification ===")
    
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
    import torch.nn.functional as F
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
        
        print("Inputs keys:", list(inputs.keys()))
        
        # Test 1: Normal generation (should work)
        print("\n--- Test 1: Normal Generation ---")
        with torch.no_grad():
            try:
                output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
                print("✓ Normal generation: SUCCESS (no warnings)")
            except Exception as e:
                print(f"✗ Normal generation: FAILED - {e}")
        
        # Test 2: Generate with modified forward method (like VSV case)
        print("\n--- Test 2: Modified Forward Method ---")
        
        # Create a dummy steering vector
        steering_vector = torch.randn(4096, device=model.device)  # Assuming hidden size 4096
        
        # Create cosine hook
        cosine_hook = CosineWeightedSLAHook(steering_vector, gamma=0.5, w=3)
        cosine_hook.register_hooks(model)
        
        # Store original forward
        original_forward = model.forward
        
        def debug_modified_forward(*args, **kwargs):
            print(f"  Modified forward called with args types: {[type(a) for a in args]}")
            print(f"  Modified forward called with kwargs keys: {list(kwargs.keys())}")
            
            # Set generation position
            if args:
                current_seq_len = args[0].size(1) if hasattr(args[0], 'size') else 'N/A'
            else:
                current_seq_len = kwargs.get('input_ids', inputs['input_ids']).size(1) if 'input_ids' in kwargs else 'N/A'
            
            print(f"  Current sequence length: {current_seq_len}")
            cosine_hook.set_generation_position(current_seq_len - 1 if current_seq_len != 'N/A' else 0)
            
            # Call original forward
            output = original_forward(*args, **kwargs)
            
            # Try to modify logits
            if hasattr(output, 'logits'):
                print(f"  Output has logits: {output.logits.shape}")
                output.logits = cosine_hook.modify_logits(output.logits)
                print("  Applied cosine similarity weighting")
            else:
                print("  Output has no logits")
            
            return output
        
        # Replace forward method
        model.forward = debug_modified_forward
        
        with torch.no_grad():
            try:
                print("  Attempting generation with modified forward...")
                output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
                print("✓ Modified forward generation: SUCCESS")
            except Exception as e:
                print(f"✗ Modified forward generation: FAILED - {e}")
        
        # Restore original forward
        model.forward = original_forward
        cosine_hook.remove_hooks()
        
        # Test 3: Check prepare_inputs_for_generation directly
        print("\n--- Test 3: prepare_inputs_for_generation Debug ---")
        
        # Test the prepare_inputs_for_generation method directly
        input_ids = inputs['input_ids']
        print(f"Input IDs shape: {input_ids.shape}")
        
        # Call prepare_inputs_for_generation directly
        try:
            prepared = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                **inputs
            )
            print(f"Prepared inputs keys: {list(prepared.keys())}")
            print("✓ prepare_inputs_for_generation: SUCCESS")
        except Exception as e:
            print(f"✗ prepare_inputs_for_generation: FAILED - {e}")
            
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_forward_modification()