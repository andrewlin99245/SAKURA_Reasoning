#!/usr/bin/env python3
"""Debug script to recreate the exact VSV scenario that causes warnings"""

import sys
import os
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src')

def debug_exact_vsv_scenario():
    print("=== Recreating Exact VSV Scenario ===")
    
    # Mimic the exact scenario from hal_inference_cosine_weighted.py
    import hal_inference_cosine_weighted
    from hal_inference_cosine_weighted import CosineWeightedSLAHook, compute_vsv_for_audio
    
    # Set global variables like in the actual script
    hal_inference_cosine_weighted.cosine_sla_gamma = 0.5
    hal_inference_cosine_weighted.cosine_sla_w = 3
    hal_inference_cosine_weighted.vsv_enabled = True
    hal_inference_cosine_weighted.vsv_lambda = 0.05
    
    hal_inference_cosine_weighted.initialize_model()
    
    # Get the initialized objects
    processor = hal_inference_cosine_weighted.processor
    model = hal_inference_cosine_weighted.model
    
    import librosa
    import torch
    audio_path = './understanding_sound_data/audio/Y11SEBDuoqSk.wav'
    prompt_text = "Is there a dog barking in this audio?"
    
    print(f"VSV enabled: {hal_inference_cosine_weighted.vsv_enabled}")
    print(f"Cosine gamma: {hal_inference_cosine_weighted.cosine_sla_gamma}")
    
    try:
        # This mirrors the exact inference() function flow
        print("\n--- Step 1: Computing VSV ---")
        vsv = compute_vsv_for_audio(audio_path, prompt_text)
        print(f"VSV computed: shape {vsv.shape}")
        
        print("\n--- Step 2: Creating Cosine Hook ---")
        cosine_hook = CosineWeightedSLAHook(vsv, hal_inference_cosine_weighted.cosine_sla_gamma, hal_inference_cosine_weighted.cosine_sla_w)
        cosine_hook.register_hooks(model)
        print("Hooks registered")
        
        print("\n--- Step 3: Preparing inputs (exact same way as inference) ---")
        # Exact same message building as in inference()
        modified_prompt = f"{prompt_text} Answer just yes or no."
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": modified_prompt},
            ]},
        ]
        
        # Exact same processing
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
        
        print(f"Inputs prepared: {list(inputs.keys())}")
        
        print("\n--- Step 4: Testing generation WITHOUT forward modification ---")
        with torch.no_grad():
            try:
                output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
                print("✓ Without forward modification: SUCCESS")
            except Exception as e:
                if "model_kwargs" in str(e):
                    print("✗ Without forward modification: model_kwargs warning")
                else:
                    print(f"✗ Without forward modification: {e}")
        
        print("\n--- Step 5: Testing generation WITH forward modification (exact VSV code) ---")
        # Store original forward
        original_forward = model.forward
        input_length = inputs['input_ids'].size(1)
        
        def modified_forward(*args, **kwargs):
            # Exact same logic as in inference()
            current_seq_len = args[0].size(1) if args else kwargs.get('input_ids', inputs['input_ids']).size(1)
            cosine_hook.set_generation_position(current_seq_len - 1)
            
            output = original_forward(*args, **kwargs)
            if hasattr(output, 'logits'):
                output.logits = cosine_hook.modify_logits(output.logits)
            return output
        
        # Replace forward method (exact same way as VSV)
        model.forward = modified_forward
        
        with torch.no_grad():
            try:
                output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
                print("✓ With forward modification: SUCCESS")
            except Exception as e:
                if "model_kwargs" in str(e):
                    print("✗ With forward modification: model_kwargs warning - THIS IS THE BUG!")
                else:
                    print(f"✗ With forward modification: {e}")
        
        # Restore original forward
        model.forward = original_forward
        cosine_hook.remove_hooks()
        
        print("\n--- Step 6: Testing generation AFTER restoring forward ---")
        with torch.no_grad():
            try:
                output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
                print("✓ After restoring forward: SUCCESS")
            except Exception as e:
                if "model_kwargs" in str(e):
                    print("✗ After restoring forward: model_kwargs warning")
                else:
                    print(f"✗ After restoring forward: {e}")
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_exact_vsv_scenario()