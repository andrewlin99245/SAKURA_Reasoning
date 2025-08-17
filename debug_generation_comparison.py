#!/usr/bin/env python3
"""Compare original model.generate() vs custom generation with gamma=0"""

import sys
import os
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src')

def compare_generation_methods():
    print("=== Comparing Original vs Custom Generation (gamma=0) ===")
    
    # Import modules
    import hal_inference_cosine_weighted
    from hal_inference_cosine_weighted import generate_with_cosine_weighting, compute_vsv_for_audio
    
    # Set parameters for gamma=0 test
    hal_inference_cosine_weighted.cosine_sla_gamma = 0.0
    hal_inference_cosine_weighted.cosine_sla_w = 3
    hal_inference_cosine_weighted.vsv_enabled = True
    hal_inference_cosine_weighted.vsv_lambda = 0.05
    
    hal_inference_cosine_weighted.initialize_model()
    
    # Get objects
    processor = hal_inference_cosine_weighted.processor
    model = hal_inference_cosine_weighted.model
    
    import librosa
    import torch
    audio_path = './understanding_sound_data/audio/Y11SEBDuoqSk.wav'
    prompt_text = "Is there a dog barking in this audio?"
    
    try:
        # Prepare inputs (same way as both implementations)
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

        inputs = processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        print(f"Input keys before device move: {list(inputs.keys())}")
        
        # Move inputs to device with proper dtype handling
        inputs_moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs_moved[key] = value.to(model.device)
                # input_ids and attention_mask should stay as Long/Int, not convert to model dtype
                if key in ['input_ids', 'attention_mask']:
                    inputs_moved[key] = inputs_moved[key].long()  # Ensure Long dtype
                # Audio features can use model dtype
                elif key in ['input_features', 'feature_attention_mask', 'audio_values']:
                    inputs_moved[key] = inputs_moved[key].to(model.dtype)
            else:
                inputs_moved[key] = value
        inputs = inputs_moved
        
        print(f"Input keys after device move: {list(inputs.keys())}")
        print(f"Input_ids shape: {inputs['input_ids'].shape}")
        
        # Test 1: Original model.generate()
        print("\n--- Test 1: Original model.generate() ---")
        with torch.no_grad():
            original_output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)
            
        original_response_tokens = original_output[:, inputs['input_ids'].shape[1]:]
        original_response = processor.batch_decode(original_response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"Original tokens: {original_response_tokens[0].tolist()}")
        print(f"Original response: {repr(original_response)}")
        print(f"Original response (display): {original_response}")
        
        # Test 2: Custom generation with gamma=0 (should be identical)
        print("\n--- Test 2: Custom Generation (gamma=0) ---")
        
        # Compute VSV (even though gamma=0, we still need it for the function)
        vsv = compute_vsv_for_audio(audio_path, prompt_text)
        
        with torch.no_grad():
            custom_output = generate_with_cosine_weighting(model, inputs, vsv, gamma=0.0, w=3, max_new_tokens=10)
            
        custom_response_tokens = custom_output[:, inputs['input_ids'].shape[1]:]
        custom_response = processor.batch_decode(custom_response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"Custom tokens: {custom_response_tokens[0].tolist()}")
        print(f"Custom response: {repr(custom_response)}")
        print(f"Custom response (display): {custom_response}")
        
        # Compare
        print(f"\n--- Comparison ---")
        print(f"Same tokens: {torch.equal(original_response_tokens, custom_response_tokens)}")
        print(f"Same response: {original_response == custom_response}")
        
        if not torch.equal(original_response_tokens, custom_response_tokens):
            print("TOKEN DIFFERENCES:")
            orig_tokens = original_response_tokens[0].tolist()
            custom_tokens = custom_response_tokens[0].tolist()
            max_len = max(len(orig_tokens), len(custom_tokens))
            
            for i in range(max_len):
                orig_token = orig_tokens[i] if i < len(orig_tokens) else "N/A"
                custom_token = custom_tokens[i] if i < len(custom_tokens) else "N/A"
                
                if orig_token != custom_token:
                    orig_text = processor.decode([orig_token], skip_special_tokens=False) if orig_token != "N/A" else "N/A"
                    custom_text = processor.decode([custom_token], skip_special_tokens=False) if custom_token != "N/A" else "N/A"
                    print(f"  Position {i}: Original={orig_token}({repr(orig_text)}) vs Custom={custom_token}({repr(custom_text)})")
        
        # Test 3: Run multiple times to check randomness
        print(f"\n--- Test 3: Multiple Runs (Check Randomness) ---")
        print("Original model.generate() - 3 runs:")
        for run in range(3):
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)
            response_tokens = output[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f"  Run {run+1}: {repr(response)}")
        
        print("Custom generation (gamma=0) - 3 runs:")
        for run in range(3):
            with torch.no_grad():
                output = generate_with_cosine_weighting(model, inputs, vsv, gamma=0.0, w=3, max_new_tokens=10)
            response_tokens = output[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f"  Run {run+1}: {repr(response)}")
            
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_generation_methods()