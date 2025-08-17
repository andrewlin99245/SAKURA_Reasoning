#!/usr/bin/env python3
"""Debug script to test SLA state impact on audio input handling"""

import sys
import os
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src')

def test_sla_states():
    print("=== Testing SLA State Impact ===")
    
    # Import cosine module
    import hal_inference_cosine_weighted
    
    hal_inference_cosine_weighted.initialize_model()
    
    # Get the initialized objects
    processor = hal_inference_cosine_weighted.processor
    model = hal_inference_cosine_weighted.model
    
    # Prepare test inputs
    import librosa
    import torch
    audio_path = './understanding_sound_data/audio/Y11SEBDuoqSk.wav'
    
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
    
    print("Initial model state:")
    print(f"  sla_enable: {getattr(model, 'sla_enable', 'Not found')}")
    print(f"  sla_gamma: {getattr(model, 'sla_gamma', 'Not found')}")
    print(f"  sla_w: {getattr(model, 'sla_w', 'Not found')}")
    
    # Test 1: Current state (should fail based on previous tests)
    print("\n--- Test 1: Current State ---")
    with torch.no_grad():
        try:
            output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
            print("✓ Current state: SUCCESS")
        except Exception as e:
            if "model_kwargs" in str(e):
                print(f"✗ Current state: model_kwargs warning")
            else:
                print(f"✗ Current state: Other error - {e}")
    
    # Test 2: Enable SLA with gamma=0
    print("\n--- Test 2: Enable SLA (gamma=0) ---")
    model.enable_sla(gamma=0.0, w=4)
    print(f"  sla_enable: {getattr(model, 'sla_enable', 'Not found')}")
    print(f"  sla_gamma: {getattr(model, 'sla_gamma', 'Not found')}")
    
    with torch.no_grad():
        try:
            output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
            print("✓ SLA enabled: SUCCESS")
        except Exception as e:
            if "model_kwargs" in str(e):
                print(f"✗ SLA enabled: model_kwargs warning")
            else:
                print(f"✗ SLA enabled: Other error - {e}")
    
    # Test 3: Disable SLA
    print("\n--- Test 3: Disable SLA ---")
    model.disable_sla()
    print(f"  sla_enable: {getattr(model, 'sla_enable', 'Not found')}")
    
    with torch.no_grad():
        try:
            output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
            print("✓ SLA disabled: SUCCESS")
        except Exception as e:
            if "model_kwargs" in str(e):
                print(f"✗ SLA disabled: model_kwargs warning")
            else:
                print(f"✗ SLA disabled: Other error - {e}")
    
    # Test 4: Enable SLA with gamma=0.5
    print("\n--- Test 4: Enable SLA (gamma=0.5) ---")
    model.enable_sla(gamma=0.5, w=4)
    print(f"  sla_enable: {getattr(model, 'sla_enable', 'Not found')}")
    print(f"  sla_gamma: {getattr(model, 'sla_gamma', 'Not found')}")
    
    with torch.no_grad():
        try:
            output = model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=1.0, top_p=0.9)
            print("✓ SLA gamma=0.5: SUCCESS")
        except Exception as e:
            if "model_kwargs" in str(e):
                print(f"✗ SLA gamma=0.5: model_kwargs warning")
            else:
                print(f"✗ SLA gamma=0.5: Other error - {e}")

if __name__ == "__main__":
    test_sla_states()