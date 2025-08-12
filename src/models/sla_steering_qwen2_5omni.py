import os
import sys
import torch
import librosa
import pandas as pd
import json
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor
# Using relative imports instead of sys.path manipulation

from ..utils.cache_config import set_hf_cache_env
from ..utils.qwen_omni_utils import process_mm_info

# Configure shared cache
set_hf_cache_env()
from ..utils.Qwen2_5Omni_patch import Qwen2_5OmniSLAForCausalLM
from .steering_vector_qwen2_5omni import obtain_vsv
from .llm_layer_qwen2_5omni import add_vsv_layers, remove_vsv_layers

# ---------------------
# Configuration
# ---------------------
MAX_SAMPLE = -1
MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/data"

# ---------------------
# Load model + SLA on
# ---------------------
model = Qwen2_5OmniSLAForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

# Enable SLA (γ=0.3, last 5 layers). Your SLA implementation runs inside forward/generate.
model.enable_sla(gamma=0.0, w=4)

# ---------------------
# Helpers to build inputs
# ---------------------
def build_messages(include_audio: bool, audio_path: str, prompt: str):
    base = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
    ]
    if include_audio:
        base.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        base.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # No audio content here (this is the 'neg' case)
            ],
        })
    return base

def build_inputs(messages, audio=None):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if audio is None:
        # For negative case - no multimodal inputs
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True
        )
    else:
        # For positive case - use the provided audio data
        inputs = processor(
            text=prompt,
            audio=[audio],  # Use the actual audio data passed as parameter
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
    
    # Move tensors to model device
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs

def inference_with_adaptive_vsv(audio_path, prompt, lam):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Build positive and negative inputs for VSV computation
    vsv_prompt = "Describe the audio in detail."
    messages_pos = build_messages(include_audio=True,  audio_path=audio_path, prompt=vsv_prompt)
    messages_neg = build_messages(include_audio=False, audio_path=audio_path, prompt=vsv_prompt)
    
    pos_inputs = build_inputs(messages_pos, audio=audio)
    neg_inputs = build_inputs(messages_neg, audio=None)
    
    # Compute VSV specific to this input
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    # Inject VSV for this specific input
    add_vsv_layers(model, vsv=vsv, lam=lam, which_stack="decoder")
    
    # Perform inference
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Use the audio data that was already loaded at the beginning of the function
    inputs = processor(text=text, audio=[audio], return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=512, thinker_do_sample=False)
    output = output[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Remove VSV hooks after inference
    remove_vsv_layers(model, which_stack="decoder")
    
    return response

def inference(audio_path, prompt):
    # Load audio data
    audio, sr = librosa.load(audio_path, sr=16000)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Use actual audio data instead of processing paths from messages
    inputs = processor(text=text, audio=[audio], return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=512, thinker_do_sample=False)
    output = output[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

def run_evaluation(gamma, w, lam, model_suffix):
    print(f"\n=== Running evaluation with gamma={gamma}, w={w}, lam={lam} ===")
    
    # Configure SLA
    model.enable_sla(gamma=gamma, w=w)
    
    # Run evaluation on all datasets
    os.makedirs("./results", exist_ok=True)
    
    for subset in ["Animal", "Emotion", "Gender", "Language"]:
        print(f"Processing {subset}...")
        df = pd.read_csv(f"{SAKURA_DATA_DIR}/{subset}/metadata.csv")
        
        # Initialize result objects
        single_result_path = f"./results/{subset}_qwen2_5omni_3b_{model_suffix}_single.json"
        single_result = {
            "attribute": subset,
            "type": "single",
            "model_config": {"gamma": gamma, "w": w, "lam": lam},
            "results": {}
        }
        multi_result_path = f"./results/{subset}_qwen2_5omni_3b_{model_suffix}_multi.json"
        multi_result = {
            "attribute": subset,
            "type": "multi",
            "model_config": {"gamma": gamma, "w": w, "lam": lam},
            "results": {}
        }
        
        max_sample = len(df) if MAX_SAMPLE == -1 else MAX_SAMPLE
        
        for i in tqdm(range(max_sample), desc=f"{subset}"):
            audio_file = df.iloc[i]["file"]
            single_instruction = df.iloc[i]["single_instruction"]
            multi_instruction = df.iloc[i]["multi_instruction"]
            
            audio_path = f"{SAKURA_DATA_DIR}/{subset}/audio/{audio_file}"
            
            # Single instruction
            response = inference_with_adaptive_vsv(audio_path, prompt=single_instruction, lam=lam)[0]
            print(f"Single - {audio_file}: {response}")
            single_result["results"][audio_file] = {
                "instruction": single_instruction,
                "response": response,
                "label": df.iloc[i]["attribute_label"]
            }
            
            # Multi instruction
            response = inference_with_adaptive_vsv(audio_path, prompt=multi_instruction, lam=lam)[0]
            print(f"Multi - {audio_file}: {response}")
            multi_result["results"][audio_file] = {
                "instruction": multi_instruction,
                "response": response,
                "label": df.iloc[i]["attribute_label"]
            }
        
        # Save results
        with open(single_result_path, "w") as f:
            json.dump(single_result, f, indent=4, ensure_ascii=False)
        
        with open(multi_result_path, "w") as f:
            json.dump(multi_result, f, indent=4, ensure_ascii=False)
        
        print(f"Finished {subset}.")

if __name__ == "__main__":
    # Run evaluation 1: SLA + VSV
    run_evaluation(gamma=0.25, w=4, lam=0.05, model_suffix="L2norm")
    # Run evaluation 2: Original model (no SLA, but with VSV)
    #run_evaluation(gamma=0.0, w=4, lam=0.0, model_suffix="original")