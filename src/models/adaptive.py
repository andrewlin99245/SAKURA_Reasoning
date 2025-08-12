import os
import sys
import torch
import librosa
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoProcessor
# Add path to utils and layers
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, "src", "utils"))
sys.path.insert(0, os.path.join(project_root, "src", "models"))
sys.path.insert(0, os.path.join(project_root, "src", "layers"))
sys.path.insert(0, os.path.join(project_root, "src", "layers", "variants"))

from Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig
from steering_vector import obtain_vsv
from llm_layer_adaptive import add_vsv_layers_adaptive, remove_vsv_layers
from cache_config import set_hf_cache_env

# Configure shared cache
set_hf_cache_env()

# ---------------------
# Configuration
# ---------------------
MAX_SAMPLE = -1
MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/data"
BASE_LAMBDA = 0.0275

# ---------------------
# Adaptive Lambda and Norm Functions
# ---------------------
def get_adaptive_lambda(layer_idx):
    """
    Get adaptive lambda value based on layer index.
    
    Layer Range    λ Range           Trend
    -----------------------------------------------
    0-3           0.008-0.015       Very low (30% of base), gentle steering for acoustic features
    4-10          0.023-0.028       Gradually increasing toward baseline
    11-21         0.025-0.030       Stable around baseline (semantic processing)
    22-24         0.032-0.037       Rising (transition begins)
    25-27         0.035-0.042       Peak values (audio→language transition)
    28-30         0.038-0.048       High but variable (depends on variance)
    31            0.040-0.044       Elevated final layer
    """
    if 0 <= layer_idx <= 3:
        # Very low (30% of base), gentle steering for acoustic features
        return BASE_LAMBDA * 0.3 + (layer_idx / 3) * (BASE_LAMBDA * 0.55 - BASE_LAMBDA * 0.3)
    elif 4 <= layer_idx <= 10:
        # Gradually increasing toward baseline
        progress = (layer_idx - 4) / (10 - 4)
        return 0.023 + progress * (0.028 - 0.023)
    elif 11 <= layer_idx <= 21:
        # Stable around baseline (semantic processing)
        progress = (layer_idx - 11) / (21 - 11)
        return 0.025 + progress * (0.030 - 0.025)
    elif 22 <= layer_idx <= 24:
        # Rising (transition begins)
        progress = (layer_idx - 22) / (24 - 22)
        return 0.032 + progress * (0.037 - 0.032)
    elif 25 <= layer_idx <= 27:
        # Peak values (audio→language transition)
        progress = (layer_idx - 25) / (27 - 25)
        return 0.035 + progress * (0.042 - 0.035)
    elif 28 <= layer_idx <= 30:
        # High but variable (depends on variance)
        progress = (layer_idx - 28) / (30 - 28)
        return 0.038 + progress * (0.048 - 0.038)
    elif layer_idx == 31:
        # Elevated final layer
        return 0.042
    else:
        # Fallback to base lambda for any unexpected layers
        return BASE_LAMBDA

def get_adaptive_norm_scale(layer_idx):
    """
    Get adaptive norm scaling based on layer index.
    
    Layer Range    Norm Scale    Effect
    -----------------------------------------------
    0-3           1.00          Preserve acoustic energy
    4-23          1.00          Maintain original norm
    24-27         1.05          5% boost during critical transition
    28-29         1.00          Return to baseline
    30-31         0.97          3% reduction to prevent overconfidence
    """
    if 0 <= layer_idx <= 3:
        return 1.00  # Preserve acoustic energy
    elif 4 <= layer_idx <= 23:
        return 1.00  # Maintain original norm
    elif 24 <= layer_idx <= 27:
        return 1.05  # 5% boost during critical transition
    elif 28 <= layer_idx <= 29:
        return 1.00  # Return to baseline
    elif 30 <= layer_idx <= 31:
        return 0.97  # 3% reduction to prevent overconfidence
    else:
        return 1.00  # Default to no scaling

# ---------------------
# Load model + SLA on
# ---------------------
config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
model = Qwen2AudioSLAForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Enable SLA (γ=0.3, last 5 layers). Your SLA implementation runs inside forward/generate.
model.enable_sla(gamma=0.0, w=4)

# ---------------------
# Helpers to build inputs
# ---------------------
def build_messages(include_audio: bool, wav_path: str, prompt: str):
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    if include_audio:
        base.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": wav_path},
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

def build_inputs(messages, audio=None, sr=16000):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if audio is None:
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=prompt,
            audios=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
    # Move tensors to model device
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs

def inference_with_adaptive_vsv(audio_path, prompt):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Build positive and negative inputs for VSV computation
    vsv_prompt = "Describe the audio in detail."
    messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    messages_neg = build_messages(include_audio=False, wav_path=audio_path, prompt=vsv_prompt)
    
    pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, audio=None,  sr=16000)
    
    # Compute VSV specific to this input
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    # Inject VSV with adaptive lambda and norm scaling
    add_vsv_layers_adaptive(model, vsv=vsv, which_stack="decoder")
    
    # Perform inference
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audios=[audio], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, max_new_tokens=512)
    output = output[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Remove VSV hooks after inference
    remove_vsv_layers(model, which_stack="decoder")
    
    return response

def inference(audio_path, prompt):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
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

    output = model.generate(**inputs, max_new_tokens=512)
    output = output[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

def run_evaluation(gamma, w, model_suffix):
    print(f"\n=== Running evaluation with adaptive lambda and norm, gamma={gamma}, w={w} ===")
    
    # Configure SLA
    model.enable_sla(gamma=gamma, w=w)
    
    # Run evaluation on all datasets
    os.makedirs("./results", exist_ok=True)
    
    for subset in ["Animal", "Emotion", "Gender", "Language"]:
        single_result_path = f"./results/{subset}_qwen2_audio_7b_{model_suffix}_single.json"
        multi_result_path = f"./results/{subset}_qwen2_audio_7b_{model_suffix}_multi.json"
        
        # Check if results already exist
        if os.path.exists(single_result_path) and os.path.exists(multi_result_path):
            print(f"Skipping {subset} - results already exist at:")
            print(f"  - {single_result_path}")
            print(f"  - {multi_result_path}")
            continue
        
        print(f"Processing {subset}...")
        df = pd.read_csv(f"{SAKURA_DATA_DIR}/{subset}/metadata.csv")
        
        # Load existing results if they exist, or initialize new ones
        if os.path.exists(single_result_path):
            with open(single_result_path, 'r') as f:
                single_result = json.load(f)
            print(f"Loaded existing single results for {subset}")
        else:
            single_result = {
                "attribute": subset,
                "type": "single",
                "model_config": {"gamma": gamma, "w": w, "base_lambda": BASE_LAMBDA, "adaptive": True},
                "results": {}
            }
            
        if os.path.exists(multi_result_path):
            with open(multi_result_path, 'r') as f:
                multi_result = json.load(f)
            print(f"Loaded existing multi results for {subset}")
        else:
            multi_result = {
                "attribute": subset,
                "type": "multi",
                "model_config": {"gamma": gamma, "w": w, "base_lambda": BASE_LAMBDA, "adaptive": True},
                "results": {}
            }
        
        max_sample = len(df) if MAX_SAMPLE == -1 else MAX_SAMPLE
        
        for i in tqdm(range(max_sample), desc=f"{subset}"):
            audio_file = df.iloc[i]["file"]
            single_instruction = df.iloc[i]["single_instruction"]
            multi_instruction = df.iloc[i]["multi_instruction"]
            
            audio_path = f"{SAKURA_DATA_DIR}/{subset}/audio/{audio_file}"
            
            # Check if this audio file has already been processed
            skip_single = audio_file in single_result["results"]
            skip_multi = audio_file in multi_result["results"]
            
            if skip_single and skip_multi:
                print(f"Skipping {audio_file} - already processed")
                continue
            
            # Single instruction
            if not skip_single:
                response = inference_with_adaptive_vsv(audio_path, prompt=single_instruction)[0]
                print(f"Single - {audio_file}: {response}")
                single_result["results"][audio_file] = {
                    "instruction": single_instruction,
                    "response": response,
                    "label": df.iloc[i]["attribute_label"]
                }
            else:
                print(f"Single - {audio_file}: already processed")
            
            # Multi instruction
            if not skip_multi:
                response = inference_with_adaptive_vsv(audio_path, prompt=multi_instruction)[0]
                print(f"Multi - {audio_file}: {response}")
                multi_result["results"][audio_file] = {
                    "instruction": multi_instruction,
                    "response": response,
                    "label": df.iloc[i]["attribute_label"]
                }
            else:
                print(f"Multi - {audio_file}: already processed")
        
        # Save results
        with open(single_result_path, "w") as f:
            json.dump(single_result, f, indent=4, ensure_ascii=False)
        
        with open(multi_result_path, "w") as f:
            json.dump(multi_result, f, indent=4, ensure_ascii=False)
        
        print(f"Finished {subset}.")

if __name__ == "__main__":
    # Run evaluation with adaptive lambda and norm scaling
    run_evaluation(gamma=0.25, w=4, model_suffix="adaptive")