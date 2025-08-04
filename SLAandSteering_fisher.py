import os
import sys
import torch
import librosa
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoProcessor
from Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig

# --- add path to the two helper files you saved earlier ---
# Make sure steering_vector_audio.py and llm_layers_audio.py are in this directory.
sys.path.insert(0, os.path.abspath("."))  # or the folder where the files are located

from steering_vector_fisher import obtain_vsv
from llm_layer_fisher import add_vsv_layers, remove_vsv_layers

# ---------------------
# Configuration
# ---------------------
MAX_SAMPLE = -1
MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/data"

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

def run_evaluation(gamma, w, lam, fisher_scale, model_suffix):
    print(f"\n=== Running evaluation with gamma={gamma}, w={w}, lam={lam}, fisher_scale={fisher_scale} ===")
    
    # Configure SLA
    model.enable_sla(gamma=gamma, w=w)
    
    # Prepare VSV using a sample from Animal dataset
    wav_path = "/home/andrew99245/SAKURA_Reasoning/data/Animal/audio/rooster39.wav"
    audio, _ = librosa.load(wav_path, sr=16000)
    
    sample_prompt = "What's the animal in the audio?"
    messages_pos = build_messages(include_audio=True,  wav_path=wav_path, prompt=sample_prompt)
    messages_neg = build_messages(include_audio=False, wav_path=wav_path, prompt=sample_prompt)
    
    pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, audio=None,  sr=16000)
    
    # Compute VSV and inject
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    # Inject VSV
    add_vsv_layers(model, vsv=vsv, lam=lam, fisher_scale=fisher_scale, which_stack="decoder")
    
    # Run evaluation on all datasets
    os.makedirs("./results", exist_ok=True)
    
    for subset in ["Animal", "Emotion", "Gender", "Language"]:
        print(f"Processing {subset}...")
        df = pd.read_csv(f"{SAKURA_DATA_DIR}/{subset}/metadata.csv")
        
        # Initialize result objects
        single_result_path = f"./results/{subset}_qwen2_audio_7b_{model_suffix}_single.json"
        single_result = {
            "attribute": subset,
            "type": "single",
            "model_config": {"gamma": gamma, "w": w, "lam": lam, "fisher_scale": fisher_scale},
            "results": {}
        }
        multi_result_path = f"./results/{subset}_qwen2_audio_7b_{model_suffix}_multi.json"
        multi_result = {
            "attribute": subset,
            "type": "multi",
            "model_config": {"gamma": gamma, "w": w, "lam": lam, "fisher_scale": fisher_scale},
            "results": {}
        }
        
        max_sample = len(df) if MAX_SAMPLE == -1 else MAX_SAMPLE
        
        for i in tqdm(range(max_sample), desc=f"{subset}"):
            audio_file = df.iloc[i]["file"]
            single_instruction = df.iloc[i]["single_instruction"]
            multi_instruction = df.iloc[i]["multi_instruction"]
            
            audio_path = f"{SAKURA_DATA_DIR}/{subset}/audio/{audio_file}"
            
            # Single instruction
            response = inference(audio_path, prompt=single_instruction)[0]
            print(f"Single - {audio_file}: {response}")
            single_result["results"][audio_file] = {
                "instruction": single_instruction,
                "response": response,
                "label": df.iloc[i]["attribute_label"]
            }
            
            # Multi instruction
            response = inference(audio_path, prompt=multi_instruction)[0]
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
    
    # Remove VSV hooks for next evaluation
    remove_vsv_layers(model, which_stack="decoder")

if __name__ == "__main__":
    # Run evaluation 1: SLA + VSV
    run_evaluation(gamma=0.25, w=4, lam=0.05, fisher_scale=0.5, model_suffix="sla_vsv")

    # Run evaluation 2: Original model (no SLA, but with VSV)
    run_evaluation(gamma=0.0, w=4, lam=0.0, fisher_scale=0.0, model_suffix="original")
