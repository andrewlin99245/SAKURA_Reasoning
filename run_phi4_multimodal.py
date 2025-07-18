from qwen_omni_utils import process_mm_info
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import librosa
import pandas as pd
import os
import json
from tqdm import tqdm

MAX_SAMPLE = -1
MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"
model_name_for_file = "phi4_multimodal"
SAKURA_DATA_DIR = "/home/anthony/SAKURA/data"

def inference(audio_path, prompt):

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    speech_prompt = prompt
    prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

    audio, sr = librosa.load(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    inputs = processor(text=prompt, audios=[(audio, 16000)], return_tensors='pt').to('cuda')

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        generation_config=generation_config,
    )

    output = generate_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype='bfloat16',
        _attn_implementation="eager"
    ).cuda()
    model.eval()

    generation_config = GenerationConfig.from_pretrained(MODEL_PATH, 'generation_config.json')
    generation_config.max_new_tokens = 512

    os.makedirs("./results", exist_ok=True)

    for subset in ["Animal", "Emotion", "Gender", "Language"]:
        df = pd.read_csv(f"{SAKURA_DATA_DIR}/{subset}/metadata_tw.csv")
        df.head()

        # Initialize the result saving path & object
        single_result_path = f"./results/{subset}_{model_name_for_file}_single.json"
        single_result = {
            "attribute": subset,
            "type": "single",
            "results": {}
        }
        multi_result_path = f"./results/{subset}_{model_name_for_file}_multi.json"
        multi_result = {
            "attribute": subset,
            "type": "multi",
            "results": {}
        }

        if MAX_SAMPLE == -1:
            max_sample = len(df)
        else:
            max_sample = MAX_SAMPLE

        for i in tqdm(range(max_sample)):
            audio_file = df.iloc[i]["file"]
            single_instruction = df.iloc[i]["single_instruction"]
            multi_instruction = df.iloc[i]["multi_instruction"]

            audio_path = f"{SAKURA_DATA_DIR}/{subset}/audio/{audio_file}"
            audio = librosa.load(audio_path, sr=16000)[0]
            
            response = inference(audio_path, prompt=single_instruction)[0]
            single_result["results"][audio_file] = {
                "instruction": single_instruction,
                "response": response,
                "label": df.iloc[i]["attribute_label"]
            }

            response = inference(audio_path, prompt=multi_instruction)[0]
            multi_result["results"][audio_file] = {
                "instruction": multi_instruction,
                "response": response,
                "label": df.iloc[i]["attribute_label"]
            }

        with open(single_result_path, "w") as f:
            json.dump(single_result, f, indent=4, ensure_ascii=False)

        with open(multi_result_path, "w") as f:
            json.dump(multi_result, f, indent=4, ensure_ascii=False)
        
        print(f"Finished {subset}.")
