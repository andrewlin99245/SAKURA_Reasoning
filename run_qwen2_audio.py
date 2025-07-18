from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import pandas as pd
import os
import json
from tqdm import tqdm

MAX_SAMPLE = -1
MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
model_name_for_file = "qwen2_audio_7b"
SAKURA_DATA_DIR = "/home/anthony/SAKURA/data"

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
    # print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(text=text, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, max_new_tokens=512)

    output = output[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

if __name__ == "__main__":
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

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
