import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"

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

    output = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=1, top_p=0.9)

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

    # Single audio file and prompt from SLAandSteering.py
    audio_path = "/home/andrew99245/SAKURA_Reasoning/data/Animal/audio/rooster39.wav"
    prompt = "Is there a sound of a dog barking in the audio? "
    
    response = inference(audio_path, prompt)[0]
    print(f"Audio: rooster39.wav")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")