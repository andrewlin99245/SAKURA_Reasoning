import torch
from transformers import AutoProcessor
from Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig
import librosa
from steering_vector import obtain_vsv
from llm_layers import add_vsv_layers, remove_vsv_layers

model_name = "Qwen/Qwen2-Audio-7B-Instruct"
config = Qwen2AudioConfig.from_pretrained(model_name)
model = Qwen2AudioSLAForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()


processor = AutoProcessor.from_pretrained(model_name)

# Enable SLA (γ=0.3, last 5 layers averaged); set γ=0.0 to disable (vanilla)
model.enable_sla(gamma=0.3,w=5)

# Build ChatML prompt with exactly one <|AUDIO|>
messages = [
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":[
        {"type":"text","text":"How many animals are in the audio? What are they?"},
        {"type":"audio","audio_url":'/home/andrew99245/SAKURA_Reasoning/data/Animal/audio/pig28.wav'},
    ]},
]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audio, _ = librosa.load("/home/andrew99245/SAKURA_Reasoning/data/Animal/audio/pig28.wav", sr=16000)

inputs = processor(
    text=prompt,
    audios=[audio],
    sampling_rate=16000,
    return_tensors="pt",
    padding=True,
)
inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k,v in inputs.items()}

# Standard HF generate still works (SLA runs inside forward)
for i in range(10):
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            #top_p=0.9,
            #temperature=1.0
        )

    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    print(text)