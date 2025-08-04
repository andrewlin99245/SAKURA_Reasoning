import os
import sys
import torch
import librosa
from transformers import AutoProcessor
from Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig

# --- add path to the two helper files you saved earlier ---
# Make sure steering_vector_audio.py and llm_layers_audio.py are in this directory.
sys.path.insert(0, os.path.abspath("."))  # or the folder where the files are located

from steering_vector import obtain_vsv
from llm_layer import add_vsv_layers, remove_vsv_layers

# ---------------------
# Load model + SLA on
# ---------------------
model_name = "Qwen/Qwen2-Audio-7B-Instruct"
config = Qwen2AudioConfig.from_pretrained(model_name)
model = Qwen2AudioSLAForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(model_name)

# Enable SLA (γ=0.3, last 5 layers). Your SLA implementation runs inside forward/generate.
model.enable_sla(gamma=0.0, w=4)

# ---------------------
# Helpers to build inputs
# ---------------------
def build_messages(include_audio: bool, wav_path: str):
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    if include_audio:
        base.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": wav_path},
                {"type": "text", "text": "Is there a sound of a dog barking in the audio? "},
            ],
        })
    else:
        base.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Is there a sound of a dog barking in the audio? "},
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

# ---------------------
# Prepare one (neg,pos) pair for VSV
# ---------------------
wav_path = "/home/andrew99245/SAKURA_Reasoning/data/Animal/audio/rooster39.wav"
audio, _ = librosa.load(wav_path, sr=16000)

messages_pos = build_messages(include_audio=True,  wav_path=wav_path)
messages_neg = build_messages(include_audio=False, wav_path=wav_path)

pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
neg_inputs = build_inputs(messages_neg, audio=None,  sr=16000)

# ---------------------
# Compute VSV and inject
# ---------------------
with torch.no_grad():
    # kwargs_list expects a list of [neg_kwargs, pos_kwargs] pairs; you can append more pairs to average.
    kwargs_list = [[neg_inputs, pos_inputs]]
    vsv = obtain_vsv(model, kwargs_list)              # [L, d_model], float32 on CPU
    vsv = vsv.to(model.device)                         # ensure same device as model

# Inject VSV at decoder blocks (post-block residual; norm-preserving)
lam = 0.075  # VSV strength - tune between 0.1-0.17 as per VISTA paper
add_vsv_layers(model, vsv=vsv, lam=lam, which_stack="decoder")

# ---------------------
# Generate with BOTH SLA + VSV enabled
# ---------------------
for i in range(10):
    with torch.no_grad():
        out = model.generate(
            **pos_inputs,       # use the positive (with-audio) inputs for decoding
            max_new_tokens=128,
            do_sample=True,
            temperature=1.5,
            top_p=0.9,
        )
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    print(text)

# (Optional) remove steering afterwards
# remove_vsv_layers(model, which_stack="decoder")
