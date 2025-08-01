from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, GenerationConfig
import librosa
import pandas as pd
import os
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List
import torch.nn.functional as F
MAX_SAMPLE = -1
MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
model_name_for_file = "qwen2_audio_7b"
SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/data"

@dataclass
class SLAConfig:
    gamma: float = 0.0        # weight on early-layer logits
    w: int = 5                  # how many late layers to average for augmentation
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True      # nucleus sampling if True, greedy otherwise
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    repetition_penalty: float = 1.0  # 1.0 = disabled

# ---------- Utilities ----------

def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9, min_tokens_to_keep: int = 1):
    """Nucleus (top-p) filtering. Works on a single-step logits vector."""
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative probability above the threshold
    cutoff = cumulative_probs > top_p
    # Keep at least min_tokens_to_keep
    cutoff[..., :min_tokens_to_keep] = False

    sorted_logits[cutoff] = float('-inf')
    # Re-scatter
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float):
    if penalty == 1.0:
        return logits
    # penalize previously generated tokens
    for token_id in set(generated_ids.tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits

# ---------- Wrapper ----------

class SelfLogitsAugmentedCausalLM(torch.nn.Module):
    """
    Generic SLA wrapper. It assumes:
      - model returns hidden_states if `output_hidden_states=True`
      - model exposes `lm_head` (as most decoder-only HF models do)
    """
    def __init__(self, model, sla_cfg: SLAConfig):
        super().__init__()
        self.model = model
        self.sla_cfg = sla_cfg

        # Try to locate lm_head automatically
        self.lm_head = getattr(self.model, "lm_head", None)
        if self.lm_head is None:
            self.lm_head = self.model.get_output_embeddings()
    @torch.no_grad()
    def generate_with_sla(
        self,
        **model_inputs,
    ):
        device = next(self.model.parameters()).device
        sla = self.sla_cfg

        # We rely on model generate loop to be manual so we can touch hidden states
        # Prepare ids / kv cache
        # Qwen2-Audio processor returns `input_ids`, `attention_mask`, and possibly `audio_values`
        # We'll pull them from model_inputs
        input_ids = model_inputs.get("input_ids").to(device)
        attention_mask = model_inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # any extra multimodal fields (audio, etc.)
        extra = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in model_inputs.items()
                 if k not in ["input_ids", "attention_mask"]}

        eos_token_id = sla.eos_token_id
        pad_token_id = sla.pad_token_id or eos_token_id

        past_key_values = None
        generated = []
        first_step = True
        for _ in range(sla.max_new_tokens):
            kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            if first_step:
                kwargs.update(extra)   # only once
                first_step = False

            out = self.model(**kwargs)

            logits = out.logits[:, -1, :]  # [B, V]
            hidden_states: List[torch.Tensor] = out.hidden_states  # length = n_layers+1 (emb + each layer)
            past_key_values = out.past_key_values

            # ---- Self-Logits Augmentation ----
            # hidden_states shape: (layer_idx: 0..L) each is [B, T, H]
            # We'll consider last w transformer layers (ignore embedding layer at idx 0).
            # Many HF models: hidden_states[1] = after layer 0, ..., hidden_states[-1] = after last layer
            # We'll take the *last token* from each hidden state.
            L = len(hidden_states) - 1  # number of transformer layers
            w = min(sla.w, L - 1)       # safe guard
            if w > 0 and sla.gamma > 0:
                hs_stack = []
                for li in range(L - w, L):  # e.g. [L-w, ..., L-1]
                    # NOTE: Qwen2-Audio returns hidden_states as tuple
                    hs_li = hidden_states[li + 1][:, -1, :]  # +1 because hidden_states[0] is embeddings
                    hs_stack.append(hs_li)
                hs_cat = torch.stack(hs_stack, dim=0)  # [w, B, H]

                # Map each to logits through lm_head and average
                aug_logits_each = self.lm_head(hs_cat)   # [w, B, V]
                aug_logits = aug_logits_each.mean(dim=0)       # [B, V]

                # Blend
                logits = (1.0 - sla.gamma) * logits + sla.gamma * aug_logits

            # repetition penalty
            if len(generated) > 0:
                logits = apply_repetition_penalty(logits, torch.tensor(generated, device=device), sla.repetition_penalty)

            # temperature
            if sla.temperature != 1.0:
                logits = logits / sla.temperature

            # sampling / greedy
            if sla.do_sample:
                logits = top_p_filtering(logits, top_p=sla.top_p)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated.append(token_id)

            # stop?
            if eos_token_id is not None and token_id == eos_token_id:
                break

            # Feed next token
            input_ids = next_token  # only the newly generated one
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )

        return torch.tensor(generated, device=device).unsqueeze(0)

def inference(audio_path, prompt):

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": 'How many animals are in the audio? what are they?'},
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
    #print(audios,text)
    # print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(text=text, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    #output = sla_model.generate_with_sla(**inputs)
    output_ids = sla_model.generate_with_sla(**inputs)
    text = processor.batch_decode(output_ids, 
         skip_special_tokens=True,
         clean_up_tokenization_spaces=False
    )
    return text
    #text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #print("Response:", text)
    #return text

if __name__ == "__main__":
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    #os.makedirs("./results", exist_ok=True)

    sla_cfg = SLAConfig(
        gamma=0.0,
        w=5,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=model.generation_config.pad_token_id
    )
    sla_model = SelfLogitsAugmentedCausalLM(model, sla_cfg)
    audio_path = '/home/andrew99245/SAKURA_Reasoning/data/Animal/audio/pig28.wav'
    audio = librosa.load(audio_path, sr=16000)[0]
    for i in range(20):
        response = inference(audio_path, prompt='')[0]
        #print(audio_path)
        print(response)
