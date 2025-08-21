import os
import sys

# ---------------- Cache setup ----------------
SHARED_CACHE_DIR = os.path.expanduser("~/.cache/sakura_reasoning")
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = SHARED_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = SHARED_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["HF_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.cache")
os.environ["TORCH_HOME"] = SHARED_CACHE_DIR
if "PYTORCH_CACHE_HOME" in os.environ:
    del os.environ["PYTORCH_CACHE_HOME"]
print(f"Cache configured: {SHARED_CACHE_DIR}")

# ---------------- Imports ----------------
import csv
import argparse
import torch
import librosa
import time
import random
import numpy as np
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from scipy.stats import pearsonr, spearmanr, ttest_ind
import torch.nn.functional as F

# ---------------- Local project imports ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

from utils.Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig

# Vector steering
try:
    from steering_vector import obtain_vsv
    from ..layers.llm_layer import add_vsv_layers, remove_vsv_layers, get_layers
    VSV_AVAILABLE = True
except ImportError as e:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.insert(0, project_dir)
        from src.models.steering_vector import obtain_vsv
        from src.layers.llm_layer import add_vsv_layers, remove_vsv_layers, get_layers
        VSV_AVAILABLE = True
    except ImportError as e2:
        print(f"Warning: Vector steering modules not available: {e2}")
        print("Vector steering will be disabled.")
        VSV_AVAILABLE = False

# ---------------- Globals ----------------
model = None
processor = None
verbose_progress = False
vsv_enabled = False
vsv_lambda = 1.0

# storage for cosines during generation (kept for your existing analyses)
cosine_measurements = []

# ---------------- Hooks (unchanged logic) ----------------
class CosineHookWithSteering:
    def __init__(self, layer_idx: int, steering_vector: torch.Tensor, lam: float):
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector.clone()
        self.lam = lam
    
    def __call__(self, module, input, output):
        global cosine_measurements
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest_output = output[1:]
        else:
            hidden_states = output
            rest_output = ()
        if hidden_states.dim() != 3:
            return output
        
        orig_dtype = hidden_states.dtype
        x = hidden_states.float()
        v = self.steering_vector.to(x.device, x.dtype).view(1,1,-1)
        x_steered = x + self.lam * v

        x_normalized = F.normalize(x_steered, p=2, dim=-1)
        v_normalized = F.normalize(v, p=2, dim=-1)
        per_token_cos_sim = torch.sum(x_normalized * v_normalized, dim=-1)
        # record the first token (you wanted BOS-like here)
        cos_sim = per_token_cos_sim[:,0]
        layer_mean_cosine = cos_sim[0].item()
        cosine_measurements.append({'layer': self.layer_idx, 'cosine_similarity': layer_mean_cosine})
        
        x_steered = x_steered.to(orig_dtype)
        if rest_output:
            return (x_steered,) + rest_output
        else:
            return x_steered

def clear_cosine_storage():
    global cosine_measurements
    cosine_measurements = []

def get_cosine_statistics():
    global cosine_measurements
    if not cosine_measurements:
        return {}
    sims = [m['cosine_similarity'] for m in cosine_measurements]
    return {
        'count': len(sims),
        'mean': np.mean(sims),
        'std': np.std(sims),
        'min': np.min(sims),
        'max': np.max(sims),
        'median': np.median(sims)
    }

def get_layer_by_layer_cosine_statistics():
    global cosine_measurements
    if not cosine_measurements:
        return {}
    layer_stats = {}
    for m in cosine_measurements:
        lid = m['layer']
        layer_stats.setdefault(lid, []).append(m['cosine_similarity'])
    result = {}
    for lid, sims in layer_stats.items():
        result[lid] = {
            'count': len(sims),
            'mean': np.mean(sims),
            'std': np.std(sims),
            'min': np.min(sims),
            'max': np.max(sims),
            'median': np.median(sims),
            'cosine_similarities': sims
        }
    return result

# ---------------- Model init ----------------
def initialize_model():
    global model, processor
    MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
    print("🚀 Initializing model...")
    print("  📦 Loading configuration...")
    config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
    print("  🤖 Loading model...")
    model = Qwen2AudioSLAForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    print("  🔧 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("  ⚡ Enabling SLA...")
    model.enable_sla(gamma=0.0, w=4)
    print("✅ Model initialization complete!")

# ---------------- Chat building ----------------
def build_messages(include_audio: bool, wav_path: str, prompt: str):
    base = []
    if include_audio:
        base.append({
            "role":"user",
            "content":[
                {"type":"audio","audio_url":wav_path},
                {"type":"text","text":prompt},
            ],
        })
    else:
        base.append({
            "role":"user",
            "content":[{"type":"text","text":prompt}],
        })
    return base

def build_inputs(messages, audio=None, sr=16000):
    global processor, model
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if audio is None:
        inputs = processor(text=prompt, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=prompt, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in inputs.items()}
    return inputs

# ---------------- VSV ----------------
def compute_vsv_for_audio(audio_path, prompt):
    global model, processor, verbose_progress
    if verbose_progress: print("    🎯 Computing vector steering vector...")
    audio, sr = librosa.load(audio_path, sr=16000)
    soundless_audio = np.zeros_like(audio)
    vsv_prompt = f"Focus on the given audio and answer the following question. {prompt} Answer just yes or no."
    messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    messages_neg = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, audio=soundless_audio, sr=16000)
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list).to(model.device)
    if verbose_progress: print(f"    ✅ VSV computed with shape: {vsv.shape}")
    return vsv

# ---------------- BOS probe (new) ----------------
def project_to_vocab_from_hidden(model, hidden):  # hidden: [B, d]
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        hidden = model.model.norm(hidden)
    # Handle different model architectures
    if hasattr(model, 'lm_head'):
        return model.lm_head(hidden)  # [B, V]
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model.lm_head(hidden)  # [B, V]
    else:
        raise AttributeError(f"Cannot find lm_head in model of type {type(model)}")

@torch.no_grad()
def probe_bos_features(audio_path, prompt_text, vsv, early_idx=None, late_idx=None):
    """
    Single forward to the end of the prompt (no generation, no steering).
    Returns a dict with:
      - cos_bos_per_layer (list len = n_layers)
      - bos_mean_late, bos_std_late
      - entropy_bos, margin_bos
      - gap_EL  (difference between early vs late BOS readouts)
      - early_idx, late_idx (the indices actually used)
    """
    global model, processor
    # Build inputs
    messages = [
        {"role":"user","content":[
            {"type":"audio","audio_url":audio_path},
            {"type":"text","text":f"Focus on the given audio and answer the following question. {prompt_text} Answer just yes or no."},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios = []
    for m in messages:
        for ele in (m.get("content") or []):
            if ele.get("type") == "audio":
                audio, sr = librosa.load(ele["audio_url"])
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audios.append(audio)
    inputs = processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in inputs.items()}

    # Forward WITHOUT steering and WITHOUT generation
    out = model(**inputs, output_hidden_states=True, use_cache=True)
    last_idx = inputs["input_ids"].shape[1] - 1

    hs_all = out.hidden_states  # tuple of [B, seq, d]
    # Try to keep only transformer layers (drop embeddings if present)
    n_layers_model = len(get_layers(model, "decoder"))
    if len(hs_all) == n_layers_model + 1:
        layer_hiddens = hs_all[1:]
    else:
        layer_hiddens = hs_all
    n_layers = len(layer_hiddens)

    # default early/late split if not provided
    if late_idx is None:
        late_start = int(0.75 * n_layers)
        late_idx = list(range(late_start, n_layers))
    if early_idx is None:
        early_idx = list(range(0, max(1, int(0.25 * n_layers)) ))

    # per-layer BOS cosines (against vsv), λ=0
    cos_bos = []
    vsv = vsv.to(model.device)
    for L in range(n_layers):
        h_bos = layer_hiddens[L][:, last_idx, :]        # [B, d]
        v = vsv[L].unsqueeze(0).to(h_bos)               # [1, d]
        h_n = F.normalize(h_bos, p=2, dim=-1)
        v_n = F.normalize(v,      p=2, dim=-1)
        cos = (h_n * v_n).sum(dim=-1)                   # [B]
        cos_bos.append(float(cos.item()))

    # late stats
    cos_late = [cos_bos[i] for i in late_idx]
    bos_mean_late = float(np.mean(cos_late))
    bos_std_late  = float(np.std(cos_late))

    # BOS logits at final head
    logits_last = out.logits[:, last_idx, :]            # [B, V]
    probs_last  = torch.softmax(logits_last, dim=-1)
    entropy_bos = float(-(probs_last * torch.log(probs_last + 1e-9)).sum(-1).item())
    top2 = torch.topk(logits_last, k=2, dim=-1).values[0].tolist()
    margin_bos = float(top2[0] - top2[1])

    # early vs late layer projections to vocab at BOS, then a distance on probs
    def avg_layer_logits(layer_ids):
        acc = None
        for i in layer_ids:
            li = project_to_vocab_from_hidden(model, layer_hiddens[i][:, last_idx, :])
            acc = li if acc is None else acc + li
        return acc / max(1, len(layer_ids))

    logits_E = avg_layer_logits(early_idx)
    logits_L = avg_layer_logits(late_idx)
    probs_E  = torch.softmax(logits_E, dim=-1)
    probs_L  = torch.softmax(logits_L, dim=-1)
    gap_EL   = float(torch.norm(probs_L - probs_E, p=2).item())   # simple, stable

    return {
        "cos_bos_per_layer": cos_bos,
        "bos_mean_late": bos_mean_late,
        "bos_std_late":  bos_std_late,
        "entropy_bos":   entropy_bos,
        "margin_bos":    margin_bos,
        "gap_EL":        gap_EL,
        "early_idx":     early_idx,
        "late_idx":      late_idx,
        "n_layers":      n_layers,
    }

# Build a fixed BOS feature vector + names (you will train on this)
def extract_bos_feature_vector(bos_feats, use_last_k_layers=8):
    """
    Returns (feature_vector, feature_names, late_indices_used)
    We take aggregates + last-K layer cosines for compactness.
    """
    cos = bos_feats["cos_bos_per_layer"]
    nL  = bos_feats["n_layers"]
    K   = min(use_last_k_layers, nL)
    late_ids_for_vector = list(range(nL - K, nL))

    vec = [
        bos_feats["bos_mean_late"],
        bos_feats["bos_std_late"],
        bos_feats["entropy_bos"],
        bos_feats["margin_bos"],
        bos_feats["gap_EL"],
    ]
    names = ["bos_mean_late","bos_std_late","entropy_bos","margin_bos","gap_EL"]
    for idx in late_ids_for_vector:
        vec.append(cos[idx])
        names.append(f"cos_L{idx}")

    return np.array(vec, dtype=np.float64), names, late_ids_for_vector

# ---------------- Inference with measurement (your original path) ----------------
def inference_with_cosine_measurement(audio_path, prompt_text, vsv_lambda=0.05):
    global model, processor, verbose_progress
    if model is None or processor is None:
        initialize_model()
    clear_cosine_storage()
    if verbose_progress: print(f"  🎵 Processing: {os.path.basename(audio_path)}")

    hooks = []
    try:
        vsv = compute_vsv_for_audio(audio_path, prompt_text)
        if verbose_progress: print("    🎯 Setting up post-steering cosine hooks...")
        layers = get_layers(model, which_stack="decoder")
        for layer_idx, layer in enumerate(layers):
            if layer_idx < len(vsv):
                hook = CosineHookWithSteering(layer_idx, vsv[layer_idx], vsv_lambda)
                hooks.append(layer.register_forward_hook(hook))
        if verbose_progress: print(f"    ✅ Hooks for {len(hooks)} layers")

        modified_prompt = f"Focus on the given audio and answer the following question. {prompt_text} Answer just yes or no."
        messages = [{"role":"user","content":[
            {"type":"audio","audio_url":audio_path},
            {"type":"text","text":modified_prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        audios = []
        for message in messages:
            for ele in (message["content"] if isinstance(message["content"], list) else []):
                if ele["type"]=="audio":
                    audio, sr = librosa.load(ele["audio_url"])
                    if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    audios.append(audio)

        inputs = processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        if verbose_progress: print("    🧠 Generating...")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)
        output = output[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.strip().lower()
        if 'yes' in response: result = "Yes"
        elif 'no' in response: result = "No"
        else: result = "No"

        layer_stats = get_layer_by_layer_cosine_statistics()
        overall_stats = get_cosine_statistics()
        return result, layer_stats, overall_stats, vsv

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "No", {}, {}, None
    finally:
        for h in hooks:
            h.remove()
        if verbose_progress and hooks: print("    🔄 Hooks removed")

# ---------------- Analyses (unchanged) ----------------
def compute_inter_layer_cosine_correlations(evaluation_results):
    if len(evaluation_results) < 2:
        print("❌ Need at least 2 samples for inter-layer correlation analysis")
        return {}
    correct_results = [r for r in evaluation_results if r['correct'] and r.get('layer_stats')]
    incorrect_results = [r for r in evaluation_results if not r['correct'] and r.get('layer_stats')]
    if len(correct_results) < 2 or len(incorrect_results) < 2:
        print("❌ Need at least 2 samples in each category for correlation analysis")
        return {}
    def build_layer_cosine_matrix(results):
        layer_cosine_matrix = {}
        for r in results:
            if r.get('layer_stats'):
                for lid, stats in r['layer_stats'].items():
                    layer_cosine_matrix.setdefault(lid, []).append(stats['mean'])
        min_samples = min(len(v) for v in layer_cosine_matrix.values()) if layer_cosine_matrix else 0
        return {lid:vals[:min_samples] for lid,vals in layer_cosine_matrix.items() if len(vals)>=min_samples}
    def compute_correlation_matrix(layer_matrix):
        correlations = {}
        layer_ids = sorted(layer_matrix.keys())
        for i,l1 in enumerate(layer_ids):
            for j,l2 in enumerate(layer_ids):
                if i<j:
                    try:
                        corr,p = pearsonr(layer_matrix[l1], layer_matrix[l2])
                        correlations[f"{l1}-{l2}"] = {
                            'correlation': corr, 'p_value': p, 'significant': p<0.05,
                            'layer1':l1, 'layer2':l2, 'abs_correlation':abs(corr)
                        }
                    except Exception as e:
                        print(f"⚠️  Could not compute correlation {l1}-{l2}: {e}")
        return correlations
    correct_matrix = build_layer_cosine_matrix(correct_results)
    incorrect_matrix = build_layer_cosine_matrix(incorrect_results)
    if not correct_matrix or not incorrect_matrix:
        print("❌ Insufficient layer data for correlation analysis")
        return {}
    correct_corr = compute_correlation_matrix(correct_matrix)
    incorrect_corr = compute_correlation_matrix(incorrect_matrix)
    common = set(correct_corr.keys()) & set(incorrect_corr.keys())
    if not common:
        print("❌ No common layer pairs")
        return {}
    comparison = {}
    for pair in common:
        comparison[pair] = {
            'correct_correlation': correct_corr[pair]['correlation'],
            'incorrect_correlation': incorrect_corr[pair]['correlation'],
            'correlation_difference': correct_corr[pair]['correlation'] - incorrect_corr[pair]['correlation'],
            'correct_p_value': correct_corr[pair]['p_value'],
            'incorrect_p_value': incorrect_corr[pair]['p_value'],
            'correct_significant': correct_corr[pair]['significant'],
            'incorrect_significant': incorrect_corr[pair]['significant'],
            'layer1': correct_corr[pair]['layer1'],
            'layer2': correct_corr[pair]['layer2']
        }
    diffs = [v['correlation_difference'] for v in comparison.values()]
    summary = {
        'num_layer_pairs': len(comparison),
        'num_correct_samples': len(correct_results),
        'num_incorrect_samples': len(incorrect_results),
        'mean_correlation_difference': np.mean(diffs) if diffs else 0,
        'std_correlation_difference': np.std(diffs) if diffs else 0,
        'max_correlation_difference': np.max(diffs) if diffs else 0,
        'min_correlation_difference': np.min(diffs) if diffs else 0
    }
    return {
        'correlation_comparison': comparison,
        'correct_correlations': correct_corr,
        'incorrect_correlations': incorrect_corr,
        'summary': summary
    }

def compute_layer_wise_cosine_accuracy_analysis(evaluation_results):
    if len(evaluation_results) < 2:
        print("❌ Need at least 2 samples for layer-wise analysis")
        return {}
    correct_results = [r for r in evaluation_results if r['correct'] and r.get('layer_stats')]
    incorrect_results = [r for r in evaluation_results if not r['correct'] and r.get('layer_stats')]
    if not correct_results or not incorrect_results:
        print("❌ Need both correct and incorrect predictions for comparison")
        return {}
    all_layers = set()
    for r in evaluation_results:
        if r.get('layer_stats'):
            all_layers.update(r['layer_stats'].keys())
    layer_analysis = {}
    for lid in sorted(all_layers):
        c_sims, i_sims = [], []
        for r in correct_results:
            if lid in r['layer_stats']:
                c_sims.extend(r['layer_stats'][lid].get('cosine_similarities', []))
        for r in incorrect_results:
            if lid in r['layer_stats']:
                i_sims.extend(r['layer_stats'][lid].get('cosine_similarities', []))
        if not c_sims or not i_sims:
            continue
        c_mean, i_mean = np.mean(c_sims), np.mean(i_sims)
        c_std,  i_std  = np.std(c_sims), np.std(i_sims)
        pooled = np.sqrt(((len(c_sims)-1)*c_std**2 + (len(i_sims)-1)*i_std**2)/ (len(c_sims)+len(i_sims)-2)) if (len(c_sims)+len(i_sims)-2)>0 else 0
        d = (c_mean - i_mean) / pooled if pooled > 0 else 0
        t_stat, p_val = ttest_ind(c_sims, i_sims, equal_var=False)
        layer_analysis[lid] = {
            'correct_predictions': {'count':len(c_sims), 'mean_cosine':c_mean, 'std_cosine':c_std, 'median_cosine':np.median(c_sims), 'min_cosine':np.min(c_sims), 'max_cosine':np.max(c_sims)},
            'incorrect_predictions': {'count':len(i_sims), 'mean_cosine':i_mean, 'std_cosine':i_std, 'median_cosine':np.median(i_sims), 'min_cosine':np.min(i_sims), 'max_cosine':np.max(i_sims)},
            'comparison': {'mean_difference':c_mean-i_mean, 'median_difference':np.median(c_sims)-np.median(i_sims), 'cohens_d':d, 't_statistic':t_stat, 'p_value':p_val, 'significant': p_val<0.05,
                           'effect_size_interpretation': ('Large' if abs(d)>=0.8 else 'Medium' if abs(d)>=0.5 else 'Small' if abs(d)>=0.2 else 'Negligible')}
        }
    return layer_analysis

# ---------------- Data loader ----------------
def load_local_dataset(file_path):
    print(f"📂 Loading local dataset from: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    header = data_lines[0].split('\t')
    data = []
    for line in data_lines[1:]:
        fields = line.split('\t')
        if len(fields) >= 6:
            data.append({
                'entry_id': fields[0],
                'audio_index': fields[1],
                'prompt_text': fields[2],
                'object': fields[3],
                'attribute': fields[4],
                'label': fields[5],
                'sampling': fields[6] if len(fields) > 6 else 'unknown'
            })
    print(f"📊 Loaded {len(data)} samples")
    return data

# ---------------- Stats utils (new) ----------------
def to_corr_from_cov(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    invd = 1.0 / d
    corr = cov * np.outer(invd, invd)
    corr[np.isnan(corr)] = 0.0
    corr = np.clip(corr, -1.0, 1.0)
    return corr

def compute_class_stats(X, y):
    """
    X: [N, d] feature matrix
    y: [N] boolean/int labels (1=correct, 0=wrong)
    Returns dict with mu1, mu0, Sigma1, Sigma0, pooled Sigma, correlations, counts, priors.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    idx1 = (y==1)
    idx0 = (y==0)
    X1 = X[idx1]; X0 = X[idx0]
    n1, n0 = X1.shape[0], X0.shape[0]
    mu1 = X1.mean(axis=0)
    mu0 = X0.mean(axis=0)
    # unbiased cov (rowvar=False)
    Sigma1 = np.cov(X1, rowvar=False, bias=False) if n1>1 else np.eye(X.shape[1]) * 1e-6
    Sigma0 = np.cov(X0, rowvar=False, bias=False) if n0>1 else np.eye(X.shape[1]) * 1e-6
    # pooled
    denom = max(n1+n0-2, 1)
    Sigma = ((n1-1)*Sigma1 + (n0-1)*Sigma0) / denom
    # correlations
    R1 = to_corr_from_cov(Sigma1)
    R0 = to_corr_from_cov(Sigma0)
    R  = to_corr_from_cov(Sigma)
    priors = {'pi1': float(n1/(n1+n0)), 'pi0': float(n0/(n1+n0))}
    return {
        'mu1': mu1.tolist(), 'mu0': mu0.tolist(),
        'Sigma1': Sigma1.tolist(), 'Sigma0': Sigma0.tolist(), 'Sigma': Sigma.tolist(),
        'R1': R1.tolist(), 'R0': R0.tolist(), 'R': R.tolist(),
        'counts': {'n1': int(n1), 'n0': int(n0)}, 'priors': priors
    }

# ---------------- Main ----------------
def main(args):
    global verbose_progress, vsv_enabled, vsv_lambda
    verbose_progress = args.verbose
    vsv_enabled = True
    vsv_lambda = args.vsv_lambda

    # dataset
    if hasattr(args, 'dataset_file') and args.dataset_file:
        print("📊 Loading local dataset...")
        dataset_samples = load_local_dataset(args.dataset_file)
        print("🔀 Shuffling...")
        random.shuffle(dataset_samples)
        if hasattr(args, 'max_samples') and args.max_samples and args.max_samples < len(dataset_samples):
            dataset_samples = dataset_samples[:args.max_samples]
            print(f"🔢 Limited to {args.max_samples} samples")
        total_samples = len(dataset_samples)
        use_local_dataset = True
    else:
        print("📊 Loading HF dataset...")
        dataset = load_dataset(args.dataset_name)
        dataset_samples = list(dataset['test'])
        print("🔀 Shuffling...")
        random.shuffle(dataset_samples)
        if hasattr(args, 'max_samples') and args.max_samples:
            dataset_samples = dataset_samples[:args.max_samples]
        total_samples = len(dataset_samples)
        use_local_dataset = False
        print(f"📝 Dataset loaded: {total_samples} samples")

    evaluation_results = []
    bos_rows = []         # <- NEW: per-sample BOS feature rows (for CSV)
    bos_X, bos_y = [], [] # <- NEW: matrices to compute class stats
    bos_feature_names = None
    saved_early_idx = None
    saved_late_idx  = None

    if model is None:
        initialize_model()

    print(f"🎯 Vector steering ENABLED with λ={vsv_lambda} for cosine measurement")
    print(f"🎯 Starting evaluation on {total_samples} samples...")
    start_time = time.time()

    for idx, sample in enumerate(tqdm(dataset_samples, desc="Processing samples", unit="sample")):
        entry_id    = sample["entry_id"]
        audio_index = sample["audio_index"]
        audio_path  = f"{args.audio_root_dir}/{audio_index}.wav"
        prompt_text = sample["prompt_text"]
        label       = sample["label"]
        sampling_method = sample.get("sampling","unknown") if use_local_dataset else "unknown"

        # Inference with generation & cosine logging (as before)
        response, layer_stats, overall_stats, vsv = inference_with_cosine_measurement(
            audio_path=audio_path, prompt_text=prompt_text, vsv_lambda=vsv_lambda
        )
        correct = (response == label)

        # --- NEW: BOS-only probe (no steering, no generation) ---
        try:
            if vsv is None:
                vsv = compute_vsv_for_audio(audio_path, prompt_text)
            bos_feats = probe_bos_features(audio_path, prompt_text, vsv)
            # Persist the early/late indices used (consistent per model run)
            if saved_early_idx is None: saved_early_idx = bos_feats["early_idx"]
            if saved_late_idx  is None: saved_late_idx  = bos_feats["late_idx"]

            # Build feature vector x and names
            x_vec, names, late_ids_for_vector = extract_bos_feature_vector(bos_feats, use_last_k_layers=8)
            if bos_feature_names is None: bos_feature_names = names
            bos_X.append(x_vec)
            bos_y.append(1 if correct else 0)

            # Row for BOS features CSV (human-readable)
            row = {
                "entry_id": entry_id,
                "audio_index": audio_index,
                "label": label,
                "response": response,
                "correct": int(correct),
                "bos_mean_late": bos_feats["bos_mean_late"],
                "bos_std_late": bos_feats["bos_std_late"],
                "entropy_bos": bos_feats["entropy_bos"],
                "margin_bos": bos_feats["margin_bos"],
                "gap_EL": bos_feats["gap_EL"],
            }
            # also write the exact cosine of the last-K layers we used in x_vec
            for j, lid in enumerate(late_ids_for_vector):
                row[f"cos_L{lid}"] = bos_feats["cos_bos_per_layer"][lid]
            bos_rows.append(row)
        except Exception as e:
            print(f"⚠️ BOS probe failed on {entry_id}: {e}")

        # Original result record (kept)
        mean_cosine   = overall_stats.get('mean', None) if overall_stats else None
        std_cosine    = overall_stats.get('std', None)  if overall_stats else None
        cosine_count  = overall_stats.get('count', 0)   if overall_stats else 0
        result_data = {
            'entry_id': entry_id,
            'audio_index': audio_index,
            'label': label,
            'response': response,
            'correct': correct,
            'mean_cosine': mean_cosine,
            'std_cosine': std_cosine,
            'cosine_count': cosine_count,
            'layer_stats': layer_stats,
            'vsv_lambda': vsv_lambda
        }
        if use_local_dataset:
            result_data['sampling_method'] = sampling_method
        evaluation_results.append(result_data)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_samples:
            correct_count = sum(1 for r in evaluation_results if r['correct'])
            accuracy = correct_count / len(evaluation_results) * 100
            elapsed = time.time() - start_time
            avg_t = elapsed / (idx+1)
            eta = (avg_t * total_samples - elapsed) / 60
            print(f"  📈 {idx+1}/{total_samples} | Acc: {accuracy:.1f}% | Avg/sample: {avg_t:.1f}s | ETA: {eta:.1f}m")

    # ---------------- Final stats & analyses ----------------
    total_time = time.time() - start_time
    correct = sum(1 for r in evaluation_results if r['correct'])
    final_accuracy = correct / len(evaluation_results) * 100
    print(f"\n🏁 Inference completed!")
    print(f"  📊 Final accuracy: {final_accuracy:.2f}% ({correct}/{total_samples})")
    print(f"  ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"  ⚡ Average time per sample: {total_time/total_samples:.1f}s")

    print(f"\n📐 Computing layer-wise cosine analysis...")
    layer_analysis = compute_layer_wise_cosine_accuracy_analysis(evaluation_results)

    print(f"\n🔗 Computing inter-layer correlation analysis...")
    correlation_analysis = compute_inter_layer_cosine_correlations(evaluation_results)

    # ---------------- Save original evaluation CSV ----------------
    output_path = args.output_path
    base, ext = (output_path.rsplit('.',1)+[""])[:2]
    eval_csv = f"{base}_cosine_correlation_lambda{vsv_lambda}.{ext or 'csv'}"
    print(f"💾 Saving results to {eval_csv}...")
    with open(eval_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        if use_local_dataset:
            writer.writerow(["entry_id","audio_index","label","response","correct","mean_cosine","std_cosine","cosine_count","sampling_method","vsv_lambda"])
        else:
            writer.writerow(["entry_id","audio_index","label","response","correct","mean_cosine","std_cosine","cosine_count","vsv_lambda"])
        for r in evaluation_results:
            if use_local_dataset:
                writer.writerow([r['entry_id'], r['audio_index'], r['label'], r['response'], r['correct'], r['mean_cosine'], r['std_cosine'], r['cosine_count'], r.get('sampling_method','unknown'), r['vsv_lambda']])
            else:
                writer.writerow([r['entry_id'], r['audio_index'], r['label'], r['response'], r['correct'], r['mean_cosine'], r['std_cosine'], r['cosine_count'], r['vsv_lambda']])

    # ---------------- Save BOS features CSV (NEW) ----------------
    bos_csv = f"{base}_BOS_features.csv"
    print(f"💾 Saving BOS features to {bos_csv}...")
    with open(bos_csv, mode='w', newline='') as f:
        fieldnames = ["entry_id","audio_index","label","response","correct","bos_mean_late","bos_std_late","entropy_bos","margin_bos","gap_EL"]
        # add the cos_L* columns present in bos_rows
        extra_cols = sorted({k for row in bos_rows for k in row.keys() if k.startswith("cos_L")}, key=lambda s:int(s.split("L")[1]))
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_cols)
        writer.writeheader()
        for row in bos_rows:
            writer.writerow(row)

    # ---------------- Save BOS class stats JSON (NEW) ----------------
    bos_stats_json = f"{base}_BOS_stats.json"
    stats_payload = {}
    if len(bos_X) >= 2 and (0 in bos_y) and (1 in bos_y):
        print("🧮 Computing BOS class statistics for LDA...")
        X = np.stack(bos_X, axis=0)
        y = np.array(bos_y, dtype=int)
        cls = compute_class_stats(X, y)
        stats_payload = {
            "feature_names": bos_feature_names or [],
            "class_stats": cls,
            "early_idx": saved_early_idx,
            "late_idx": saved_late_idx,
            "feature_vector_note": "x = [bos_mean_late, bos_std_late, entropy_bos, margin_bos, gap_EL, cos_L{last-K..last-1}]",
            "lda_equations": {
                "w": "w = Sigma^{-1} (mu1 - mu0)",
                "b": "b = -0.5 (mu1 + mu0)^T Sigma^{-1} (mu1 - mu0) + log(pi1/pi0)",
                "p_correct": "p = 1 / (1 + exp(-(w^T x + b)))"
            }
        }
    else:
        print("⚠️ Not enough BOS data to compute class stats (need both classes).")

    # Analysis JSON (kept + augmented)
    analysis_results = {
        'experiment_config': {
            'num_samples': total_samples,
            'vsv_lambda': vsv_lambda,
            'dataset_file': args.dataset_file if hasattr(args,'dataset_file') else args.dataset_name,
            'audio_root_dir': args.audio_root_dir
        },
        'layer_wise_analysis': layer_analysis,
        'inter_layer_correlation_analysis': correlation_analysis,
        'summary': {
            'accuracy': final_accuracy,
            'total_time_minutes': total_time/60,
            'samples_with_cosine_data': len([r for r in evaluation_results if r['mean_cosine'] is not None]),
            'layers_analyzed': len(layer_analysis) if layer_analysis else 0,
            'significant_layers': len([l for l,s in layer_analysis.items() if s['comparison']['significant']]) if layer_analysis else 0,
            'layer_pairs_analyzed': len(correlation_analysis.get('correlation_comparison', {})) if correlation_analysis else 0,
            'mean_correlation_difference': correlation_analysis.get('summary',{}).get('mean_correlation_difference',0) if correlation_analysis else 0
        },
        'bos': {
            'features_csv': os.path.abspath(bos_csv),
            'stats_json_target': os.path.abspath(bos_stats_json),
        }
    }

    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k,v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(x) for x in obj]
        elif isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.bool_): return bool(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        else: return obj

    analysis_json = f"{base}_analysis.json"
    with open(analysis_json, 'w') as f:
        json.dump(convert_numpy_types(analysis_results), f, indent=2)
    print(f"📊 Detailed analysis saved to {analysis_json}")

    if stats_payload:
        with open(bos_stats_json, 'w') as f:
            json.dump(convert_numpy_types(stats_payload), f, indent=2)
        print(f"📦 BOS LDA stats saved to {bos_stats_json}")

    print(f"✅ Done.")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio hallucination evaluation + BOS feature logging")
    parser.add_argument("--dataset_name", type=str, default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, default="./understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, default="./BOS_features.csv")
    parser.add_argument("--verbose","-v", action="store_true")
    parser.add_argument("--vsv_lambda", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
