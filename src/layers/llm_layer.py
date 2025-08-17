
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel
from typing import Optional, Tuple, List
import statistics

"""
Audio-Language Model correspondence of llm_layers.py
Implements VSV injection per VISTA Eq. (5)-(6), but adapted to wrap
the *whole transformer block output* (i.e., residual stream), not just MLP.
No SLA here.

Public API:
- get_layers(model, which_stack="auto"): choose decoder/encoder stack for ALMs
- add_vsv_layers(model, vsv, lam, tar_layers=None, which_stack="auto")
- remove_vsv_layers(model, which_stack="auto")
"""

# ---------------------------
# Model graph helpers
# ---------------------------

def get_nested_attr(obj, attr_path: str):
    attrs = attr_path.split(".") if attr_path else []
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def find_longest_modulelist(model: nn.Module, path: str = "") -> Tuple[str, int]:
    longest_path = path
    longest_len = 0
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path
    return longest_path, longest_len

def _try_layer_paths(model: nn.Module, candidates: List[str]) -> Optional[nn.ModuleList]:
    for p in candidates:
        try:
            layers = get_nested_attr(model, p)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
        except Exception:
            continue
    return None

def get_layers_path(model: PreTrainedModel, which_stack: str = "auto") -> str:
    """
    Returns the dotted path to the chosen ModuleList of blocks.
    Prefers decoder stacks when present.
    """
    which_stack = (which_stack or "auto").lower()

    if which_stack in ("decoder", "auto"):
        dec_candidates = [
            "model.decoder.layers",    # whisper/seq2seq
            "decoder.layers",
            "model.model.layers",      # llama/qwen-like
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "language_model.model.layers",
            "model.language_model.model.layers",
        ]
        layers = _try_layer_paths(model, dec_candidates)
        if layers is not None:
            for p in dec_candidates:
                try:
                    if get_nested_attr(model, p) is layers:
                        return p
                except Exception:
                    pass

    if which_stack in ("encoder", "auto"):
        enc_candidates = [
            "model.encoder.layers",
            "encoder.layers",
            "audio_encoder.layers",
            "speech_encoder.layers",
        ]
        layers = _try_layer_paths(model, enc_candidates)
        if layers is not None:
            for p in enc_candidates:
                try:
                    if get_nested_attr(model, p) is layers:
                        return p
                except Exception:
                    pass

    # Fallback: longest ModuleList
    longest_path, _ = find_longest_modulelist(model)
    return longest_path

def get_layers(model: PreTrainedModel, which_stack: str = "auto") -> nn.ModuleList:
    path = get_layers_path(model, which_stack=which_stack)
    return get_nested_attr(model, path)

# Global storage for angle measurements
_angle_storage = []

# ---------------------------
# VSV layer (per paper, single vector per layer, norm-preserving)
# ---------------------------

class VSVLayer(nn.Module):
    """
    Adds a fixed steering direction v^l with strength lambda (lam) to hidden states,
    then renormalizes to preserve original L2 norm.
    This layer expects inputs of shape [B, T, d_model].
    """
    def __init__(self, vsv_l: Tensor, lam: float):
        super().__init__()
        assert vsv_l.dim() == 1, "vsv_l must be [d_model] for this block"
        self.register_buffer("vsv_l", vsv_l)
        self.lam = float(lam)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d]
        if self.vsv_l is None:
            return x
        orig_dtype = x.dtype
        x = x.float()
        
        # Store original norm for each token (VISTA Eq. 6)
        original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # [B, T, 1]
        
        # VISTA Eq. 5: h_tilde = h + λ * v_steer
        # Note: vsv_l is NOT pre-normalized, following paper exactly
        v = self.vsv_l.view(1, 1, -1)                              # [1,1,d] for broadcasting
        x = x + self.lam * v                                       # [B,T,d] + [1,1,d] -> [B,T,d]
        
        # VISTA Eq. 6: normalize to preserve original norm magnitude
        x = torch.nn.functional.normalize(x, p=2, dim=-1) * original_norm
        
        return x.to(orig_dtype)

# ---------------------------
# Block wrapper (post-block residual steering)
# ---------------------------

class BlockPostVSV(nn.Module):
    """
    Wraps a transformer block. After the block computes its output hidden states,
    apply VSVLayer to the output tensor (first element if tuple).
    """
    def __init__(self, block: nn.Module, vsv_l: Tensor, lam: float):
        super().__init__()
        self.block = block
        self.vsv = VSVLayer(vsv_l=vsv_l, lam=lam)

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            h = out[0]
            h = self.vsv(h)
            return (h,) + out[1:]
        else:
            return self.vsv(out)

# ---------------------------
# Injection / removal
# ---------------------------

def _slice_layers_and_vsv(layers: nn.ModuleList, vsv: Tensor, tar_layers: Optional[str]):
    """
    vsv expected shape: [L, d_model], aligned with 'layers' length.
    tar_layers can be:
      - "s,e" to select source [s:e] from both layers and vsv,
      - "s1,e1,s2,e2" to map vsv[s1:e1] onto layers[s2:e2].
    """
    if tar_layers is None:
        assert len(vsv) == len(layers), f"vsv length {len(vsv)} != #layers {len(layers)}"
        return layers, vsv
    parts = [int(x) for x in tar_layers.split(",")]
    if len(parts) == 2:
        s, e = parts
        #assert len(vsv) >= (e - s), "vsv too short for selected layers"
        return layers[s:e], vsv[s:e]
    if len(parts) == 4:
        s1, e1, s2, e2 = parts
        return layers[s2:e2], vsv[s1:e1]
    raise ValueError("Invalid tar_layers; use 's,e' or 's1,e1,s2,e2'.")

def _make_vsv_hook(v_l: Tensor, lam: float, layer_idx: int = -1):
    """
    Creates a hook that implements VISTA Eq. 5-6 correctly:
    - Eq. 5: h_tilde = h + λ * v_steer  
    - Eq. 6: normalize to preserve original norm
    - Also measures angles between steering vector and hidden states before steering
    """
    def hook(_module, _inp, out):
        global _angle_storage
        h = out[0] if isinstance(out, tuple) else out  # [B,T,D]
        orig_dtype = h.dtype
        x = h.float()
        
        # Calculate angle between steering vector and hidden states (before steering)
        # Use per-token cosine similarity then average (Approach 2)
        # This gives equal weight to all tokens regardless of magnitude
        
        # Normalize hidden states per token: [B, T, D]
        x_normalized = F.normalize(x, p=2, dim=-1)  # [B, T, D]
        
        # Normalize steering vector: [D] -> [1, 1, D] for broadcasting
        v = v_l.view(1, 1, -1).to(x.device)  # [1, 1, D]
        v_normalized = F.normalize(v, p=2, dim=-1)  # [1, 1, D]
        
        # Compute per-token cosine similarity: [B, T]
        per_token_cos_sim = torch.sum(x_normalized * v_normalized, dim=-1)  # [B, T]
        
        # Average cosine similarity across tokens: [B]
        cos_sim = per_token_cos_sim.mean(dim=1)  # [B]
        
        # Convert to angle
        angle_rad = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))  # [B]
        angle_deg = angle_rad * 180.0 / torch.pi  # [B]
        
        # Store angles for later analysis (using first batch item)
        if layer_idx >= 0:  # Only store if layer_idx is valid
            _angle_storage.append({
                'layer_idx': layer_idx,
                'angle': angle_deg[0].item()
            })
        
        # Store original norm for each token (VISTA Eq. 6 preparation)
        norm = x.norm(p=2, dim=-1, keepdim=True)       # [B,T,1]
        #if layer_idx == 30 or layer_idx==31:
            #norm = norm * (1-(cos_sim[0]/5))
            #print(f"Layer {layer_idx}: Adjusted norm = {1-(cos_sim[0]/5)}")
            #print(cos_sim)
        #    pass  # Placeholder - all code is commented out
        # VISTA Eq. 5: h_tilde = h + λ * v_steer
        # Note: v_l is NOT pre-normalized, as per paper
        y = lam * v_l.view(1, 1, -1).to(x.device)      # [1,1,D] broadcast to [B,T,D]
        x = x + y
        
        # VISTA Eq. 6: normalize to preserve original norm magnitude
        x = F.normalize(x, p=2, dim=-1) * norm
        
        x = x.to(orig_dtype)
        return (x,) + out[1:] if isinstance(out, tuple) else x
    return hook

def add_vsv_layers(
    model: PreTrainedModel,
    vsv: Tensor,                 # [L, d_model]
    lam: float,
    tar_layers: Optional[str] = None,
    which_stack: str = "decoder",
):
    layers = get_layers(model, which_stack=which_stack)
    layers, vsv = _slice_layers_and_vsv(layers, vsv, tar_layers)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for i, blk in enumerate(layers):
        h = blk.register_forward_hook(_make_vsv_hook(vsv[i], lam, layer_idx=i))
        handles.append(h)
    # stash handles so we can remove later
    if not hasattr(model, "_vsv_handles"):
        model._vsv_handles = []
    model._vsv_handles.extend(handles)

def remove_vsv_layers(model: PreTrainedModel, which_stack: str = "decoder"):
    if hasattr(model, "_vsv_handles"):
        for h in model._vsv_handles:
            try: h.remove()
            except: pass
        model._vsv_handles = []

def clear_angle_storage():
    """
    Clear the global angle storage.
    Call this before starting a new generation to reset measurements.
    """
    global _angle_storage
    _angle_storage = []

def print_angle_statistics():
    """
    Compute and print average angle and standard deviation from collected measurements.
    Call this after generation is complete.
    """
    global _angle_storage
    
    if not _angle_storage:
        print("No angle measurements collected.")
        return
    
    # Extract all angles
    all_angles = [entry['angle'] for entry in _angle_storage]
    
    # Compute statistics
    avg_angle = statistics.mean(all_angles)
    
    if len(all_angles) > 1:
        std_angle = statistics.stdev(all_angles)
    else:
        std_angle = 0.0
    
    # Print results
    print(f"\n=== Angle Statistics ====")
    print(f"Total measurements: {len(all_angles)}")
    print(f"Average angle: {avg_angle:.2f}°")
    print(f"Standard deviation: {std_angle:.2f}°")
    print(f"Min angle: {min(all_angles):.2f}°")
    print(f"Max angle: {max(all_angles):.2f}°")
    print(f"========================\n")

def get_angle_statistics():
    """
    Return angle statistics as a dictionary instead of printing.
    """
    global _angle_storage
    
    if not _angle_storage:
        return None
    
    # Extract all angles
    all_angles = [entry['angle'] for entry in _angle_storage]
    
    # Compute statistics
    avg_angle = statistics.mean(all_angles)
    
    if len(all_angles) > 1:
        std_angle = statistics.stdev(all_angles)
    else:
        std_angle = 0.0
    
    return {
        'count': len(all_angles),
        'mean': avg_angle,
        'std': std_angle,
        'min': min(all_angles),
        'max': max(all_angles),
        'all_angles': all_angles
    }

def get_layer_by_layer_statistics():
    """
    Return layer-by-layer angle statistics as a dictionary.
    """
    global _angle_storage
    
    if not _angle_storage:
        return None
    
    # Group angles by layer
    layer_angles = {}
    for entry in _angle_storage:
        layer_idx = entry['layer_idx']
        angle = entry['angle']
        
        if layer_idx not in layer_angles:
            layer_angles[layer_idx] = []
        layer_angles[layer_idx].append(angle)
    
    # Compute statistics for each layer
    layer_stats = {}
    for layer_idx, angles in layer_angles.items():
        if len(angles) > 1:
            std_angle = statistics.stdev(angles)
        else:
            std_angle = 0.0
            
        layer_stats[layer_idx] = {
            'count': len(angles),
            'mean': statistics.mean(angles),
            'std': std_angle,
            'min': min(angles),
            'max': max(angles),
            'angles': angles
        }
    
    return layer_stats

def print_layer_by_layer_statistics():
    """
    Print layer-by-layer angle statistics in a formatted table.
    """
    layer_stats = get_layer_by_layer_statistics()
    
    if not layer_stats:
        print("No layer-by-layer measurements collected.")
        return
    
    print("\n=== Layer-by-Layer Angle Statistics ===")
    print(f"{'Layer':<6} {'Count':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 50)
    
    for layer_idx in sorted(layer_stats.keys()):
        stats = layer_stats[layer_idx]
        print(f"{layer_idx:<6} {stats['count']:<6} {stats['mean']:<8.2f} {stats['std']:<8.2f} {stats['min']:<8.2f} {stats['max']:<8.2f}")
    
    print("=" * 50 + "\n")
