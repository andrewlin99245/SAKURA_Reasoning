import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel
from typing import Optional, Tuple, List

"""
Adaptive Audio-Language Model VSV Layer Implementation
Extends the base llm_layer.py to support layer-specific lambda and norm scaling
based on layer index for optimal audio-to-language transition handling.

Public API:
- get_layers(model, which_stack="auto"): choose decoder/encoder stack for ALMs
- add_vsv_layers_adaptive(model, vsv, tar_layers=None, which_stack="auto")
- remove_vsv_layers(model, which_stack="auto")
"""

# Import helper functions from the base module
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from llm_layer import get_layers, get_nested_attr, _slice_layers_and_vsv

# ---------------------
# Adaptive Functions
# ---------------------
BASE_LAMBDA = 0.0275

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

# ---------------------------
# Adaptive VSV layer
# ---------------------------

class AdaptiveVSVLayer(nn.Module):
    """
    Adds a fixed steering direction v^l with adaptive strength lambda and norm scaling
    based on layer index, then renormalizes with adaptive scaling to preserve original L2 norm.
    This layer expects inputs of shape [B, T, d_model].
    """
    def __init__(self, vsv_l: Tensor, layer_idx: int):
        super().__init__()
        assert vsv_l.dim() == 1, "vsv_l must be [d_model] for this block"
        self.register_buffer("vsv_l", vsv_l)
        self.layer_idx = layer_idx
        self.lam = get_adaptive_lambda(layer_idx)
        self.norm_scale = get_adaptive_norm_scale(layer_idx)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d]
        if self.vsv_l is None:
            return x
        orig_dtype = x.dtype
        x = x.float()
        
        # Store original norm for each token (VISTA Eq. 6)
        original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # [B, T, 1]
        
        # VISTA Eq. 5: h_tilde = h + λ * v_steer (with adaptive lambda)
        # Note: vsv_l is NOT pre-normalized, following paper exactly
        v = self.vsv_l.view(1, 1, -1)                              # [1,1,d] for broadcasting
        x = x + self.lam * v                                       # [B,T,d] + [1,1,d] -> [B,T,d]
        
        # VISTA Eq. 6: normalize to preserve original norm magnitude with adaptive scaling
        x = torch.nn.functional.normalize(x, p=2, dim=-1) * (original_norm * self.norm_scale)
        
        return x.to(orig_dtype)

# ---------------------------
# Adaptive hooks
# ---------------------------

def _make_adaptive_vsv_hook(v_l: Tensor, layer_idx: int):
    """
    Creates a hook that implements VISTA Eq. 5-6 with adaptive lambda and norm scaling:
    - Eq. 5: h_tilde = h + λ * v_steer (with adaptive λ based on layer)
    - Eq. 6: normalize to preserve original norm (with adaptive scaling)
    """
    lam = get_adaptive_lambda(layer_idx)
    norm_scale = get_adaptive_norm_scale(layer_idx)
    
    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out  # [B,T,D]
        orig_dtype = h.dtype
        x = h.float()
        
        # Store original norm for each token (VISTA Eq. 6 preparation)
        norm = x.norm(p=2, dim=-1, keepdim=True)       # [B,T,1]
        
        # VISTA Eq. 5: h_tilde = h + λ * v_steer (with adaptive λ)
        # Note: v_l is NOT pre-normalized, as per paper
        y = lam * v_l.view(1, 1, -1).to(x.device)      # [1,1,D] broadcast to [B,T,D]
        x = x + y
        
        # VISTA Eq. 6: normalize to preserve original norm magnitude (with adaptive scaling)
        x = F.normalize(x, p=2, dim=-1) * (norm * norm_scale)
        
        x = x.to(orig_dtype)
        return (x,) + out[1:] if isinstance(out, tuple) else x
    return hook

# ---------------------------
# Adaptive injection / removal
# ---------------------------

def add_vsv_layers_adaptive(
    model: PreTrainedModel,
    vsv: Tensor,                 # [L, d_model]
    tar_layers: Optional[str] = None,
    which_stack: str = "decoder",
):
    """
    Add adaptive VSV layers with layer-specific lambda and norm scaling.
    """
    layers = get_layers(model, which_stack=which_stack)
    layers, vsv = _slice_layers_and_vsv(layers, vsv, tar_layers)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    
    # Calculate the starting layer index offset for proper adaptive calculation
    all_layers = get_layers(model, which_stack=which_stack)
    layer_offset = 0
    if tar_layers is not None:
        parts = [int(x) for x in tar_layers.split(",")]
        if len(parts) == 2:
            layer_offset = parts[0]  # start index
        elif len(parts) == 4:
            layer_offset = parts[2]  # layer start index
    
    for i, blk in enumerate(layers):
        actual_layer_idx = layer_offset + i
        h = blk.register_forward_hook(_make_adaptive_vsv_hook(vsv[i], actual_layer_idx))
        handles.append(h)
    
    # stash handles so we can remove later
    if not hasattr(model, "_vsv_handles"):
        model._vsv_handles = []
    model._vsv_handles.extend(handles)

def remove_vsv_layers(model: PreTrainedModel, which_stack: str = "decoder"):
    """
    Remove VSV layer hooks (same as base implementation).
    """
    if hasattr(model, "_vsv_handles"):
        for h in model._vsv_handles:
            try: h.remove()
            except: pass
        model._vsv_handles = []