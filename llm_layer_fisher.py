
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel
from typing import Optional, Tuple, List

"""
Audio-Language Model correspondence of llm_layers.py with Fisher (KL-trust-region) scaling.
Replaces VISTA's "add Δ then L2-renorm" with per-layer Fisher information scaling.
Uses Fisher diagonal approximation for KL divergence trust region constraint.

Key changes from original VISTA:
- Replaces L2 norm preservation with Fisher-based trust region scaling
- Computes Fisher diagonal from empirical covariance of input activations
- Scales perturbations to satisfy KL divergence constraint: δ^T F δ ≤ bound

Public API:
- get_layers(model, which_stack="auto"): choose decoder/encoder stack for ALMs
- add_vsv_layers(model, vsv, lam, fisher_scale, tar_layers=None, which_stack="auto")
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

# ---------------------------
# VSV layer (per paper, single vector per layer, norm-preserving)
# ---------------------------

class VSVLayer(nn.Module):
    """
    Adds a fixed steering direction v^l with Fisher (KL-trust-region) scaling.
    Uses per-layer Fisher information matrix diagonal for trust region constraint.
    This layer expects inputs of shape [B, T, d_model].
    """
    def __init__(self, vsv_l: Tensor, lam: float, fisher_scale: float = 1.0):
        super().__init__()
        assert vsv_l.dim() == 1, "vsv_l must be [d_model] for this block"
        self.register_buffer("vsv_l", vsv_l)
        self.lam = float(lam)
        self.fisher_scale = float(fisher_scale)
        # Fisher diagonal will be computed dynamically

    def compute_fisher_scaling(self, x: Tensor, delta: Tensor) -> Tensor:
        """
        Compute Fisher-based scaling for KL trust region constraint.
        
        Args:
            x: Input hidden states [B, T, d]
            delta: Steering perturbation [B, T, d]
            
        Returns:
            Scaled delta with Fisher trust region constraint
        """
        # Estimate Fisher information diagonal from input covariance
        # This approximates the Fisher diagonal as the empirical variance
        x_centered = x - x.mean(dim=(0, 1), keepdim=True)  # Center the data
        fisher_diag = (x_centered ** 2).mean(dim=(0, 1)) + 1e-8  # [d] + epsilon for stability
        
        # Compute KL divergence constraint: delta^T F delta <= trust_region_bound
        # Scale delta to satisfy trust region: delta_scaled = delta / sqrt(delta^T F delta / bound)
        delta_flat = delta.view(-1, delta.size(-1))  # [B*T, d]
        
        # Quadratic form: delta^T F delta for each token
        quadratic_form = (delta_flat ** 2) * fisher_diag.unsqueeze(0)  # [B*T, d]
        kl_divergence = quadratic_form.sum(dim=-1, keepdim=True)  # [B*T, 1]
        
        # Trust region bound (adaptive based on fisher_scale)
        trust_bound = self.fisher_scale
        
        # Scale factor to keep within trust region
        scale_factor = torch.sqrt(trust_bound / (kl_divergence + 1e-8))
        scale_factor = torch.clamp(scale_factor, max=1.0)  # Don't amplify, only reduce
        
        # Apply scaling
        scale_factor = scale_factor.view(delta.shape[0], delta.shape[1], 1)  # [B, T, 1]
        scaled_delta = delta * scale_factor
        
        return scaled_delta

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d]
        if self.vsv_l is None:
            return x
            
        orig_dtype = x.dtype
        x = x.float()
        
        # Compute steering perturbation
        v = self.vsv_l.view(1, 1, -1)  # [1,1,d] for broadcasting
        delta = self.lam * v  # [1,1,d] -> broadcasts to [B,T,d]
        delta = delta.expand_as(x)  # Explicitly expand for Fisher computation
        
        # Apply Fisher-based trust region scaling
        scaled_delta = self.compute_fisher_scaling(x, delta)
        
        # Apply scaled perturbation
        x_perturbed = x + scaled_delta
        
        return x_perturbed.to(orig_dtype)

# ---------------------------
# Block wrapper (post-block residual steering)
# ---------------------------

class BlockPostVSV(nn.Module):
    """
    Wraps a transformer block. After the block computes its output hidden states,
    apply VSVLayer with Fisher scaling to the output tensor (first element if tuple).
    """
    def __init__(self, block: nn.Module, vsv_l: Tensor, lam: float, fisher_scale: float = 1.0):
        super().__init__()
        self.block = block
        self.vsv = VSVLayer(vsv_l=vsv_l, lam=lam, fisher_scale=fisher_scale)

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

def _make_vsv_hook(v_l: Tensor, lam: float, fisher_scale: float = 1.0):
    """
    Creates a hook that implements Fisher (KL-trust-region) scaling.
    """
    def compute_fisher_scaling(x: Tensor, delta: Tensor) -> Tensor:
        """Fisher-based trust region scaling."""
        # Estimate Fisher information diagonal from input covariance
        x_centered = x - x.mean(dim=(0, 1), keepdim=True)  # Center the data
        fisher_diag = (x_centered ** 2).mean(dim=(0, 1)) + 1e-8  # [d] + epsilon for stability
        
        # Compute KL divergence constraint: delta^T F delta <= trust_region_bound
        delta_flat = delta.view(-1, delta.size(-1))  # [B*T, d]
        
        # Quadratic form: delta^T F delta for each token
        quadratic_form = (delta_flat ** 2) * fisher_diag.unsqueeze(0)  # [B*T, d]
        kl_divergence = quadratic_form.sum(dim=-1, keepdim=True)  # [B*T, 1]
        
        # Trust region bound
        trust_bound = fisher_scale
        
        # Scale factor to keep within trust region
        scale_factor = torch.sqrt(trust_bound / (kl_divergence + 1e-8))
        scale_factor = torch.clamp(scale_factor, max=1.0)  # Don't amplify, only reduce
        
        # Apply scaling
        scale_factor = scale_factor.view(delta.shape[0], delta.shape[1], 1)  # [B, T, 1]
        scaled_delta = delta * scale_factor
        
        return scaled_delta
    
    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out  # [B,T,D]
        orig_dtype = h.dtype
        x = h.float()
        
        # Compute steering perturbation
        v = v_l.view(1, 1, -1).to(x.device)  # [1,1,D] for broadcasting
        delta = lam * v  # [1,1,D] -> broadcasts to [B,T,D]
        delta = delta.expand_as(x)  # Explicitly expand for Fisher computation
        
        # Apply Fisher-based trust region scaling
        scaled_delta = compute_fisher_scaling(x, delta)
        
        # Apply scaled perturbation
        x = x + scaled_delta
        
        x = x.to(orig_dtype)
        return (x,) + out[1:] if isinstance(out, tuple) else x
    return hook

def add_vsv_layers(
    model: PreTrainedModel,
    vsv: Tensor,                 # [L, d_model]
    lam: float,
    tar_layers: Optional[str] = None,
    which_stack: str = "decoder",
    fisher_scale: float = 1.0,   # Fisher trust region scale
):
    layers = get_layers(model, which_stack=which_stack)
    layers, vsv = _slice_layers_and_vsv(layers, vsv, tar_layers)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for i, blk in enumerate(layers):
        h = blk.register_forward_hook(_make_vsv_hook(vsv[i], lam, fisher_scale))
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
