
import torch
from dataclasses import dataclass
from typing import List, Tuple
from transformers import PreTrainedModel

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, "src", "layers", "variants"))
from llm_layer_fisher import get_layers

"""
Audio-Language Model correspondence of steering_vector.py
Implements per-image, per-layer VSV as V_p - V_n (Eq. 4), using
the *residual stream at the last token* from each transformer block.
No PCA, no SLA here—faithful to paper's VSV.
"""

@dataclass
class ResidualStream:
    # collected last-token hidden per layer: List[Tensor [B, D]]
    hidden: List[torch.Tensor]

class ForwardTrace:
    def __init__(self, num_layers: int):
        self.residual_stream = ResidualStream(hidden=[None] * num_layers)
        self._num_layers = num_layers

class ForwardTracer:
    """
    Hook on each transformer block (as returned by get_layers)
    and capture its output hidden states' LAST TOKEN for the batch.
    """
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.layers = get_layers(model,which_stack='decoder')  # ModuleList of blocks
        self.trace = ForwardTrace(num_layers=len(self.layers))
        self._hooks = []

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def _register_hooks(self):
        def store_last_token(layer_idx: int):
            def hook(_module, _inp, out):
                # out may be Tensor [B,T,D] or tuple (hidden_states, ...)
                if isinstance(out, tuple):
                    out = out[0]
                # shape check
                if out.dim() == 2:
                    # [T, D] -> treat T as sequence with B=1
                    last = out[-1:, :].contiguous()  # [1, D]
                elif out.dim() == 3:
                    # [B, T, D]
                    last = out[:, -1, :].contiguous()  # [B, D]
                else:
                    raise RuntimeError(f"Unexpected block output shape: {tuple(out.shape)}")
                last = last.float().cpu()
                self.trace.residual_stream.hidden[layer_idx] = last
            return hook

        for i, layer in enumerate(self.layers):
            self._hooks.append(layer.register_forward_hook(store_last_token(i)))

    # After model forward, returns Tensor [L, D] (batch-averaged if B>1)
    def stacked_last_token(self) -> torch.Tensor:
        h_list = self.trace.residual_stream.hidden
        if any(h is None for h in h_list):
            missing = [i for i, h in enumerate(h_list) if h is None]
            raise RuntimeError(f"Missing hidden for layers: {missing}")
        # average across batch dim if present
        h_proc = []
        for h in h_list:
            if h.dim() == 2:  # [B, D]
                h_proc.append(h.mean(dim=0, keepdim=False))  # [D]
            else:  # [1, D]
                h_proc.append(h.squeeze(0))
        return torch.stack(h_proc, dim=0)  # [L, D]

@torch.no_grad()
def get_hiddenstates(model: PreTrainedModel, kwargs_list: List[List[dict]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Replicates original signature conceptually:
      kwargs_list[example_id] = [neg_kwargs, pos_kwargs]
    Returns: list of tuples (Vn, Vp), each of shape [L, D]
    """
    results = []
    for pair in kwargs_list:
        assert len(pair) == 2, "Each entry must be [neg_kwargs, pos_kwargs]"
        # Negative (no audio tokens)
        with ForwardTracer(model) as tr:
            _ = model(use_cache=True, **pair[0])
            Vn = tr.stacked_last_token()  # [L, D]
        # Positive (with audio tokens)
        with ForwardTracer(model) as tr:
            _ = model(use_cache=True, **pair[1])
            Vp = tr.stacked_last_token()  # [L, D]
        results.append((Vn, Vp))
    return results

@torch.no_grad()
def obtain_vsv(model: PreTrainedModel, kwargs_list: List[List[dict]]) -> torch.Tensor:
    """
    Compute per-layer VSV by averaging (Vp - Vn) across provided pairs.
    Returns Tensor [L, D].
    """
    pairs = get_hiddenstates(model, kwargs_list)
    diffs = [Vp - Vn for (Vn, Vp) in pairs]  # each [L, D]
    vsv = torch.stack(diffs, dim=0).mean(dim=0)  # [L, D]
    return vsv
