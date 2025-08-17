
import inspect
from typing import Optional, List, Union, Any, Tuple

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioForConditionalGeneration,
)


def _filter_kwargs_for_fn(fn, kwargs: dict):
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}


class Qwen2AudioCosineSLAForCausalLM(Qwen2AudioForConditionalGeneration):
    """
    Drop-in replacement with cosine-SLA gating:
      - Only blend *preceding* layers' logits when their cosine similarity
        (vs its layer's steering vector) is greater than the final layer's cosine.
      - Gating is per-example (elementwise), not per-batch.
      - Projects only the last position to save compute/memory.
      - Uses safe, non-negative weights and (optionally) renormalizes them.
    """
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__(config)
        # Cosine SLA knobs
        self.sla_enable = False
        self.sla_gamma = 0.3      # scale for intermediate weights
        self.sla_w = 3            # number of *preceding* layers to consider
        self.sla_renorm = True    # renormalize weights so they sum to 1
        self.sla_debug = False    # set True to print debug once per forward
        self.sla_lambda = 0.05    # steering strength for cosine similarity measurement

        # Head for logits
        self._lm_head = self.get_output_embeddings()

        # Steering vectors
        self.steering_vectors = None          # [w_eff, H]  for preceding layers (excludes final)
        self.final_steering_vector = None     # [H]         for final layer
        self._steering_vector_ready = False

    # ---------------- public API ----------------
    def enable_sla(self, gamma: float = 0.3, w: int = 3, renorm: bool = True, debug: bool = False, lam: float = 0.05):
        """Enable cosine similarity-based SLA. Call after steering vectors are computed."""
        if not self._steering_vector_ready:
            print("Warning: SLA enabled but steering vectors not ready. SLA will have no effect.")
        self.sla_enable = True
        self.sla_gamma = float(gamma)
        self.sla_w = int(w)
        self.sla_renorm = bool(renorm)
        self.sla_debug = bool(debug)
        self.sla_lambda = float(lam)

    def disable_sla(self):
        self.sla_enable = False

    def set_steering_vectors(self, steering_vectors: torch.Tensor):
        """
        Set steering vectors for the last w+1 layers.

        Args:
            steering_vectors: Tensor of shape [L, H] where L is total number of transformer layers.
                              We will store the last w preceding layers (L-1-w .. L-2) and the final layer (L-1).
        """
        if steering_vectors is None:
            self.steering_vectors = None
            self.final_steering_vector = None
            self._steering_vector_ready = False
            return

        assert steering_vectors.dim() == 2, "steering_vectors must be [L, H]"
        L = steering_vectors.shape[0]
        if L < 2:
            # need at least 1 preceding + L-1 (final)
            self.steering_vectors = None
            self.final_steering_vector = None
            self._steering_vector_ready = False
            return

        w_eff = max(0, min(self.sla_w, L - 1))  # preceding layers count (exclude only L-1)
        start = (L - 1) - w_eff                  # inclusive
        end_excl = L - 1                         # exclusive (final layer L-1 excluded here)
        
        # Adjust to match the actual layers used in forward pass
        # forward pass uses: range(L-1-w_eff, L-1) → layers [L-1-w_eff+1, L-1-w_eff+2, ..., L-1]
        # So we need steering vectors for layers [L-1-w_eff+1, L-1-w_eff+2, ..., L-1] 
        actual_start = start + 1  # Add 1 because forward pass uses i+1 indexing
        actual_end_excl = end_excl + 1

        if w_eff > 0:
            self.steering_vectors = steering_vectors[actual_start:actual_end_excl].contiguous().clone()  # [w_eff, H]
        else:
            self.steering_vectors = torch.empty((0, steering_vectors.shape[1]), dtype=steering_vectors.dtype,
                                                device=steering_vectors.device)

        self.final_steering_vector = steering_vectors[L - 1].contiguous().clone()          # [H]
        self._steering_vector_ready = True

        #if self.sla_debug:
        #    print(f"✅ Steering vectors set for layers {start}..{L-2} (count={w_eff}) + final {L-1}")

    def is_steering_ready(self) -> bool:
        """Check if steering vectors are ready for cosine SLA."""
        return self._steering_vector_ready and (self.final_steering_vector is not None)

    # ------------- generation plumbing ----------
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        parent_fn = super().prepare_inputs_for_generation
        # call parent first with only what it understands
        parent_inputs = _filter_kwargs_for_fn(
            parent_fn,
            dict(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            ),
        )
        # Remove input_ids from parent_inputs to avoid duplicate argument
        input_ids_arg = parent_inputs.pop('input_ids', input_ids)
        model_inputs = parent_fn(input_ids_arg, **parent_inputs)

        # ensure audio-related tensors survive the 1st step (if present)
        for k in ["input_features", "feature_attention_mask",
                  "audio_values", "audio_attention_mask"]:
            if k in kwargs and k not in model_inputs:
                model_inputs[k] = kwargs[k]

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Declare the common audio kwargs. If your version doesn't have them,
        # _filter_kwargs_for_fn will just drop them.
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        need_hidden = (self.sla_enable and self.sla_gamma > 0.0 and
                       self.sla_w > 0 and self.is_steering_ready())
        if need_hidden and output_hidden_states is None:
            output_hidden_states = True

        # forward to parent with only the args it supports
        parent_fn = super().forward
        parent_kwargs = _filter_kwargs_for_fn(
            parent_fn,
            dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_values=audio_values,
                audio_attention_mask=audio_attention_mask,
                **kwargs,
            ),
        )
        outputs = parent_fn(**parent_kwargs)

        # no SLA? just return
        if not need_hidden or outputs.hidden_states is None:
            if self.sla_debug:
                print("⚠️  Cosine SLA: Skipping (need_hidden={}, hidden_states={})"
                      .format(need_hidden, outputs.hidden_states is not None))
            return outputs

        logits = outputs.logits                  # [B, T, V] - final layer logits
        hidden_states = outputs.hidden_states    # tuple length = L+1 (including embeddings)

        L = len(hidden_states) - 1  # number of transformer layers
        if L < 2:  # Need at least 2 layers to have preceding layers after excluding only L-1
            return outputs

        w_eff = min(self.sla_w, L - 1)  # preceding layers count (exclude only L-1)
        if self.steering_vectors is None or self.steering_vectors.shape[0] < w_eff:
            # Not enough vectors cached; be conservative.
            if self.sla_debug:
                sv = None if self.steering_vectors is None else tuple(self.steering_vectors.shape)
                print(f"⚠️  Cosine SLA: Early return (w_eff={w_eff}, steering_vectors.shape={sv})")
            return outputs

        B, T, V = logits.shape

        # 1) Final-layer cosine threshold (use last position only)
        # Apply steering temporarily only for cosine similarity computation
        final_layer_hs = hidden_states[L]                    # [B, T, H]
        final_vec = self.final_steering_vector.to(final_layer_hs.device)  # [H]
        
        # Apply steering temporarily only for cosine similarity computation (don't modify original hs)
        final_hs_steered_temp = final_layer_hs + self.sla_lambda * final_vec.view(1, 1, -1)  # [B, T, H]
        
        # Use per-token cosine similarity then average across tokens (same method as llm_layer.py)
        final_hs_norm = F.normalize(final_hs_steered_temp, p=2, dim=-1)  # [B, T, H]
        final_vec_norm = F.normalize(final_vec.view(1, 1, -1), p=2, dim=-1) # [1, 1, H]
        
        # Compute per-token cosine similarity: [B, T]
        per_token_cos_sim = torch.sum(final_hs_norm * final_vec_norm, dim=-1)  # [B, T]
        
        # Average across tokens to get final cosine similarity: [B]
        final_cos_sim = per_token_cos_sim.mean(dim=1)  # [B]

        # 2) Gather preceding layers' hidden states (exclude only final layer L-1, include L-2)
        start_i = L - 1 - w_eff
        target_hidden_states: List[torch.Tensor] = [hidden_states[i + 1] for i in range(start_i, L - 1)]  # [B,T,H] each

        # 3) Compute cosine similarities and last-position logits per preceding layer
        cosine_gated_list: List[torch.Tensor] = []   # each [B]
        layer_last_logits: List[torch.Tensor] = []   # each [B,1,V]

        for i, hs in enumerate(target_hidden_states):
            # steering for corresponding layer
            steer_vec = self.steering_vectors[i].to(hs.device)           # [H]
            
            # Apply steering temporarily only for cosine similarity computation (don't modify original hs)
            hs_steered_temp = hs + self.sla_lambda * steer_vec.view(1, 1, -1)  # [B, T, H]
            
            # Compute per-token cosine similarity then average (same method as llm_layer.py)
            hs_norm = F.normalize(hs_steered_temp, p=2, dim=-1)          # [B, T, H]
            steer_norm = F.normalize(steer_vec.view(1, 1, -1), p=2, dim=-1)  # [1, 1, H]
            
            # Per-token cosine similarity: [B, T]
            per_token_cos_sim = torch.sum(hs_norm * steer_norm, dim=-1)  # [B, T]
            
            # Average across tokens: [B]
            cos_sim = per_token_cos_sim.mean(dim=1)                      # [B]
            #print(cos_sim)
            
            # Hard gating: layer activated when cosine similarity > 0.2 OR negative
            mask_bool = (cos_sim > 0.2)  # [B] bool
            
            # Gate cosine (for weights) - zero out inactive layers
            cosine_gated = cos_sim * mask_bool.float()  # [B]
            cosine_gated_list.append(cosine_gated)

            # Project only last position to logits and gate
            logits_last = self._lm_head(hs[:, -1:, :])                   # [B,1,V]
            logits_last = logits_last * mask_bool.float().unsqueeze(-1).unsqueeze(-1)  # [B,1,V]
            layer_last_logits.append(logits_last)

        if w_eff == 0:
            return outputs  # nothing to blend

        # [B, w_eff], [B, w_eff, 1, V]
        cos_stack = torch.stack(cosine_gated_list, dim=1)
        logits_stack = torch.stack(layer_last_logits, dim=1)

        # 4) Build weights (allow negative weights); weights automatically sum to 1
        alpha = (self.sla_gamma * cos_stack)                        # [B, w_eff] - removed clamp_min(0.0)
        alpha_sum = alpha.sum(dim=1, keepdim=True)                   # [B, 1]
        final_w = (1.0 - alpha_sum)                                 # [B, 1] - directly ensures sum = 1

        # 5) Weighted sum for last position only, then splice back into logits
        weighted_intermediate = (alpha.unsqueeze(-1).unsqueeze(-1) * logits_stack).sum(dim=1)  # [B,1,V]
        final_last = logits[:, -1:, :]                                                                     # [B,1,V]
        blended_last = weighted_intermediate + final_w.unsqueeze(-1) * final_last                          # [B,1,V]

        blended_logits = logits.clone()
        blended_logits[:, -1:, :] = blended_last

        #Debug (optional)
        if self.sla_debug:
            start_layer = L - 1 - w_eff
            print(f"\\n🎯 Cosine SLA (γ={self.sla_gamma}, w={w_eff}, seq_len={T}, batch={B}):")
            print(f"   Final layer {L-1} cos_sim(b0): {final_cos_sim[0].item():.4f}")
            for i in range(w_eff):
                layer_idx = start_layer + i
                cos_val = cos_stack[0, i].item()  # cosine value for b0
                weight_val = alpha[0, i].item()  # alpha weight for this layer
                print(f"   Layer {layer_idx}: cos_sim(b0)={cos_val:.4f}, weight={weight_val:.4f}")
            print(f"   α_sum(b0)={alpha_sum[0,0].item():.4f}, final_w(b0)={final_w[0,0].item():.4f}")
            print(f"   Logits shape: {blended_logits.shape}\\n")

        if not return_dict:
            # outputs = (logits, *rest) or (loss, logits, *rest)
            if isinstance(outputs, tuple):
                # (loss, logits, ...)
                if isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 0:
                    return (outputs[0], blended_logits) + outputs[2:]
                else:
                    return (blended_logits,) + outputs[1:]
            return outputs  # should not happen, but be safe

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=blended_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
