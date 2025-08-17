import inspect
from typing import Optional, List, Union, Any, Tuple

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioForConditionalGeneration,
)


def _filter_kwargs_for_fn(fn, kwargs: dict):
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}


class Qwen2AudioSLAForCausalLM(Qwen2AudioForConditionalGeneration):
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__(config)
        # SLA knobs
        self.sla_enable = False
        self.sla_gamma = 0.3
        self.sla_w = 5
        self._lm_head = self.get_output_embeddings()

    # ---------------- public API ----------------
    def enable_sla(self, gamma: float = 0.3, w: int = 5):
        self.sla_enable = True
        self.sla_gamma = gamma
        self.sla_w = w

    def disable_sla(self):
        self.sla_enable = False

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

        need_hidden = self.sla_enable and self.sla_gamma > 0.0 and self.sla_w > 0
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
            return outputs

        logits = outputs.logits
        hidden_states = outputs.hidden_states  # tuple length = L+1

        L = len(hidden_states) - 1
        if L < 2:
            return outputs

        w = min(self.sla_w, L - 1)
        if w <= 0:
            return outputs

        hs_stack = []
        for li in range(L - w, L):
            hs_stack.append(hidden_states[li + 1])  # ignore embedding layer at 0

        hs_cat = torch.stack(hs_stack, dim=0)  # [w, B, T, H]
        aug_logits_each = self._lm_head(hs_cat)  # [w, B, T, V]
        aug_logits = aug_logits_each.mean(dim=0)  # [B, T, V]

        gamma = self.sla_gamma
        blended_logits = (1.0 - gamma) * logits + gamma * aug_logits

        if not return_dict:
            # outputs = (logits, *rest) or (loss, logits, *rest)
            if isinstance(outputs, tuple):
                # figure out if first element is loss
                if isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 0:
                    # (loss, logits, ...)
                    return (outputs[0], blended_logits) + outputs[2:]
                else:
                    # (logits, ...)
                    return (blended_logits,) + outputs[1:]
            return outputs  # should not happen, but be safe

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=blended_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )