import inspect
from typing import Optional, List, Union, Any, Tuple

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5OmniForConditionalGeneration


def _filter_kwargs_for_fn(fn, kwargs: dict):
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}


class Qwen2_5OmniSLAForCausalLM(Qwen2_5OmniForConditionalGeneration):
    def __init__(self, config):
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
        # call parent first with only what it understands
        parent_inputs = _filter_kwargs_for_fn(
            super().prepare_inputs_for_generation,
            dict(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            ),
        )
        model_inputs = super().prepare_inputs_for_generation(**parent_inputs)

        # ensure multimodal-related tensors survive the 1st step (if present)
        for k in ["audio_values", "audio_attention_mask", "image_grid_thw", 
                  "video_grid_thw", "image_values", "video_values"]:
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
        # Declare the common multimodal kwargs
        audio_values: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_values: Optional[torch.FloatTensor] = None,
        video_values: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Since the parent class forward is not implemented, we need to 
        # implement our own forward that uses the thinker subcomponent
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Filter arguments for the thinker component
        thinker_kwargs = _filter_kwargs_for_fn(
            self.thinker.forward,
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
                audio_values=audio_values,
                audio_attention_mask=audio_attention_mask,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_values=image_values,
                video_values=video_values,
            ),
        )
        
        # Delegate to the thinker component which has the actual implementation
        return self.thinker(**thinker_kwargs)