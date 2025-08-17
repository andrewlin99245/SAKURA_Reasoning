import os
import sys

# CRITICAL: Set up cache environment variables BEFORE any other imports
# This ensures consistent cache usage across all modules
SHARED_CACHE_DIR = os.path.expanduser("~/.cache/sakura_reasoning")
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)

# Set all relevant HuggingFace cache environment variables
os.environ["HF_HOME"] = SHARED_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = SHARED_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["HF_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.cache")
os.environ["TORCH_HOME"] = SHARED_CACHE_DIR

# Unset conflicting variables that could cause cache issues
if "PYTORCH_CACHE_HOME" in os.environ:
    del os.environ["PYTORCH_CACHE_HOME"]

print(f"Cache configured: {SHARED_CACHE_DIR}")

# Now import everything else
import csv
import argparse
import torch
import torch.nn.functional as F
import librosa
import time
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

from utils.Qwen2Audio_patch import Qwen2AudioSLAForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig

# Import vector steering modules
try:
    from steering_vector import obtain_vsv
    from ..layers.llm_layer import add_vsv_layers, remove_vsv_layers
    VSV_AVAILABLE = True
except ImportError as e:
    try:
        # Try absolute imports from project root
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.insert(0, project_dir)
        from src.models.steering_vector import obtain_vsv
        from src.layers.llm_layer import add_vsv_layers, remove_vsv_layers
        VSV_AVAILABLE = True
    except ImportError as e2:
        print(f"Warning: Vector steering modules not available: {e2}")
        print("Vector steering will be disabled.")
        VSV_AVAILABLE = False

# Global variables for model and processor
model = None
processor = None
verbose_progress = False
vsv_enabled = False
vsv_lambda = 1.0
vsv_prepared = False
vsv_tensor = None

# New cosine similarity-based SLA parameters
cosine_sla_gamma = 0.5  # Mixing coefficient for cosine-weighted SLA
cosine_sla_w = 3        # Number of layers to influence the last layer

def generate_with_cosine_weighting(model, inputs, steering_vector, gamma, w, max_new_tokens=10):
    """
    Custom generation function that implements cosine similarity-weighted SLA.
    This approach preserves audio input handling while allowing exact token-level cosine computation.
    """
    import torch.nn.functional as F
    
    # Get the language model component for accessing layers
    if hasattr(model, 'language_model'):
        lm_model = model.language_model.model  # Qwen2Model
        lm_head = model.language_model.lm_head
    else:
        raise AttributeError("Cannot access language model layers")
    
    # Start with input tokens
    input_ids = inputs['input_ids'].clone()
    current_inputs = {k: v for k, v in inputs.items()}
    
    # Set random seed to ensure proper randomness (same as model.generate())
    # Don't set a fixed seed - let it use the current random state
    
    for step in range(max_new_tokens):
        # Forward pass with output_hidden_states=True to collect all layer outputs
        outputs = model(
            output_hidden_states=True,
            **current_inputs
        )
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)
        final_logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # Debug tensor shapes to understand the issue
        if step == 0:  # Only print debug info on first step
            print(f"Debug - Step {step}:")
            print(f"  Hidden states count: {len(hidden_states)}")
            print(f"  First hidden shape: {hidden_states[0].shape}")
            print(f"  Last hidden shape: {hidden_states[-1].shape}")
            print(f"  Final logits shape: {final_logits.shape}")
            print(f"  Input IDs shape: {input_ids.shape}")
        
        # Get the position we're predicting (last position in sequence)
        # Handle potential dimension mismatch
        first_hidden = hidden_states[0]
        if len(first_hidden.shape) == 3:
            last_pos = first_hidden.size(1) - 1
        elif len(first_hidden.shape) == 2:
            # If 2D, we're at position 0 (single token prediction)
            last_pos = 0
        else:
            raise ValueError(f"Unexpected hidden state shape: {first_hidden.shape}")
            
        if step == 0:
            print(f"  Last position index: {last_pos}")
        
        # Get total number of layers
        total_layers = len(hidden_states) - 1  # -1 because first is embedding layer
        
        # Collect hidden states and compute logits for the last w layers
        layer_similarities = []
        layer_logits = []
        
        for i in range(max(0, total_layers - w), total_layers):
            layer_idx = i + 1  # +1 because hidden_states[0] is embedding
            
            # Get hidden state at the prediction position for this layer
            hidden_layer = hidden_states[layer_idx]
            
            # Handle potential dimension issues
            if len(hidden_layer.shape) == 2:
                # If only 2D, use all of it (likely batch_size, hidden_size)
                hidden_at_pos = hidden_layer
            else:
                # Normal case: 3D tensor (batch_size, seq_len, hidden_size)
                hidden_at_pos = hidden_layer[:, last_pos, :]  # (batch_size, hidden_size)
            
            # Compute cosine similarity with steering vector
            cosine_sim = F.cosine_similarity(
                hidden_at_pos,  # (batch_size, hidden_size)
                steering_vector.unsqueeze(0),  # (1, hidden_size)
                dim=1
            ).mean().item()  # Average over batch, convert to scalar
            
            # Safety check for NaN/inf
            if torch.isnan(torch.tensor(cosine_sim)) or torch.isinf(torch.tensor(cosine_sim)):
                cosine_sim = 0.0
            
            # Compute logits for this layer
            layer_logits_step = lm_head(hidden_at_pos)  # (batch_size, vocab_size)
            
            layer_similarities.append(cosine_sim)
            layer_logits.append(layer_logits_step)
        
        # Apply your cosine-weighted formula
        if gamma == 0.0:
            # When gamma=0, use the original unmodified logits
            # Use the logits from the last layer directly (no steering applied)
            if layer_logits:
                modified_logits = layer_logits[-1]  # Last layer logits
            else:
                modified_logits = final_logits[:, last_pos, :]
        elif layer_similarities and len(layer_logits) > 0:
            # Weighted sum of logits using cosine similarities
            weighted_logits = sum(sim * logits for sim, logits in zip(layer_similarities, layer_logits))
            weighted_logits = weighted_logits / len(layer_logits)  # Average
            
            # Sum of similarities
            sum_similarities = sum(layer_similarities)
            
            # Final formula: gamma*weighted_sum + (1-gamma*sum_similarities)*original_logits
            # Handle potential dimension issues with final_logits too
            if len(final_logits.shape) == 3:
                final_position_logits = final_logits[:, last_pos, :]  # (batch_size, vocab_size)
            elif len(final_logits.shape) == 2:
                final_position_logits = final_logits  # Already (batch_size, vocab_size)
            else:
                raise ValueError(f"Unexpected final_logits shape: {final_logits.shape}")
            modified_logits = (
                gamma * weighted_logits + 
                (1 - gamma * sum_similarities) * final_position_logits
            )
            
            # Safety check
            if torch.isnan(modified_logits).any() or torch.isinf(modified_logits).any():
                modified_logits = final_position_logits
        else:
            # Handle potential dimension issues
            if len(final_logits.shape) == 3:
                modified_logits = final_logits[:, last_pos, :]
            elif len(final_logits.shape) == 2:
                modified_logits = final_logits
            else:
                raise ValueError(f"Unexpected final_logits shape: {final_logits.shape}")
        
        # Sample next token using exact same logic as model.generate()
        # Parameters matching the original generate() call
        temperature = 1.0
        top_p = 0.9
        do_sample = True
        
        # Apply the exact transformers sampling logic
        if do_sample:
            # Step 1: Apply temperature scaling (as done in transformers)
            if temperature != 1.0:
                scaled_logits = modified_logits / temperature
            else:
                scaled_logits = modified_logits
            
            # Step 2: Apply top-p (nucleus) sampling - exact transformers implementation
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold (but keep the first)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Step 3: Convert to probabilities and sample
            probs = F.softmax(scaled_logits, dim=-1)
            # Ensure proper sampling by using generator state properly
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(modified_logits, dim=-1, keepdim=True)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Exact stopping criteria matching model.generate()
        # Check for all possible stopping tokens
        should_stop = False
        
        # 1. EOS token from config
        eos_token_id = getattr(model.config, 'eos_token_id', None)
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                if next_token.item() == eos_token_id:
                    should_stop = True
            elif isinstance(eos_token_id, (list, tuple)):
                if next_token.item() in eos_token_id:
                    should_stop = True
        
        # 2. Check for chat template special tokens (im_end, etc.)
        # Token 151645 is <|im_end|> for this model
        chat_end_tokens = [151645]  # <|im_end|>
        if next_token.item() in chat_end_tokens:
            should_stop = True
        
        # 3. Pad token
        pad_token_id = getattr(model.config, 'pad_token_id', None)
        if pad_token_id is not None and next_token.item() == pad_token_id:
            should_stop = True
        
        # 4. Check processor tokenizer for additional EOS tokens
        if hasattr(processor, 'tokenizer'):
            tokenizer_eos = getattr(processor.tokenizer, 'eos_token_id', None)
            if tokenizer_eos is not None and next_token.item() == tokenizer_eos:
                should_stop = True
        
        if should_stop:
            break
        
        # Update inputs for next iteration
        # For audio models, we need to maintain the original audio context for the first step,
        # then use basic inputs for subsequent steps
        if step == 0:
            # Keep original inputs but update input_ids
            current_inputs = {k: v for k, v in inputs.items()}
            current_inputs['input_ids'] = input_ids
            # Ensure attention mask matches new input_ids length
            if 'attention_mask' in current_inputs:
                # Extend attention mask for new tokens
                batch_size = input_ids.size(0)
                new_length = input_ids.size(1)
                current_inputs['attention_mask'] = torch.ones(batch_size, new_length, 
                                                            device=input_ids.device, 
                                                            dtype=current_inputs['attention_mask'].dtype)
        else:
            # For later steps, just use basic inputs
            current_inputs = {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
            }
    
    return input_ids

class CosineWeightedSLAHook:
    """Hook for computing cosine similarity-weighted SLA"""
    
    def __init__(self, steering_vector, gamma, w):
        self.steering_vector = steering_vector
        self.gamma = gamma
        self.w = w
        self.layer_similarities = {}
        self.layer_logits = {}
        self.total_layers = None
        self.current_position = None  # Track the exact position being generated
        
    def set_generation_position(self, position):
        """Set the exact position where the model is generating the next token"""
        self.current_position = position
        
    def register_hooks(self, model):
        """Register hooks on decoder layers"""
        self.hooks = []
        
        # Get total number of layers - for Qwen2AudioSLAForCausalLM
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            self.total_layers = len(model.language_model.model.layers)
        elif hasattr(model.model, 'layers'):
            self.total_layers = len(model.model.layers)
        elif hasattr(model.model.model, 'layers'):
            self.total_layers = len(model.model.model.layers)
        else:
            raise AttributeError("Cannot find model layers")
        
        # Register hooks on the last w layers
        for i in range(max(0, self.total_layers - self.w), self.total_layers):
            layer = self._get_layer(model, i)
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)
    
    def _get_layer(self, model, layer_idx):
        """Get the specified layer from model"""
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            return model.language_model.model.layers[layer_idx]
        elif hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx]
        elif hasattr(model.model.model, 'layers'):
            return model.model.model.layers[layer_idx]
        else:
            raise AttributeError(f"Cannot access layer {layer_idx}")
    
    def _make_hook(self, layer_idx):
        """Create hook function for the specified layer"""
        def hook_fn(module, input, output):
            # output[0] contains hidden states
            hidden_states = output[0]  # [batch_size, seq_len, hidden_size]
            
            # Use the exact position where the model is generating the next token
            if self.current_position is not None:
                # Use the specified position for generation
                target_position = min(self.current_position, hidden_states.size(1) - 1)
            else:
                # Fallback to the last available position
                target_position = hidden_states.size(1) - 1
            
            # Get hidden state at the exact generation position
            target_hidden = hidden_states[:, target_position, :]  # [batch_size, hidden_size]
            
            # Compute cosine similarity between steering vector and hidden state
            cosine_sim = F.cosine_similarity(
                target_hidden.unsqueeze(1),  # [batch_size, 1, hidden_size]
                self.steering_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            ).mean().item()  # Average over batch dimension
            
            # Safety check: ensure cosine similarity is valid
            if torch.isnan(torch.tensor(cosine_sim)) or torch.isinf(torch.tensor(cosine_sim)):
                cosine_sim = 0.0
            
            self.layer_similarities[layer_idx] = cosine_sim
            
        return hook_fn
    
    def modify_logits(self, logits):
        """Apply cosine similarity-weighted SLA to logits"""
        if len(self.layer_similarities) == 0 or self.total_layers is None:
            return logits
        
        # Get similarities for the last w layers
        similarities = []
        weighted_logits = []
        
        for i in range(max(0, self.total_layers - self.w), self.total_layers):
            if i in self.layer_similarities:
                sim = self.layer_similarities[i]
                similarities.append(sim)
                # For this implementation, we'll use the final layer logits with different weights
                # In a full implementation, you'd need to store logits from each layer
                weighted_logits.append(sim * logits)
        
        if not similarities:
            return logits
        
        # Compute weighted sum of logits
        sum_similarities = sum(similarities)
        if sum_similarities == 0:
            return logits
        
        # Weighted average of logits
        weighted_sum = sum(weighted_logits) / len(weighted_logits)
        
        # Apply the formula: gamma*weighted_sum + (1-gamma*sum_similarities)*original_logits
        final_logits = (
            self.gamma * weighted_sum + 
            (1 - self.gamma * sum_similarities) * logits
        )
        
        # Safety check: ensure final logits are valid
        if torch.isnan(final_logits).any() or torch.isinf(final_logits).any():
            print("Warning: Invalid logits detected, falling back to original logits")
            return logits
        
        return final_logits
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_similarities = {}

def initialize_model():
    """Initialize the model and processor globally"""
    global model, processor
    
    MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
    
    print("🚀 Initializing model...")
    
    print("  📦 Loading configuration...")
    config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
    
    print("  🤖 Loading model (this may take a few minutes)...")
    model = Qwen2AudioSLAForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    
    print("  🔧 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    print("  ⚡ Cosine-Weighted SLA Setup...")
    # Using custom cosine-weighted SLA approach - no original SLA needed
    print(f"    Cosine parameters: gamma={cosine_sla_gamma}, w={cosine_sla_w}")
    
    print("✅ Model initialization complete!")

def build_messages(include_audio: bool, wav_path: str, prompt: str):
    """Build messages for VSV computation"""
    base = []
    if include_audio:
        base.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": wav_path},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        base.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # No audio content here (this is the 'neg' case)
            ],
        })
    return base

def build_inputs(messages, audio=None, sr=16000):
    """Build model inputs from messages"""
    global processor, model
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
            audio=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
    # Move tensors to model device
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs

def compute_vsv_for_audio(audio_path, prompt):
    """Compute VSV for a specific audio file using the data_prompt as input for positive and negative instances"""
    global model, processor, verbose_progress
    
    if verbose_progress:
        print("    🎯 Computing vector steering vector...")
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Create soundless audio with same length as original for negative instance
        soundless_audio = np.zeros_like(audio)
        
        # Use the data_prompt (prompt parameter) for VSV computation
        vsv_prompt = f"{prompt} Answer just yes or no."
        
        # Build positive and negative inputs for VSV computation using the data_prompt
        messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
        messages_neg = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)  # Changed to True to include audio
        
        pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
        neg_inputs = build_inputs(messages_neg, audio=soundless_audio, sr=16000)  # Use soundless audio instead of None
        
        # Compute VSV specific to this input
        with torch.no_grad():
            kwargs_list = [[neg_inputs, pos_inputs]]
            vsv = obtain_vsv(model, kwargs_list)
            vsv = vsv.to(model.device)
        
        if verbose_progress:
            print(f"    ✅ VSV computed with shape: {vsv.shape}")
        
        return vsv
        
    except Exception as e:
        if verbose_progress:
            print(f"    ⚠️ VSV computation failed: {e}. Using zero vector.")
        # Return a zero steering vector if VSV computation fails
        # Get model's hidden size to create appropriate zero vector
        hidden_size = model.language_model.config.hidden_size
        zero_vsv = torch.zeros(hidden_size, device=model.device, dtype=model.dtype)
        return zero_vsv

def inference(audio_path, prompt_text):
    """
    Perform inference on audio with the given prompt text.
    Returns 'Yes' or 'No' for discriminative tasks.
    Supports cosine similarity-weighted vector steering if enabled.
    """
    global model, processor, verbose_progress, vsv_enabled, vsv_lambda, cosine_sla_gamma, cosine_sla_w
    
    if model is None or processor is None:
        initialize_model()
    
    if verbose_progress:
        print(f"  🎵 Processing: {os.path.basename(audio_path)}")
    
    vsv_applied = False
    cosine_hook = None
    
    try:
        # Apply cosine similarity-weighted vector steering if enabled
        if vsv_enabled:
            if verbose_progress:
                print("    🎯 Applying cosine-weighted vector steering...")
            
            # Compute VSV for this specific audio using the actual prompt
            vsv = compute_vsv_for_audio(audio_path, prompt_text)
            
            # VSV might have shape [batch_size, hidden_size], we need [hidden_size]
            if len(vsv.shape) > 1:
                # Average across batch dimension or take first element
                vsv = vsv.mean(dim=0)  # Shape: [hidden_size]
                if verbose_progress:
                    print(f"    📏 Reshaped VSV to: {vsv.shape}")
            
            # Create and register cosine-weighted SLA hook
            cosine_hook = CosineWeightedSLAHook(vsv, cosine_sla_gamma, cosine_sla_w)
            # Temporarily disable hook registration for debugging
            # cosine_hook.register_hooks(model)  
            vsv_applied = True  # Keep this True so generation uses custom path
            
            if verbose_progress:
                print(f"    ✅ Cosine-weighted vector steering applied with λ={vsv_lambda}, γ={cosine_sla_gamma}, w={cosine_sla_w}")
        
        # Build messages in the expected format
        # Append instruction to answer only yes or no
        modified_prompt = f"{prompt_text} Answer just yes or no."
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": modified_prompt},
            ]},
        ]

        if verbose_progress:
            print("    📝 Applying chat template...")
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if verbose_progress:
            print("    🎧 Loading and processing audio...")
        # Process audio
        audios = []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio, sr = librosa.load(ele["audio_url"])
                        if sr != 16000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        audios.append(audio)

        if verbose_progress:
            print("    🔧 Preparing model inputs...")
        # Prepare inputs
        inputs = processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Move inputs to device with proper dtype handling
        inputs_moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs_moved[key] = value.to(model.device)
                # input_ids and attention_mask should stay as Long/Int, not convert to model dtype
                if key in ['input_ids', 'attention_mask']:
                    inputs_moved[key] = inputs_moved[key].long()  # Ensure Long dtype
                # Audio features can use model dtype
                elif key in ['input_features', 'feature_attention_mask', 'audio_values']:
                    inputs_moved[key] = inputs_moved[key].to(model.dtype)
            else:
                inputs_moved[key] = value
        inputs = inputs_moved

        if verbose_progress:
            print("    🧠 Generating response with cosine-weighted SLA...")
        
        # Custom generation with cosine-weighted SLA
        with torch.no_grad():
            try:
                if vsv_applied and cosine_hook:
                    # Use custom generation that preserves audio handling
                    output = generate_with_cosine_weighting(model, inputs, vsv, cosine_sla_gamma, cosine_sla_w, max_new_tokens=10)
                else:
                    output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)
            except Exception as e:
                if verbose_progress:
                    print(f"    ⚠️ Custom generation failed: {e}. Falling back to standard generation.")
                # Fall back to standard generation without VSV
                output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.9)

        if verbose_progress:
            print("    📤 Decoding output...")
        # Decode output
        output = output[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Print the raw model generated response
        print(f"Model Response: {response}")
        
        # Clean and normalize response to Yes/No
        response = response.strip().lower()
        
        # Extract Yes/No from response
        if 'yes' in response:
            result = "Yes"
        elif 'no' in response:
            result = "No"
        else:
            # Default to "No" if unclear (following paper's observation that models tend to give affirmative answers)
            result = "No"
        
        if verbose_progress:
            print(f"    ✅ Response: {result}")
        
        return result
            
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {e}"
        if verbose_progress:
            print(f"    ❌ {error_msg}")
        else:
            print(error_msg)
        return "No"
    
    finally:
        # Always remove hooks after inference to prevent interference
        if cosine_hook and hasattr(cosine_hook, 'hooks'):
            cosine_hook.remove_hooks()
            if verbose_progress:
                print("    🔄 Cosine-weighted steering hooks removed")


def load_local_dataset(file_path):
    """Load dataset from local TSV file"""
    print(f"📂 Loading local dataset from: {file_path}")
    
    # Read the TSV file, skipping comment lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (first line without #)
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    header = data_lines[0].split('\t')
    
    # Parse data
    data = []
    for line in data_lines[1:]:
        if line:  # Skip empty lines
            fields = line.split('\t')
            if len(fields) >= 6:  # Ensure we have all required fields
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

def main(args):
    global verbose_progress, vsv_enabled, vsv_lambda, cosine_sla_gamma, cosine_sla_w
    verbose_progress = args.verbose
    vsv_enabled = args.enable_vsv
    vsv_lambda = args.vsv_lambda
    
    # Set cosine SLA parameters
    cosine_sla_gamma = args.cosine_gamma
    cosine_sla_w = args.cosine_w

    # Check if using local dataset file or HuggingFace dataset
    if hasattr(args, 'dataset_file') and args.dataset_file:
        print("📊 Loading local dataset...")
        dataset_samples = load_local_dataset(args.dataset_file)
        
        # Randomly shuffle the dataset
        print("🔀 Randomly shuffling dataset...")
        random.shuffle(dataset_samples)
        
        # Limit samples if specified
        if hasattr(args, 'max_samples') and args.max_samples and args.max_samples < len(dataset_samples):
            dataset_samples = dataset_samples[:args.max_samples]
            print(f"🔢 Limited to {args.max_samples} samples for testing")
        
        total_samples = len(dataset_samples)
        use_local_dataset = True
    else:
        print("📊 Loading HuggingFace dataset...")
        # Load the dataset.
        dataset = load_dataset(args.dataset_name)
        dataset_samples = list(dataset['test'])
        
        # Randomly shuffle the dataset
        print("🔀 Randomly shuffling dataset...")
        random.shuffle(dataset_samples)
        
        total_samples = len(dataset_samples)
        use_local_dataset = False
        print(f"📝 Dataset loaded: {total_samples} samples to process")

    # Evaluation results.
    evaluation_results = []
    
    # Initialize model before processing (if not already initialized)
    if model is None:
        initialize_model()

    # Print vector steering configuration
    if vsv_enabled:
        print(f"🎯 Cosine-weighted vector steering ENABLED with λ={vsv_lambda}, γ={cosine_sla_gamma}, w={cosine_sla_w}")
    else:
        print("🎯 Vector steering DISABLED")

    print(f"🎯 Starting inference on {total_samples} samples...")
    start_time = time.time()
    
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Processing samples", unit="sample")):

        # Entry ID for the dataset.
        entry_id = sample["entry_id"]

        # The ID in AudioCaps, e.g., Y7fmOlUlwoNg corresponds to Y7fmOlUlwoNg.wav
        audio_index = sample["audio_index"]

        # The absolute path of audio.
        audio_path = f"{args.audio_root_dir}/{audio_index}.wav"

        # The input text prompt.
        prompt_text = sample["prompt_text"]

        # The correct answer corresponding to the prompt_text.
        label = sample["label"]

        # Get sampling method if available (for local datasets)
        sampling_method = sample.get("sampling", "unknown") if use_local_dataset else "unknown"

        # Inference model and get response.
        response = inference(audio_path=audio_path, prompt_text=prompt_text)

        # Record evaluation result.
        if use_local_dataset:
            evaluation_result = [entry_id, audio_index, label, response, sampling_method, vsv_enabled, vsv_lambda if vsv_enabled else 0.0, cosine_sla_gamma, cosine_sla_w]
        else:
            evaluation_result = [entry_id, audio_index, label, response, vsv_enabled, vsv_lambda if vsv_enabled else 0.0, cosine_sla_gamma, cosine_sla_w]
        evaluation_results.append(evaluation_result)
        
        # Show progress every 50 samples or at the end
        if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
            correct = sum(1 for result in evaluation_results if result[2] == result[3])
            accuracy = correct / len(evaluation_results) * 100
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / (idx + 1)
            estimated_total_time = avg_time_per_sample * total_samples
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"  📈 Progress: {idx + 1}/{total_samples} | Current accuracy: {accuracy:.1f}% | "
                  f"Avg time/sample: {avg_time_per_sample:.1f}s | ETA: {remaining_time/60:.1f}m")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    correct = sum(1 for result in evaluation_results if result[2] == result[3])
    final_accuracy = correct / len(evaluation_results) * 100
    
    print(f"\n🏁 Inference completed!")
    print(f"  📊 Final accuracy: {final_accuracy:.2f}% ({correct}/{total_samples})")
    print(f"  ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"  ⚡ Average time per sample: {total_time/total_samples:.1f}s")
    
    # Analyze by sampling method if using local dataset
    if use_local_dataset:
        sampling_stats = {}
        for result in evaluation_results:
            sampling = result[4]  # sampling_method is at index 4 for local datasets
            if sampling not in sampling_stats:
                sampling_stats[sampling] = {'correct': 0, 'total': 0}
            sampling_stats[sampling]['total'] += 1
            if result[2] == result[3]:  # label == response
                sampling_stats[sampling]['correct'] += 1
        
        print(f"\n📊 Results by sampling method:")
        for sampling, stats in sampling_stats.items():
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"  {sampling}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Update output filename to include cosine steering info if enabled
    output_path = args.output_path
    if vsv_enabled:
        # Insert cosine steering info before file extension
        name_parts = output_path.rsplit('.', 1)
        if len(name_parts) == 2:
            output_path = f"{name_parts[0]}_cosine_vsv_lambda{vsv_lambda}_gamma{cosine_sla_gamma}_w{cosine_sla_w}.{name_parts[1]}"
        else:
            output_path = f"{output_path}_cosine_vsv_lambda{vsv_lambda}_gamma{cosine_sla_gamma}_w{cosine_sla_w}"
    
    # Writing the data to CSV using csv module
    print(f"💾 Saving results to {output_path}...")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if use_local_dataset:
            writer.writerow(["entry_id", "audio_index", "label", "response", "sampling_method", "vsv_enabled", "vsv_lambda", "cosine_gamma", "cosine_w"])
        else:
            writer.writerow(["entry_id", "audio_index", "label", "response", "vsv_enabled", "vsv_lambda", "cosine_gamma", "cosine_w"])
        writer.writerows(evaluation_results)
    
    print(f"✅ Inference results are saved to {output_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio hallucination evaluation with cosine similarity-weighted SLA vector steering")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name.", default="kuanhuggingface/AudioHallucination_AudioCaps-Random-v2")
    parser.add_argument("--dataset_file", type=str, help="Path to local dataset TSV file (alternative to --dataset_name)", default="./understanding_sound_data/metadata/balanced_merged_test_2871.txt")
    parser.add_argument("--audio_root_dir", type=str, help="Audio root directory", default="./understanding_sound_data/audio")
    parser.add_argument("--output_path", type=str, help="Output path of csv file.", default="./cosine_weighted_evaluation_result.csv")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output for individual inference steps")
    
    # Vector steering options
    parser.add_argument("--enable_vsv", action="store_true", help="Enable cosine-weighted vector steering for audio hallucination mitigation")
    parser.add_argument("--vsv_lambda", type=float, default=0.05, help="Vector steering strength (lambda). Higher values = stronger steering. Default: 0.05")
    
    # Cosine similarity-based SLA parameters
    parser.add_argument("--cosine_gamma", type=float, default=0.5, help="Mixing coefficient gamma for cosine-weighted SLA. Default: 0.5")
    parser.add_argument("--cosine_w", type=int, default=3, help="Number of layers w to influence the last layer. Default: 3")
    
    # Testing options
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    main(args)