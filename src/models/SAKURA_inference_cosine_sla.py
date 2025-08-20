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
import argparse
import torch
import librosa
import time
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoProcessor

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

from utils.Qwen2Audio_cosine_SLA_patch import Qwen2AudioCosineSLAForCausalLM
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
vsv_lambda = 0.05
# Cosine SLA parameters
cosine_sla_enabled = False
cosine_sla_gamma = 0.25
cosine_sla_w = 4
# SAKURA dataset configuration
SAKURA_DATA_DIR = "/home/andrew99245/SAKURA_Reasoning/sdata"
MAX_SAMPLE = -1

def initialize_model():
    """Initialize the model and processor globally"""
    global model, processor
    
    MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
    
    print("🚀 Initializing model...")
    
    print("  📦 Loading configuration...")
    config = Qwen2AudioConfig.from_pretrained(MODEL_PATH)
    
    print("  🤖 Loading model (this may take a few minutes)...")
    model = Qwen2AudioCosineSLAForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    
    print("  🔧 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    print("  ⚡ SLA ready (will be enabled when cosine_sla_enabled=True)")
    
    print("✅ Model initialization complete!")

def build_messages(include_audio: bool, wav_path: str, prompt: str):
    """Build messages for VSV computation"""
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
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
    """Compute VSV for a specific audio file using descriptive prompt for positive and negative instances"""
    global model, processor, verbose_progress
    
    if verbose_progress:
        print("    🎯 Computing vector steering vector...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    soundless_audio = np.zeros_like(audio)
    vsv_prompt = prompt
    
    # Build positive and negative inputs for VSV computation
    messages_pos = build_messages(include_audio=True,  wav_path=audio_path, prompt=vsv_prompt)
    messages_neg = build_messages(include_audio=True, wav_path=audio_path, prompt=vsv_prompt)
    
    pos_inputs = build_inputs(messages_pos, audio=audio, sr=16000)
    neg_inputs = build_inputs(messages_neg, audio=soundless_audio, sr=16000)
    

    # Compute VSV specific to this input
    with torch.no_grad():
        kwargs_list = [[neg_inputs, pos_inputs]]
        vsv = obtain_vsv(model, kwargs_list)
        vsv = vsv.to(model.device)
    
    if verbose_progress:
        print(f"    ✅ VSV computed with shape: {vsv.shape}")
    
    return vsv

def inference(audio_path, prompt_text):
    """
    Perform inference on audio with the given prompt text.
    Returns the full response from the model.
    Supports both vector steering and cosine SLA if enabled.
    """
    global model, processor, verbose_progress, vsv_enabled, vsv_lambda, cosine_sla_enabled, cosine_sla_gamma, cosine_sla_w
    
    if model is None or processor is None:
        initialize_model()
    
    if verbose_progress:
        print(f"  🎵 Processing: {os.path.basename(audio_path)}")
    
    vsv_applied = False
    vsv = None
    try:
        # Compute VSV if either vector steering or cosine SLA is enabled
        if vsv_enabled or cosine_sla_enabled:
            if verbose_progress:
                print("    🎯 Computing steering vector...")
            
            # Compute VSV for this specific audio
            vsv = compute_vsv_for_audio(audio_path, prompt_text)
            
            if verbose_progress:
                print(f"    ✅ Steering vector computed with shape: {vsv.shape}")
        
        # Apply vector steering if enabled
        if vsv_enabled and vsv is not None:
            if verbose_progress:
                print("    🎯 Applying vector steering...")
            
            # Inject VSV
            add_vsv_layers(model, vsv=vsv, lam=vsv_lambda, which_stack="decoder")
            vsv_applied = True
            
            if verbose_progress:
                print(f"    ✅ Vector steering applied with λ={vsv_lambda}")
        
        # Set up cosine SLA if enabled
        if cosine_sla_enabled and vsv is not None:
            if verbose_progress:
                print("    🎯 Setting up cosine SLA...")
            
            # Set steering vectors in the model
            model.set_steering_vectors(vsv)
            
            # Enable cosine SLA
            model.enable_sla(gamma=cosine_sla_gamma, w=cosine_sla_w, debug=True)
            
            if verbose_progress:
                print(f"    ✅ Cosine SLA enabled with γ={cosine_sla_gamma}, w={cosine_sla_w}")
        elif not cosine_sla_enabled:
            # Disable cosine SLA if not enabled
            model.disable_sla()
        
        # Build messages in the expected format
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt_text},
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
        inputs = inputs.to(model.device).to(model.dtype)

        if verbose_progress:
            print("    🧠 Generating response...")
        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=0.9)

        if verbose_progress:
            print("    📤 Decoding output...")
        # Decode output
        output = output[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        if verbose_progress:
            print(f"    ✅ Response: {response}")
        
        return response
            
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {e}"
        if verbose_progress:
            print(f"    ❌ {error_msg}")
        else:
            print(error_msg)
        return "Error occurred during inference"
    
    finally:
        # Always remove VSV hooks after inference to prevent interference
        if vsv_applied:
            remove_vsv_layers(model, which_stack="decoder")
            if verbose_progress:
                print("    🔄 Vector steering hooks removed")

def load_sakura_dataset(subset):
    """Load SAKURA dataset for a specific subset"""
    print(f"📂 Loading SAKURA dataset subset: {subset}")
    
    csv_path = f"{SAKURA_DATA_DIR}/{subset}/metadata.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Loaded {len(df)} samples for {subset}")
    return df

def run_sakura_evaluation(gamma, w, lam, model_suffix):
    """Run evaluation on SAKURA dataset with specific parameters"""
    print(f"\n=== Running SAKURA evaluation with gamma={gamma}, w={w}, lam={lam} ===")
    
    global cosine_sla_enabled, cosine_sla_gamma, cosine_sla_w, vsv_enabled, vsv_lambda
    
    # Set global parameters
    cosine_sla_enabled = gamma > 0 or w > 0
    cosine_sla_gamma = gamma
    cosine_sla_w = w
    vsv_enabled = lam > 0
    vsv_lambda = lam
    
    # Initialize model if not already done
    if model is None:
        initialize_model()
    
    # Run evaluation on all SAKURA datasets
    os.makedirs("./results", exist_ok=True)
    
    for subset in ["Animal", "Emotion", "Gender", "Language"]:
        print(f"Processing {subset}...")
        df = load_sakura_dataset(subset)
        
        # Initialize result objects
        single_result_path = f"./results/{subset}_qwen2_audio_7b_cosine_{model_suffix}_single.json"
        single_result = {
            "attribute": subset,
            "type": "single",
            "model_config": {"gamma": gamma, "w": w, "lam": lam},
            "results": {}
        }
        multi_result_path = f"./results/{subset}_qwen2_audio_7b_cosine_{model_suffix}_multi.json"
        multi_result = {
            "attribute": subset,
            "type": "multi",
            "model_config": {"gamma": gamma, "w": w, "lam": lam},
            "results": {}
        }
        
        max_sample = len(df) if MAX_SAMPLE == -1 else MAX_SAMPLE
        
        for i in tqdm(range(max_sample), desc=f"{subset}"):
            audio_file = df.iloc[i]["file"]
            single_instruction = df.iloc[i]["single_instruction"]
            multi_instruction = df.iloc[i]["multi_instruction"]
            
            audio_path = f"{SAKURA_DATA_DIR}/{subset}/audio/{audio_file}"
            
            # Single instruction
            response = inference(audio_path=audio_path, prompt_text=single_instruction)
            print(f"Single - {audio_file}: {response}")
            single_result["results"][audio_file] = {
                "instruction": single_instruction,
                "response": response,
                "label": df.iloc[i]["attribute_label"]
            }
            
            # Multi instruction
            response = inference(audio_path=audio_path, prompt_text=multi_instruction)
            print(f"Multi - {audio_file}: {response}")
            multi_result["results"][audio_file] = {
                "instruction": multi_instruction,
                "response": response,
                "label": df.iloc[i]["attribute_label"]
            }
        
        # Save results
        with open(single_result_path, "w") as f:
            json.dump(single_result, f, indent=4, ensure_ascii=False)
        
        with open(multi_result_path, "w") as f:
            json.dump(multi_result, f, indent=4, ensure_ascii=False)
        
        print(f"Finished {subset}.")

def main(args):
    """Main function to run SAKURA evaluation"""
    global verbose_progress
    verbose_progress = args.verbose
    
    # Run evaluation with specified parameters
    run_sakura_evaluation(
        gamma=args.cosine_sla_gamma,
        w=args.cosine_sla_w,
        lam=args.vsv_lambda if args.enable_vsv else 0.0,
        model_suffix=args.model_suffix
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAKURA dataset evaluation with cosine SLA and vector steering")
    
    # Model and output options
    parser.add_argument("--model_suffix", type=str, default="cosine_vsv", help="Suffix for output filenames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose progress output")
    
    # Vector steering options
    parser.add_argument("--enable_vsv", action="store_true", help="Enable vector steering")
    parser.add_argument("--vsv_lambda", type=float, default=0.05, help="Vector steering strength (lambda)")
    
    # Cosine SLA options
    parser.add_argument("--cosine_sla_gamma", type=float, default=1, help="Cosine SLA gamma parameter")
    parser.add_argument("--cosine_sla_w", type=int, default=1, help="Cosine SLA w parameter (number of layers)")
    parser.add_argument("--cosine_sla_enabled", action="store_true", help="Enable cosine SLA")
    # Testing options
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum samples per subset (-1 for all)")
    args = parser.parse_args()
    
    # Update global MAX_SAMPLE
    MAX_SAMPLE = args.max_samples
    
    main(args)