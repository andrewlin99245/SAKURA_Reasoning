#!/usr/bin/env python3
"""
Generative Task Evaluation for Large Audio-Language Models
Based on the UnderstandingSound paper methodology

This script evaluates LALMs on generative tasks (audio captioning) to identify object hallucination.
It uses the 5 prompts specified in the paper and generates captions that can later be analyzed
for ECHO and Cover metrics.
"""

import csv
import json
import argparse
import torch
import librosa
import os
import sys
import pandas as pd
from tqdm import tqdm

# Conditional imports for real models
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets package not available. Using local metadata only.")

# For real model implementations, uncomment and modify these imports:
# from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
# from transformers import LTUAudioProcessor, LTUAudioForConditionalGeneration
# from transformers import SalmoNNProcessor, SalmoNNForConditionalGeneration


class MockAudioLanguageModel:
    """
    Mock model for testing purposes. 
    Replace this with real model implementations in production.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Mock model initialized on device: {self.device}")
        
        # Mock responses that vary by prompt to simulate real model behavior
        self.base_responses = {
            "Describe the audio.": [
                "The audio contains sounds of people talking and some background noise.",
                "This audio features mechanical sounds with human voices in the background.",
                "I hear various environmental sounds and human activity.",
            ],
            "What do you hear?": [
                "I hear speech and ambient sounds in the background.",
                "There are mechanical noises and people speaking.",
                "I can hear voices, machinery, and environmental sounds.",
            ],
            "What can be inferred from the audio?": [
                "This appears to be a recording of human conversation with background activity.",
                "This seems to be from an indoor environment with people and machines.",
                "The audio suggests a workplace or public space setting.",
            ],
            "This is a sound of": [
                "human speech with background activity.",
                "people talking in an environment with mechanical sounds.",
                "voices and machinery in what appears to be a workspace.",
            ],
            "Generate audio caption:": [
                "Audio clip featuring people speaking with environmental sounds.",
                "Recording of human voices with mechanical background noise.",
                "Sound clip of conversation in an active environment.",
            ]
        }
        self.response_index = 0
    
    def process_audio_and_generate(self, audio_path, prompt_text):
        """Mock inference that returns varied responses"""
        if prompt_text in self.base_responses:
            responses = self.base_responses[prompt_text]
            response = responses[self.response_index % len(responses)]
            self.response_index += 1
            return response
        return "I hear various sounds in this audio clip."


def get_generative_task_prompt():
    """
    Get the generative task prompts as specified in the UnderstandingSound paper.
    These are the 5 prompts used to evaluate generative capabilities.
    """
    return [
        "Describe the audio.",
        "What do you hear?", 
        "What can be inferred from the audio?",
        "This is a sound of",
        "Generate audio caption:"
    ]


def load_model_and_processor(model_name="mock"):
    """
    Load the audio-language model and processor.
    
    Args:
        model_name: Name of the model to load. Use "mock" for testing.
        
    Returns:
        tuple: (model, processor) or (None, None) if loading fails
    """
    if model_name == "mock":
        return MockAudioLanguageModel(), None
    
    # Qwen2-Audio model loading
    if "qwen" in model_name.lower() and "audio" in model_name.lower():
        try:
            from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
            print(f"Loading Qwen2-Audio model: {model_name}")
            
            processor = Qwen2AudioProcessor.from_pretrained(model_name)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True  # May be needed for some models
            )
            
            print(f"✓ Successfully loaded {model_name}")
            return model, processor
            
        except Exception as e:
            print(f"✗ Error loading Qwen2-Audio model: {e}")
            print("Falling back to mock model...")
            return MockAudioLanguageModel(), None
    
    # Regular Qwen models (text-only)
    elif "qwen" in model_name.lower():
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"Loading Qwen text model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"✓ Successfully loaded {model_name} (text-only)")
            return model, tokenizer
            
        except Exception as e:
            print(f"✗ Error loading Qwen text model: {e}")
            print("Falling back to mock model...")
            return MockAudioLanguageModel(), None
    
    # Add other model types here:
    # SALMONN, LTU-AS, etc.
    
    print(f"Model '{model_name}' not implemented. Using mock model instead.")
    return MockAudioLanguageModel(), None


def preprocess_audio(audio_path, processor=None, target_sample_rate=16000):
    """
    Load and preprocess audio file for the model.
    
    Args:
        audio_path: Path to audio file
        processor: Model processor (if None, uses mock processing)
        target_sample_rate: Target sampling rate
        
    Returns:
        dict: Processed audio inputs or None if failed
    """
    try:
        # Verify audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None
            
        # For mock model, skip actual audio loading
        if processor is None:
            return {"mock": True, "path": audio_path}
        
        # Real audio preprocessing:
        try:
            audio, sr = librosa.load(audio_path, sr=target_sample_rate)
            
            # Process with model-specific processor
            inputs = processor.feature_extractor(
                audio, 
                sampling_rate=target_sample_rate, 
                return_tensors="pt"
            )
            return inputs
            
        except Exception as librosa_error:
            print(f"Audio loading error: {librosa_error}")
            return None
        
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None


def inference(audio_path, prompt_text, model, processor=None):
    """
    Perform inference on audio with given prompt.
    
    Args:
        audio_path: Path to audio file
        prompt_text: Text prompt for generation
        model: Loaded model
        processor: Model processor
        
    Returns:
        str: Generated caption/response
    """
    try:
        # Preprocess audio
        audio_inputs = preprocess_audio(audio_path, processor)
        if audio_inputs is None:
            return "Error: Could not process audio"
        
        # For mock model
        if isinstance(model, MockAudioLanguageModel):
            return model.process_audio_and_generate(audio_path, prompt_text)
        
        # Real Qwen2-Audio model inference
        if hasattr(processor, 'tokenizer'):  # Qwen2AudioProcessor
            try:
                # Load and process audio
                audio, sr = librosa.load(audio_path, sr=16000)
                
                # Prepare inputs using the processor
                inputs = processor(
                    audios=audio,
                    text=prompt_text,
                    return_tensors="pt",
                    sampling_rate=16000
                )
                
                # Move to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Generate response with paper's parameters
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=1.0,     # Paper uses temperature=1.0 for sample decoding
                        top_p=0.9,          # Paper uses top_p=0.9
                        top_k=50,           # Paper uses top_k=50
                        do_sample=True,     # Sample decoding strategy
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # Decode response (skip the input tokens)
                input_length = inputs['input_ids'].shape[1]
                response = processor.tokenizer.decode(
                    outputs[0][input_length:], 
                    skip_special_tokens=True
                )
                
                return response.strip()
                
            except Exception as e:
                print(f"Error in Qwen2-Audio inference: {e}")
                return f"Error: {str(e)}"
        
        # Fallback for other model types
        return "Real model inference not fully implemented for this model type"
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return f"Error: {str(e)}"


def load_local_metadata(metadata_file):
    """Load metadata from local CSV file"""
    try:
        df = pd.read_csv(metadata_file)
        print(f"Loaded {len(df)} entries from local metadata")
        return df
    except Exception as e:
        print(f"Error loading local metadata: {e}")
        return None


def evaluate_with_local_data(args):
    """Evaluate using local metadata and audio files"""
    
    # Paths
    metadata_file = os.path.join(args.audio_root_dir, "../metadata/generative_data_metadata.csv")
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return None
    
    if not os.path.exists(args.audio_root_dir):
        print(f"Audio directory not found: {args.audio_root_dir}")
        return None
    
    # Load metadata
    df = load_local_metadata(metadata_file)
    if df is None:
        return None
    
    # Load model
    print("Loading model and processor...")
    model, processor = load_model_and_processor(args.model_name)
    if model is None:
        print("Failed to load model. Exiting.")
        return None
    
    # Get prompts
    prompts = get_generative_task_prompt()
    
    # Results storage
    all_results = {}
    
    # Determine number of samples to process
    total_samples = len(df)
    if args.max_samples and args.max_samples < total_samples:
        total_samples = args.max_samples
        print(f"Limited to {total_samples} samples for testing")
    
    processed_count = 0
    
    for index, row in tqdm(df.iterrows(), total=total_samples, desc="Processing audio samples"):
        if processed_count >= total_samples:
            break
            
        try:
            # Extract data from metadata
            youtube_id = row['youtube_id']
            
            # Parse caption and label (they might be string representations of lists)
            caption = eval(row['caption']) if isinstance(row['caption'], str) else row['caption']
            label = eval(row['label']) if isinstance(row['label'], str) else row['label']
            
            # Construct audio path
            audio_filename = f"{youtube_id}.wav"
            audio_path = os.path.join(args.audio_root_dir, audio_filename)
            
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue
            
            # Initialize results for this audio
            all_results[youtube_id] = {}
            
            # Test each prompt
            for prompt_text in prompts:
                # Perform inference
                response = inference(audio_path, prompt_text, model, processor)
                
                # Store results
                all_results[youtube_id][prompt_text] = {
                    "prediction": response,
                    "caption": caption,
                    "label": label,
                    "task": "generative",
                    "audio_path": audio_path
                }
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    print(f"Processed {processed_count} audio samples")
    return all_results


def evaluate_with_hf_dataset(args):
    """Evaluate using Hugging Face dataset"""
    
    if not DATASETS_AVAILABLE:
        print("datasets package not available. Cannot use Hugging Face dataset.")
        return None
    
    # Load dataset
    print("Loading Hugging Face dataset...")
    try:
        dataset = load_dataset(args.dataset_name, split="test")
        dataset_length = len(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    print(f"Dataset loaded. Total samples: {dataset_length}")
    
    # Load model
    print("Loading model and processor...")
    model, processor = load_model_and_processor(args.model_name)
    if model is None:
        print("Failed to load model. Exiting.")
        return None
    
    # Limit samples if specified
    if args.max_samples and args.max_samples < dataset_length:
        dataset_length = args.max_samples
        print(f"Limited to {dataset_length} samples for testing")
    
    # Get prompts
    prompts = get_generative_task_prompt()
    
    # Results storage
    all_results = {}
    
    for index in tqdm(range(dataset_length), desc="Processing audio samples"):
        try:
            # Get audio path from dataset
            audio_rel_path = dataset['audio'][index]['path']
            audio_abs_path = os.path.join(args.audio_root_dir, audio_rel_path)
            
            if not os.path.exists(audio_abs_path):
                print(f"Audio file not found: {audio_abs_path}")
                continue
            
            # Get audio index for results
            audio_index = audio_rel_path.replace(".wav", "")
            all_results[audio_index] = {}
            
            # Get ground truth data
            ground_truth_caption = dataset['caption'][index]
            ground_truth_label = dataset['label'][index]
            
            # Test each prompt
            for prompt_text in prompts:
                response = inference(audio_abs_path, prompt_text, model, processor)
                
                all_results[audio_index][prompt_text] = {
                    "prediction": response,
                    "caption": ground_truth_caption,
                    "label": ground_truth_label,
                    "task": "generative",
                }
        
        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            continue
    
    return all_results


def main(args):
    """Main evaluation function"""
    
    print("="*60)
    print("Generative Task Evaluation for Audio-Language Models")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Audio root: {args.audio_root_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print("="*60)
    
    # Choose evaluation method
    if args.use_local_metadata:
        print("Using local metadata and audio files...")
        results = evaluate_with_local_data(args)
    else:
        print("Using Hugging Face dataset...")
        results = evaluate_with_hf_dataset(args)
        
        # Fallback to local if HF dataset fails
        if results is None:
            print("Falling back to local metadata...")
            args.use_local_metadata = True
            results = evaluate_with_local_data(args)
    
    if results is None:
        print("No results generated")
        return
    
    # Save results
    output_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generative_evaluation_results.json")
    
    print(f"Saving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation completed! Processed {len(results)} audio samples")
    
    # Print sample results
    if results:
        print("\nSample result:")
        first_audio = list(results.keys())[0]
        first_prompt = list(results[first_audio].keys())[0]
        sample = results[first_audio][first_prompt]
        print(f"Audio: {first_audio}")
        print(f"Prompt: {first_prompt}")
        print(f"Prediction: {sample['prediction']}")
        print(f"Ground truth labels: {sample['label']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generative Task Evaluation for Audio-Language Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, 
                       help="Hugging Face dataset name", 
                       default="kuanhuggingface/audiocaps_hallucination")
    parser.add_argument("--audio_root_dir", type=str, 
                       help="Audio root directory", 
                       default="./understanding_sound_data/audio")
    parser.add_argument("--use_local_metadata", action="store_true",
                       help="Use local metadata instead of HF dataset")
    
    # Model arguments
    parser.add_argument("--model_name", type=str,
                       help="Model name/path for evaluation",
                       default="mock")
    
    # Output arguments
    parser.add_argument("--output_path", type=str, 
                       help="Output directory path", 
                       default="./evaluation_results")
    
    # Testing arguments
    parser.add_argument("--max_samples", type=int,
                       help="Maximum number of samples to process",
                       default=None)
    
    args = parser.parse_args()
    
    try:
        results = main(args)
        if results:
            print(f"\n✓ Successfully processed {len(results)} audio samples")
            print("Results saved and ready for ECHO/Cover analysis")
        else:
            print("\n✗ No results generated")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()