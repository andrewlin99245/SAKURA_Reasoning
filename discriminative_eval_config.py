"""
Configuration for discriminative evaluation tasks based on the Understanding Sound paper.
This configuration implements the discriminative evaluation setup described in the paper.
"""

# Model configuration
MODEL_CONFIG = {
    "model_path": "Qwen/Qwen2-Audio-7B-Instruct",
    "torch_dtype": "float16",
    "device_map": "auto",
    "max_new_tokens": 10,  # Short answers for Yes/No
    "temperature": 1.0,
    "top_p": 0.9,
    "do_sample": False,  # Deterministic for evaluation
}

# SLA configuration (as used in the existing codebase)
SLA_CONFIG = {
    "gamma": 0.0,
    "w": 4,
}

# Dataset configuration
DATASET_CONFIG = {
    "dataset_name": "kuanhuggingface/AudioHallucination_AudioCaps-Random-v2",
    "audio_root_dir": "./understanding_sound_data/audio",
    "sampling_strategies": ["random", "popular", "adversarial"],
}

# Evaluation prompts (based on paper's methodology)
DISCRIMINATIVE_PROMPTS = [
    "Is there a sound of {object} in the audio?",
    "Does the audio contain the sound of {object}?",
    "Have you noticed the sound of {object}?", 
    "Can you hear the sound of {object}?",
    "Can you detect the sound of {object}?",
]

# System prompt to encourage Yes/No answers
SYSTEM_PROMPT = "You are a helpful assistant. Answer with only 'Yes' or 'No' to the question about the audio content."

# Prompt engineering prefixes (based on Table 3 in the paper)
PROMPT_PREFIXES = {
    "P1": "Listen.",
    "P2": "Listen closely to the audio before answering the following question.",
    "P3": "Carefully consider the question before providing your answer.",
    "P4": "Listen closely to the audio and carefully consider the question before providing your answer.",
    "P5": "Focus and answer the following question.",
    "P6": "Focus on the given audio and answer the following question.",
    "P7": "Focus on the question and provide the answer.",
    "P8": "Focus on the given audio and the question and provide the answer.",
}

# Output configuration
OUTPUT_CONFIG = {
    "csv_columns": ["entry_id", "audio_index", "label", "response"],
    "default_output_path": "./evaluation_result.csv",
}