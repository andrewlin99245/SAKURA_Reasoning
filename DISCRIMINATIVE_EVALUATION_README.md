# Discriminative Evaluation for Audio Hallucination Assessment

This implementation provides discriminative inference capabilities for evaluating Large Audio-Language Models (LALMs) on object hallucination tasks, based on the paper "Understanding Sounds, Missing the Questions: The Challenge of Object Hallucination in Large Audio-Language Models."

## Overview

The discriminative evaluation assesses whether LALMs can correctly identify the presence or absence of specific objects/sounds in audio clips by asking binary Yes/No questions.

## Key Features

- **Model Integration**: Uses Qwen2-Audio-7B-Instruct with SLA steering capabilities
- **Sampling Strategies**: Supports random, popular, and adversarial sampling as described in the paper
- **Prompt Engineering**: Implements the prompt prefixes that showed performance improvements
- **Evaluation Metrics**: Calculates accuracy, precision, recall, and F1 scores
- **Local Dataset Support**: Works with the provided AudioCaps dataset

## Files

### Core Implementation
- `scripts/runners/hal_inference.py` - Main inference function with SLA-enabled Qwen2-Audio model
- `discriminative_eval_config.py` - Configuration settings based on paper methodology
- `run_discriminative_evaluation.py` - Comprehensive evaluation script

### Test Files
- `test_discriminative_inference.py` - Simple test script for validation

## Usage

### Basic Evaluation

Run evaluation on a specific sampling strategy:

```bash
# Random sampling
python run_discriminative_evaluation.py \
    --metadata_file understanding_sound_data/metadata/random_test.txt \
    --audio_root_dir understanding_sound_data/audio \
    --output_path random_evaluation_results.csv

# Popular sampling  
python run_discriminative_evaluation.py \
    --metadata_file understanding_sound_data/metadata/popular_test.txt \
    --output_path popular_evaluation_results.csv

# Adversarial sampling
python run_discriminative_evaluation.py \
    --metadata_file understanding_sound_data/metadata/adversarial_test.txt \
    --output_path adversarial_evaluation_results.csv
```

### With Prompt Engineering

Apply prompt prefixes that showed improvement in the paper:

```bash
# Using P4 prefix (best performing)
python run_discriminative_evaluation.py \
    --metadata_file understanding_sound_data/metadata/random_test.txt \
    --prompt_prefix P4 \
    --output_path random_with_p4_results.csv

# Using P8 prefix  
python run_discriminative_evaluation.py \
    --metadata_file understanding_sound_data/metadata/random_test.txt \
    --prompt_prefix P8 \
    --output_path random_with_p8_results.csv
```

### Limited Testing

For quick testing with fewer samples:

```bash
python run_discriminative_evaluation.py \
    --metadata_file understanding_sound_data/metadata/random_test.txt \
    --max_samples 100 \
    --output_path test_results.csv
```

### Using the Original Template

The original template in `hal_inference.py` is now fully implemented and can be used directly:

```bash
python scripts/runners/hal_inference.py \
    --dataset_name kuanhuggingface/AudioHallucination_AudioCaps-Random-v2 \
    --audio_root_dir understanding_sound_data/audio \
    --output_path evaluation_result.csv
```

## Configuration

The implementation follows the paper's experimental setup:

### Model Settings
- **Model**: Qwen2-Audio-7B-Instruct
- **SLA**: Enabled with γ=0.0, w=4
- **Generation**: max_new_tokens=10, do_sample=False
- **Response Format**: Binary Yes/No answers

### Prompt Engineering Prefixes
Based on Table 3 in the paper:
- **P1**: "Listen."
- **P2**: "Listen closely to the audio before answering the following question."
- **P3**: "Carefully consider the question before providing your answer."
- **P4**: "Listen closely to the audio and carefully consider the question before providing your answer."
- **P5**: "Focus and answer the following question."
- **P6**: "Focus on the given audio and answer the following question."
- **P7**: "Focus on the question and provide the answer."
- **P8**: "Focus on the given audio and the question and provide the answer."

## Dataset Structure

The implementation expects metadata files with the following format:
```
entry_id    audio_index    prompt_text    object    attribute    label    sampling
0          Y7fmOlUlwoNg   Is there a sound of speech in the audio?    speech    positive    Yes    random
```

## Expected Results

Based on the paper's findings:
- LALMs tend to give affirmative answers (high Yes rate)
- Performance varies by sampling strategy: Random > Popular > Adversarial
- Prompt engineering (especially P4, P8) can improve F1 scores significantly
- Models show lower recall than precision on discriminative tasks

## Performance Notes

- Model loading takes time on first run
- Audio processing is performed at 16kHz sampling rate
- GPU memory usage depends on model precision (float16 recommended)
- Expected processing time: ~1-2 seconds per audio file

## Dependencies

Required packages:
- torch
- transformers
- librosa  
- datasets
- tqdm
- pandas (for extended analysis)

The implementation integrates with the existing codebase and uses the SLA steering implementation in `src/models/sla_steering.py`.