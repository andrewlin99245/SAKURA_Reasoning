Understanding Sound Paper - Discriminative Tasks Data
==================================================

This folder contains the filtered AudioCaps data used in the paper:
'Understanding Sounds, Missing the Questions: The Challenge of Object
Hallucination in Large Audio-Language Models'

Structure:
- audio/: Contains 957 unique audio files (.wav) used in discriminative tasks
- metadata/: Contains text files with question datasets from HuggingFace
  - random_test.txt: Random sampling questions
  - popular_test.txt: Popular sampling questions
  - adversarial_test.txt: Adversarial sampling questions

Each metadata file contains:
- entry_id: Unique identifier for the question
- audio_index: Audio file identifier (matches .wav filename)
- prompt_text: The discriminative question text
- object: The object being asked about
- attribute: 'positive' or 'negative' sample
- label: Expected answer ('Yes' or 'No')
- sampling: Sampling strategy used

Total questions per strategy:
- Random: ~30,220 questions
- Popular: ~31,376 questions
- Adversarial: ~31,047 questions
