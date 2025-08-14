#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import defaultdict
import random

def classify_question_format(prompt_text):
    """Classify question format based on the prompt text"""
    if prompt_text.startswith("Is there a sound of"):
        return "Format1"
    elif prompt_text.startswith("Does the audio contain the sound of"):
        return "Format2"
    elif prompt_text.startswith("Have you noticed the sound of"):
        return "Format3"
    elif prompt_text.startswith("Can you hear the sound of"):
        return "Format4"
    elif prompt_text.startswith("Can you detect the sound of"):
        return "Format5"
    else:
        return "Unknown"

def load_and_process_dataset(filepath):
    """Load dataset and classify question formats"""
    # Read the file manually since it has comments at the top
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (starts with entry_id)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('entry_id'):
            header_idx = i
            break
    
    # Read data starting from header
    data_lines = lines[header_idx:]
    
    # Parse the data
    rows = []
    header = data_lines[0].strip().split('\t')
    
    for line in data_lines[1:]:
        if line.strip():  # Skip empty lines
            rows.append(line.strip().split('\t'))
    
    df = pd.DataFrame(rows, columns=header)
    
    # Add question format classification
    df['question_format'] = df['prompt_text'].apply(classify_question_format)
    
    return df

def create_balanced_dataset(df, target_size=957):
    """Create a balanced dataset with specified size and question format distribution"""
    
    # Get original format distribution
    format_counts = df['question_format'].value_counts()
    total_original = len(df)
    
    # Calculate target distribution (proportional to original)
    target_distribution = {}
    for fmt in format_counts.index:
        proportion = format_counts[fmt] / total_original
        target_distribution[fmt] = max(1, round(proportion * target_size))
    
    # Adjust to exactly target_size
    current_total = sum(target_distribution.values())
    if current_total != target_size:
        # Adjust the largest category
        largest_fmt = max(target_distribution.keys(), key=lambda x: target_distribution[x])
        target_distribution[largest_fmt] += (target_size - current_total)
    
    print(f"Target distribution for {target_size} samples:")
    for fmt in sorted(target_distribution.keys()):
        print(f"  {fmt}: {target_distribution[fmt]} ({target_distribution[fmt]/target_size*100:.1f}%)")
    
    # Get unique audio files
    unique_audios = df['audio_index'].unique()
    print(f"Total unique audio files: {len(unique_audios)}")
    
    if len(unique_audios) != target_size:
        print(f"Warning: Expected {target_size} unique audio files, found {len(unique_audios)}")
    
    # Create balanced dataset
    selected_samples = []
    
    # For each audio file, we need to select one question
    for audio_id in unique_audios:
        audio_samples = df[df['audio_index'] == audio_id]
        
        # Group by format and label for this audio
        format_label_groups = defaultdict(lambda: defaultdict(list))
        for _, row in audio_samples.iterrows():
            fmt = row['question_format']
            label = row['label']
            format_label_groups[fmt][label].append(row)
        
        # Select a format for this audio (based on availability and target distribution)
        available_formats = list(format_label_groups.keys())
        if available_formats:
            # Prefer formats we still need more of
            remaining_needs = {fmt: max(0, target_distribution.get(fmt, 0) - 
                             len([s for s in selected_samples if s['question_format'] == fmt])) 
                             for fmt in available_formats}
            
            # If we have remaining needs, pick from those, otherwise pick randomly
            formats_with_need = [fmt for fmt in available_formats if remaining_needs[fmt] > 0]
            if formats_with_need:
                selected_format = random.choice(formats_with_need)
            else:
                selected_format = random.choice(available_formats)
            
            # From the selected format, try to balance Yes/No if possible
            format_samples = format_label_groups[selected_format]
            
            # Count current Yes/No balance in selected samples
            current_yes = len([s for s in selected_samples if s['label'] == 'Yes'])
            current_no = len([s for s in selected_samples if s['label'] == 'No'])
            
            # Prefer the label we have fewer of
            if 'Yes' in format_samples and 'No' in format_samples:
                if current_yes <= current_no:
                    preferred_label = 'Yes'
                else:
                    preferred_label = 'No'
                selected_sample = random.choice(format_samples[preferred_label])
            elif 'Yes' in format_samples:
                selected_sample = random.choice(format_samples['Yes'])
            elif 'No' in format_samples:
                selected_sample = random.choice(format_samples['No'])
            else:
                # This shouldn't happen, but fallback
                selected_sample = audio_samples.iloc[0]
            
            selected_samples.append(selected_sample.to_dict())
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(selected_samples)
    
    # Print final statistics
    print(f"\nFinal dataset statistics:")
    print(f"Total samples: {len(result_df)}")
    print(f"Unique audio files: {result_df['audio_index'].nunique()}")
    
    print(f"\nQuestion format distribution:")
    final_format_counts = result_df['question_format'].value_counts()
    for fmt in sorted(final_format_counts.index):
        count = final_format_counts[fmt]
        print(f"  {fmt}: {count} ({count/len(result_df)*100:.1f}%)")
    
    print(f"\nLabel distribution:")
    final_label_counts = result_df['label'].value_counts()
    for label in sorted(final_label_counts.index):
        count = final_label_counts[label]
        print(f"  {label}: {count} ({count/len(result_df)*100:.1f}%)")
    
    return result_df

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Process each dataset
    datasets = {
        'popular': 'popular_test.txt',
        'random': 'random_test.txt', 
        'adversarial': 'adversarial_test.txt'
    }
    
    for dataset_name, filename in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'='*50}")
        
        # Load dataset
        df = load_and_process_dataset(filename)
        print(f"Loaded {len(df)} samples from {filename}")
        
        # Show original distribution
        print(f"\nOriginal question format distribution:")
        original_format_counts = df['question_format'].value_counts()
        for fmt in sorted(original_format_counts.index):
            count = original_format_counts[fmt]
            print(f"  {fmt}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\nOriginal label distribution:")
        original_label_counts = df['label'].value_counts()
        for label in sorted(original_label_counts.index):
            count = original_label_counts[label]
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Create balanced dataset
        balanced_df = create_balanced_dataset(df, target_size=957)
        
        # Save the balanced dataset
        output_filename = f'balanced_{dataset_name}_test_957.txt'
        
        # Write with the same format as original
        with open(output_filename, 'w') as f:
            f.write(f"# Balanced {dataset_name.title()} Sampling - Test Split (957 samples)\n")
            f.write(f"# Total samples: {len(balanced_df)}\n")
            f.write(f"# Columns: {list(balanced_df.columns)}\n")
            f.write("\n")
            
            # Write header
            f.write('\t'.join(balanced_df.columns) + '\n')
            
            # Write data
            for _, row in balanced_df.iterrows():
                f.write('\t'.join(str(row[col]) for col in balanced_df.columns) + '\n')
        
        print(f"\nSaved balanced dataset to: {output_filename}")

if __name__ == "__main__":
    main()