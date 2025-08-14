#!/usr/bin/env python3
import pandas as pd

def load_balanced_dataset(filepath):
    """Load balanced dataset"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('entry_id'):
            header_idx = i
            break
    
    data_lines = lines[header_idx:]
    rows = []
    header = data_lines[0].strip().split('\t')
    
    for line in data_lines[1:]:
        if line.strip():
            rows.append(line.strip().split('\t'))
    
    return pd.DataFrame(rows, columns=header)

def main():
    # Load all three balanced datasets
    datasets = []
    dataset_names = ['popular', 'random', 'adversarial']
    
    for name in dataset_names:
        filename = f'balanced_{name}_test_957.txt'
        print(f"Loading {filename}...")
        df = load_balanced_dataset(filename)
        datasets.append(df)
        print(f"  Loaded {len(df)} samples")
    
    # Merge all datasets
    print(f"\nMerging {len(datasets)} datasets...")
    merged_df = pd.concat(datasets, ignore_index=True)
    
    # Print statistics
    print(f"\nMerged dataset statistics:")
    print(f"Total samples: {len(merged_df)}")
    print(f"Unique audio files: {merged_df['audio_index'].nunique()}")
    
    print(f"\nSampling method distribution:")
    sampling_counts = merged_df['sampling'].value_counts()
    for method in sorted(sampling_counts.index):
        count = sampling_counts[method]
        print(f"  {method}: {count} ({count/len(merged_df)*100:.1f}%)")
    
    print(f"\nLabel distribution:")
    label_counts = merged_df['label'].value_counts()
    for label in sorted(label_counts.index):
        count = label_counts[label]
        print(f"  {label}: {count} ({count/len(merged_df)*100:.1f}%)")
    
    print(f"\nQuestion format distribution:")
    format_counts = merged_df['question_format'].value_counts()
    for fmt in sorted(format_counts.index):
        count = format_counts[fmt]
        print(f"  {fmt}: {count} ({count/len(merged_df)*100:.1f}%)")
    
    # Save merged dataset
    output_filename = 'balanced_merged_test_2871.txt'
    
    with open(output_filename, 'w') as f:
        f.write(f"# Balanced Merged Test Dataset (All Sampling Methods)\n")
        f.write(f"# Total samples: {len(merged_df)}\n")
        f.write(f"# Sampling methods: popular (957), random (957), adversarial (957)\n")
        f.write(f"# Columns: {list(merged_df.columns)}\n")
        f.write("\n")
        
        # Write header
        f.write('\t'.join(merged_df.columns) + '\n')
        
        # Write data
        for _, row in merged_df.iterrows():
            f.write('\t'.join(str(row[col]) for col in merged_df.columns) + '\n')
    
    print(f"\nSaved merged dataset to: {output_filename}")
    
    # Verify unique audio files per sampling method
    print(f"\nVerification - Unique audio files per sampling method:")
    for method in dataset_names:
        method_df = merged_df[merged_df['sampling'] == method]
        unique_audio = method_df['audio_index'].nunique()
        total_samples = len(method_df)
        print(f"  {method}: {total_samples} samples, {unique_audio} unique audio files")
        
        if total_samples != unique_audio:
            print(f"    ⚠️  Warning: Expected 1 sample per audio file!")

if __name__ == "__main__":
    main()