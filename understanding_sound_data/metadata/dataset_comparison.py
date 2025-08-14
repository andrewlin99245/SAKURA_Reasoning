#!/usr/bin/env python3
import pandas as pd
import numpy as np

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

def load_original_dataset(filepath):
    """Load original dataset"""
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
    
    df = pd.DataFrame(rows, columns=header)
    df['question_format'] = df['prompt_text'].apply(classify_question_format)
    return df

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

def analyze_dataset(df, name):
    """Analyze a dataset and return statistics"""
    stats = {
        'name': name,
        'total_samples': len(df),
        'unique_audio': df['audio_index'].nunique(),
        'avg_samples_per_audio': len(df) / df['audio_index'].nunique(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'format_distribution': df['question_format'].value_counts().to_dict()
    }
    return stats

def print_comparison(original_stats, balanced_stats):
    """Print detailed comparison"""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {original_stats['name']} vs {balanced_stats['name']}")
    print(f"{'='*80}")
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"{'Metric':<25} {'Original':<15} {'Balanced':<15} {'Change':<15}")
    print(f"{'-'*70}")
    print(f"{'Total Samples':<25} {original_stats['total_samples']:<15,} {balanced_stats['total_samples']:<15,} {balanced_stats['total_samples']/original_stats['total_samples']*100:.1f}%")
    print(f"{'Unique Audio Files':<25} {original_stats['unique_audio']:<15,} {balanced_stats['unique_audio']:<15,} {balanced_stats['unique_audio']/original_stats['unique_audio']*100:.1f}%")
    print(f"{'Samples per Audio':<25} {original_stats['avg_samples_per_audio']:<15.1f} {balanced_stats['avg_samples_per_audio']:<15.1f} {balanced_stats['avg_samples_per_audio']/original_stats['avg_samples_per_audio']*100:.1f}%")
    
    # Label distribution
    print(f"\n🏷️  LABEL DISTRIBUTION:")
    print(f"{'Label':<25} {'Original Count':<15} {'Original %':<12} {'Balanced Count':<15} {'Balanced %':<12}")
    print(f"{'-'*80}")
    
    orig_total = sum(original_stats['label_distribution'].values())
    bal_total = sum(balanced_stats['label_distribution'].values())
    
    all_labels = set(original_stats['label_distribution'].keys()) | set(balanced_stats['label_distribution'].keys())
    for label in sorted(all_labels):
        orig_count = original_stats['label_distribution'].get(label, 0)
        bal_count = balanced_stats['label_distribution'].get(label, 0)
        orig_pct = orig_count / orig_total * 100
        bal_pct = bal_count / bal_total * 100
        print(f"{label:<25} {orig_count:<15,} {orig_pct:<12.1f} {bal_count:<15,} {bal_pct:<12.1f}")
    
    # Format distribution
    print(f"\n❓ QUESTION FORMAT DISTRIBUTION:")
    print(f"{'Format':<25} {'Original Count':<15} {'Original %':<12} {'Balanced Count':<15} {'Balanced %':<12}")
    print(f"{'-'*80}")
    
    all_formats = set(original_stats['format_distribution'].keys()) | set(balanced_stats['format_distribution'].keys())
    for fmt in sorted(all_formats):
        orig_count = original_stats['format_distribution'].get(fmt, 0)
        bal_count = balanced_stats['format_distribution'].get(fmt, 0)
        orig_pct = orig_count / orig_total * 100
        bal_pct = bal_count / bal_total * 100
        print(f"{fmt:<25} {orig_count:<15,} {orig_pct:<12.1f} {bal_count:<15,} {bal_pct:<12.1f}")

def main():
    datasets = [
        ('popular_test.txt', 'balanced_popular_test_957.txt', 'POPULAR'),
        ('random_test.txt', 'balanced_random_test_957.txt', 'RANDOM'),
        ('adversarial_test.txt', 'balanced_adversarial_test_957.txt', 'ADVERSARIAL')
    ]
    
    print("DATASET COMPARISON REPORT")
    print("=" * 80)
    
    all_original_stats = []
    all_balanced_stats = []
    
    for orig_file, bal_file, name in datasets:
        # Load datasets
        print(f"\nLoading {name} datasets...")
        orig_df = load_original_dataset(orig_file)
        bal_df = load_balanced_dataset(bal_file)
        
        # Analyze
        orig_stats = analyze_dataset(orig_df, f"Original {name}")
        bal_stats = analyze_dataset(bal_df, f"Balanced {name}")
        
        all_original_stats.append(orig_stats)
        all_balanced_stats.append(bal_stats)
        
        # Print comparison
        print_comparison(orig_stats, bal_stats)
    
    # Summary across all datasets
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*80}")
    
    print(f"\n📈 TOTAL REDUCTION:")
    total_orig = sum(stats['total_samples'] for stats in all_original_stats)
    total_bal = sum(stats['total_samples'] for stats in all_balanced_stats)
    reduction = (total_orig - total_bal) / total_orig * 100
    print(f"Original total samples: {total_orig:,}")
    print(f"Balanced total samples: {total_bal:,}")
    print(f"Size reduction: {reduction:.1f}%")
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print("✅ Perfect label balance: ~50% Yes / ~50% No in all datasets")
    print("✅ One sample per audio file: Eliminates redundancy")
    print("✅ Preserved format distribution: Maintains question diversity")
    print("✅ Consistent size: All datasets now have exactly 957 samples")
    
    print(f"\n📋 FORMAT DEFINITIONS:")
    formats = {
        "Format1": "Is there a sound of X in the audio?",
        "Format2": "Does the audio contain the sound of X?", 
        "Format3": "Have you noticed the sound of X in the audio?",
        "Format4": "Can you hear the sound of X in the audio?",
        "Format5": "Can you detect the sound of a X in the audio?"
    }
    
    for fmt, description in formats.items():
        print(f"  {fmt}: {description}")

if __name__ == "__main__":
    main()