#!/usr/bin/env python3
"""
Analysis script for discriminative evaluation results
Processes CSV results and generates metrics, visualizations, and summaries
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def normalize_response(response):
    """
    Normalize model responses to binary labels
    """
    if isinstance(response, str):
        response_lower = response.lower()
        if any(word in response_lower for word in ['yes', 'true', 'positive', 'present', 'hear', 'detect']):
            return 'positive'
        else:
            return 'negative'
    return 'negative'

def calculate_metrics(df):
    """
    Calculate comprehensive evaluation metrics from results dataframe
    """
    # Normalize responses
    df['normalized_response'] = df['response'].apply(normalize_response)
    
    # Convert to binary for sklearn
    y_true = (df['label'] == 'positive').astype(int)
    y_pred = (df['normalized_response'] == 'positive').astype(int)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Additional metrics
    yes_rate = (df['normalized_response'] == 'positive').mean()
    positive_rate = (df['label'] == 'positive').mean()
    
    # Per-object analysis
    object_metrics = {}
    for obj in df['object'].unique():
        obj_df = df[df['object'] == obj]
        if len(obj_df) > 0:
            obj_y_true = (obj_df['label'] == 'positive').astype(int)
            obj_y_pred = (obj_df['normalized_response'] == 'positive').astype(int)
            
            object_metrics[obj] = {
                'accuracy': accuracy_score(obj_y_true, obj_y_pred),
                'precision': precision_score(obj_y_true, obj_y_pred, zero_division=0),
                'recall': recall_score(obj_y_true, obj_y_pred, zero_division=0),
                'f1': f1_score(obj_y_true, obj_y_pred, zero_division=0),
                'yes_rate': (obj_df['normalized_response'] == 'positive').mean(),
                'positive_rate': (obj_df['label'] == 'positive').mean(),
                'count': len(obj_df)
            }
    
    # Per-sampling method analysis
    sampling_metrics = {}
    for sampling in df['sampling'].unique():
        samp_df = df[df['sampling'] == sampling]
        if len(samp_df) > 0:
            samp_y_true = (samp_df['label'] == 'positive').astype(int)
            samp_y_pred = (samp_df['normalized_response'] == 'positive').astype(int)
            
            sampling_metrics[sampling] = {
                'accuracy': accuracy_score(samp_y_true, samp_y_pred),
                'precision': precision_score(samp_y_true, samp_y_pred, zero_division=0),
                'recall': recall_score(samp_y_true, samp_y_pred, zero_division=0),
                'f1': f1_score(samp_y_true, samp_y_pred, zero_division=0),
                'yes_rate': (samp_df['normalized_response'] == 'positive').mean(),
                'positive_rate': (samp_df['label'] == 'positive').mean(),
                'count': len(samp_df)
            }
    
    return {
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'yes_rate': yes_rate,
            'positive_rate': positive_rate,
            'total_samples': len(df)
        },
        'by_object': object_metrics,
        'by_sampling': sampling_metrics
    }

def analyze_single_file(csv_file):
    """
    Analyze a single CSV result file
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"\nAnalyzing: {csv_file}")
        print(f"Total samples: {len(df)}")
        
        metrics = calculate_metrics(df)
        
        # Print overall metrics
        overall = metrics['overall']
        print(f"Overall Performance:")
        print(f"  Accuracy: {overall['accuracy']:.3f}")
        print(f"  Precision: {overall['precision']:.3f}")
        print(f"  Recall: {overall['recall']:.3f}")
        print(f"  F1 Score: {overall['f1']:.3f}")
        print(f"  Yes Rate: {overall['yes_rate']:.3f}")
        print(f"  Positive Rate: {overall['positive_rate']:.3f}")
        
        # Print per-object metrics
        print(f"\nPer-Object Performance:")
        for obj, obj_metrics in metrics['by_object'].items():
            print(f"  {obj}: Acc={obj_metrics['accuracy']:.3f}, F1={obj_metrics['f1']:.3f}, Yes={obj_metrics['yes_rate']:.3f} (n={obj_metrics['count']})")
        
        # Print per-sampling metrics
        print(f"\nPer-Sampling Performance:")
        for sampling, samp_metrics in metrics['by_sampling'].items():
            print(f"  {sampling}: Acc={samp_metrics['accuracy']:.3f}, F1={samp_metrics['f1']:.3f}, Yes={samp_metrics['yes_rate']:.3f} (n={samp_metrics['count']})")
        
        return metrics, df
        
    except Exception as e:
        print(f"Error analyzing {csv_file}: {e}")
        return None, None

def compare_configurations(results_dir):
    """
    Compare results across different configurations
    """
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    
    print(f"Found {len(csv_files)} result files")
    
    all_results = {}
    comparison_data = []
    
    for csv_file in csv_files:
        file_path = os.path.join(results_dir, csv_file)
        metrics, df = analyze_single_file(file_path)
        
        if metrics:
            config_name = csv_file.replace('.csv', '')
            all_results[config_name] = metrics
            
            # Extract dataset type and model suffix from filename
            parts = config_name.split('_')
            dataset_type = parts[0] if parts else 'unknown'
            model_suffix = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
            
            comparison_data.append({
                'config': config_name,
                'dataset_type': dataset_type,
                'model_suffix': model_suffix,
                'accuracy': metrics['overall']['accuracy'],
                'precision': metrics['overall']['precision'],
                'recall': metrics['overall']['recall'],
                'f1': metrics['overall']['f1'],
                'yes_rate': metrics['overall']['yes_rate'],
                'samples': metrics['overall']['total_samples']
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        print(f"\n=== Configuration Comparison ===")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Save comparison
        comparison_file = os.path.join(results_dir, 'configuration_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nComparison saved to: {comparison_file}")
        
        # Create visualizations
        create_comparison_plots(comparison_df, results_dir)
    
    # Save detailed analysis
    analysis_file = os.path.join(results_dir, 'detailed_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed analysis saved to: {analysis_file}")
    
    return comparison_df, all_results

def create_comparison_plots(comparison_df, results_dir):
    """
    Create comparison plots for different configurations
    """
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # Set up the plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Discriminative Evaluation Results Comparison', fontsize=16)
    
    # Plot 1: Accuracy by configuration
    ax1 = axes[0, 0]
    comparison_df.plot(x='config', y='accuracy', kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Accuracy by Configuration')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: F1 Score by configuration
    ax2 = axes[0, 1]
    comparison_df.plot(x='config', y='f1', kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('F1 Score by Configuration')
    ax2.set_ylabel('F1 Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Precision vs Recall
    ax3 = axes[1, 0]
    ax3.scatter(comparison_df['recall'], comparison_df['precision'], 
                c=comparison_df['f1'], cmap='viridis', s=100)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall (colored by F1)')
    for i, config in enumerate(comparison_df['config']):
        ax3.annotate(config.split('_')[-1], 
                    (comparison_df['recall'].iloc[i], comparison_df['precision'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Yes Rate analysis
    ax4 = axes[1, 1]
    comparison_df.plot(x='config', y='yes_rate', kind='bar', ax=ax4, color='orange')
    ax4.set_title('Yes Rate by Configuration')
    ax4.set_ylabel('Yes Rate')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(results_dir, 'comparison_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_file}")
    
    plt.show()

def analyze_by_dataset_type(results_dir):
    """
    Analyze performance by dataset type (Random, Popular, Adversarial)
    """
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    dataset_analysis = defaultdict(list)
    
    for csv_file in csv_files:
        # Extract dataset type from filename
        if 'Random' in csv_file:
            dataset_type = 'Random'
        elif 'Popular' in csv_file:
            dataset_type = 'Popular'
        elif 'Adversarial' in csv_file:
            dataset_type = 'Adversarial'
        else:
            continue
        
        file_path = os.path.join(results_dir, csv_file)
        metrics, df = analyze_single_file(file_path)
        
        if metrics:
            dataset_analysis[dataset_type].append({
                'file': csv_file,
                'metrics': metrics['overall']
            })
    
    print(f"\n=== Analysis by Dataset Type ===")
    for dataset_type, results in dataset_analysis.items():
        print(f"\n{dataset_type} Dataset:")
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in results])
        avg_f1 = np.mean([r['metrics']['f1'] for r in results])
        avg_yes_rate = np.mean([r['metrics']['yes_rate'] for r in results])
        
        print(f"  Average Accuracy: {avg_accuracy:.3f}")
        print(f"  Average F1: {avg_f1:.3f}")
        print(f"  Average Yes Rate: {avg_yes_rate:.3f}")
        print(f"  Number of configurations: {len(results)}")

def main():
    parser = argparse.ArgumentParser(description="Analyze discriminative evaluation results")
    parser.add_argument("--results_dir", type=str, default="./discriminative_results",
                        help="Directory containing CSV result files")
    parser.add_argument("--single_file", type=str,
                        help="Analyze a single CSV file")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all configurations in results directory")
    parser.add_argument("--by_dataset", action="store_true",
                        help="Analyze performance by dataset type")
    
    args = parser.parse_args()
    
    if args.single_file:
        if os.path.exists(args.single_file):
            analyze_single_file(args.single_file)
        else:
            print(f"File not found: {args.single_file}")
    
    elif args.compare:
        if os.path.exists(args.results_dir):
            compare_configurations(args.results_dir)
        else:
            print(f"Results directory not found: {args.results_dir}")
    
    elif args.by_dataset:
        if os.path.exists(args.results_dir):
            analyze_by_dataset_type(args.results_dir)
        else:
            print(f"Results directory not found: {args.results_dir}")
    
    else:
        # Default: analyze all files in results directory
        if os.path.exists(args.results_dir):
            print("Running comprehensive analysis...")
            compare_configurations(args.results_dir)
            analyze_by_dataset_type(args.results_dir)
        else:
            print(f"Results directory not found: {args.results_dir}")
            print("Create the directory and run evaluations first.")

if __name__ == "__main__":
    main()