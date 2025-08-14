import pandas as pd
import sys
from sklearn.metrics import accuracy_score, f1_score

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy and F1 score for binary classification"""
    # Convert Yes/No to 1/0 for sklearn
    y_true_binary = [1 if label == 'Yes' else 0 for label in y_true]
    y_pred_binary = [1 if response == 'Yes' else 0 for response in y_pred]
    
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return accuracy, f1

def analyze_responses(filename):
    df = pd.read_csv(filename)
    
    # Count yes responses (original functionality)
    yes_count = (df['response'] == 'Yes').sum()
    total_count = len(df)
    
    # Calculate overall metrics
    overall_accuracy, overall_f1 = calculate_metrics(df['label'], df['response'])
    
    # Calculate metrics by sampling method
    sampling_methods = df['sampling_method'].unique()
    method_metrics = {}
    
    for method in sampling_methods:
        method_df = df[df['sampling_method'] == method]
        if len(method_df) > 0:
            accuracy, f1 = calculate_metrics(method_df['label'], method_df['response'])
            method_metrics[method] = {
                'accuracy': accuracy,
                'f1': f1,
                'count': len(method_df)
            }
    
    return yes_count, total_count, overall_accuracy, overall_f1, method_metrics

if len(sys.argv) < 2:
    print("Usage: python count_yes_responses.py <csv_file1> [csv_file2] ...")
    sys.exit(1)

for filename in sys.argv[1:]:
    try:
        yes_count, total_count, overall_accuracy, overall_f1, method_metrics = analyze_responses(filename)
        
        print(f"\n=== Analysis for {filename} ===")
        print(f"Total responses: {total_count}")
        print(f"'Yes' responses: {yes_count}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {overall_accuracy:.4f}")
        print(f"  F1 Score: {overall_f1:.4f}")
        
        print(f"\nMetrics by Sampling Method:")
        for method, metrics in method_metrics.items():
            print(f"  {method}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")