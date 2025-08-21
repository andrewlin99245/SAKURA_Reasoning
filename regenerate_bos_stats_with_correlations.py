#!/usr/bin/env python3
"""
Regenerate BOS stats JSON with correlation features from existing BOS CSV data.
"""
import csv
import json
import numpy as np
import sys
import os

# Add the models directory to the path to import our functions
sys.path.append('src/models')
from cosine_correlation_bos_with_lda import compute_correlation_features, compute_class_stats

def load_bos_csv(csv_path):
    """Load BOS features from CSV file."""
    data = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    print(f"Loaded {len(data)} samples from {csv_path}")
    return data

def extract_features_with_correlations(row, use_last_k_layers=8):
    """Extract BOS features including correlations from CSV row."""
    # Base features (excluding entropy_bos which has NaN)
    base_features = [
        float(row["bos_mean_late"]),
        float(row["bos_std_late"]),
        float(row["margin_bos"]),
        float(row["gap_EL"]),
    ]
    base_names = ["bos_mean_late", "bos_std_late", "margin_bos", "gap_EL"]
    
    # Layer cosines (assuming layers 24-31 based on existing data)
    layer_cosines = []
    layer_names = []
    for i in range(24, 32):  # cos_L24 through cos_L31
        col_name = f"cos_L{i}"
        if col_name in row:
            layer_cosines.append(float(row[col_name]))
            layer_names.append(col_name)
    
    # Create cos_per_layer for correlation computation
    cos_per_layer = {24+i: val for i, val in enumerate(layer_cosines)}
    layer_indices = list(range(24, 24 + len(layer_cosines)))
    
    # Compute correlation features
    corr_features = compute_correlation_features(cos_per_layer, layer_indices)
    
    # Combine all features
    all_features = base_features + layer_cosines + corr_features["values"]
    all_names = base_names + layer_names + corr_features["names"]
    
    return np.array(all_features, dtype=np.float64), all_names

def main():
    csv_path = "BOS_features_BOS_features.csv"
    output_path = "BOS_features_BOS_stats_with_correlations.json"
    
    print("🔄 Regenerating BOS stats with correlation features...")
    
    # Load data
    data = load_bos_csv(csv_path)
    
    # Extract features for all samples
    X = []
    y = []
    feature_names = None
    
    for row in data:
        try:
            features, names = extract_features_with_correlations(row)
            if feature_names is None:
                feature_names = names
            X.append(features)
            y.append(int(row["correct"]))
        except Exception as e:
            print(f"⚠️ Error processing row {row.get('entry_id', '?')}: {e}")
            continue
    
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=int)
    
    print(f"📊 Processing {len(X)} samples with {X.shape[1]} features")
    print(f"📊 Features: {feature_names}")
    print(f"📊 Class distribution: {np.sum(y)} correct, {len(y) - np.sum(y)} incorrect")
    
    # Compute class statistics
    class_stats = compute_class_stats(X, y)
    
    # Save to JSON
    stats_payload = {
        "feature_names": feature_names,
        "class_stats": class_stats,
        "early_idx": list(range(0, 8)),  # Placeholder 
        "late_idx": list(range(24, 32)),  # Layers 24-31
        "feature_vector_note": "x = [bos_mean_late, bos_std_late, margin_bos, gap_EL, cos_L24-31, correlation_features]",
        "correlation_features_note": "Added: consecutive differences, trend slope, early-late correlation, variance, range, first-last diff, peaks/valleys",
        "lda_equations": {
            "w": "w = Sigma^{-1} (mu1 - mu0)",
            "b": "b = -0.5 (mu1 + mu0)^T Sigma^{-1} (mu1 - mu0) + log(pi1/pi0)",
            "p_correct": "p = 1 / (1 + exp(-(w^T x + b)))"
        }
    }
    
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(x) for x in obj]
        elif isinstance(obj, np.integer): 
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.bool_): 
            return bool(obj)
        elif isinstance(obj, np.ndarray): 
            return obj.tolist()
        else: 
            return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(stats_payload), f, indent=2)
    
    print(f"✅ Saved enhanced BOS stats to: {output_path}")
    print(f"📈 Features expanded from 12 to {len(feature_names)}")

if __name__ == "__main__":
    main()