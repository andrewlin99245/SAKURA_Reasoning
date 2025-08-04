#!/usr/bin/env python3
"""
Parameter sweep script for discriminative evaluation
Allows testing multiple parameter combinations systematically
"""

import os
import sys
import itertools
import json
from datetime import datetime
from discriminative_evaluation import run_discriminative_evaluation

# Import required modules
sys.path.insert(0, os.path.abspath("."))

# ---------------------
# Parameter Configurations
# ---------------------

# Define parameter grids for systematic evaluation
PARAMETER_GRIDS = {
    "gamma": [0.0, 0.1, 0.25, 0.5],
    "w": [2, 4, 8],
    "lam": [0.0, 0.01, 0.05, 0.1],
    "fisher_scale": [0.0, 0.25, 0.5, 1.0]
}

# Predefined parameter combinations for common scenarios
PRESET_CONFIGURATIONS = {
    "baseline": {
        "gamma": 0.0,
        "w": 4,
        "lam": 0.0,
        "fisher_scale": 0.0,
        "description": "Baseline model without SLA or VSV"
    },
    "sla_only": {
        "gamma": 0.25,
        "w": 4,
        "lam": 0.0,
        "fisher_scale": 0.0,
        "description": "SLA enabled, no VSV"
    },
    "vsv_only": {
        "gamma": 0.0,
        "w": 4,
        "lam": 0.05,
        "fisher_scale": 0.5,
        "description": "VSV enabled, no SLA"
    },
    "sla_vsv_standard": {
        "gamma": 0.25,
        "w": 4,
        "lam": 0.05,
        "fisher_scale": 0.5,
        "description": "Standard SLA + VSV configuration"
    },
    "sla_vsv_aggressive": {
        "gamma": 0.5,
        "w": 8,
        "lam": 0.1,
        "fisher_scale": 1.0,
        "description": "Aggressive SLA + VSV configuration"
    },
    "sla_vsv_conservative": {
        "gamma": 0.1,
        "w": 2,
        "lam": 0.01,
        "fisher_scale": 0.25,
        "description": "Conservative SLA + VSV configuration"
    }
}

# Dataset configuration
DISCRIMINATIVE_DATASETS = [
    "kuanhuggingface/AudioHallucination_AudioCaps-Random",
    "kuanhuggingface/AudioHallucination_AudioCaps-Popular", 
    "kuanhuggingface/AudioHallucination_AudioCaps-Adversarial"
]

def run_preset_evaluations(audio_root_dir="./audiocaps", max_samples=-1, datasets=None):
    """
    Run evaluation with predefined parameter presets
    """
    if datasets is None:
        datasets = DISCRIMINATIVE_DATASETS
    
    print("=== Running Preset Parameter Evaluations ===")
    
    results_summary = []
    
    for preset_name, config in PRESET_CONFIGURATIONS.items():
        print(f"\n--- Running preset: {preset_name} ---")
        print(f"Description: {config['description']}")
        print(f"Parameters: gamma={config['gamma']}, w={config['w']}, lam={config['lam']}, fisher_scale={config['fisher_scale']}")
        
        # Create model suffix based on preset name
        model_suffix = f"preset_{preset_name}"
        
        preset_results = {
            "preset_name": preset_name,
            "description": config["description"],
            "parameters": {k: v for k, v in config.items() if k != "description"},
            "datasets": {}
        }
        
        for dataset_name in datasets:
            try:
                print(f"\nEvaluating on {dataset_name}")
                run_discriminative_evaluation(
                    gamma=config["gamma"],
                    w=config["w"],
                    lam=config["lam"],
                    fisher_scale=config["fisher_scale"],
                    model_suffix=model_suffix,
                    dataset_name=dataset_name,
                    audio_root_dir=audio_root_dir,
                    max_samples=max_samples
                )
                preset_results["datasets"][dataset_name] = "completed"
            except Exception as e:
                print(f"Error evaluating {dataset_name} with preset {preset_name}: {e}")
                preset_results["datasets"][dataset_name] = f"error: {str(e)}"
        
        results_summary.append(preset_results)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"./discriminative_results/preset_evaluation_summary_{timestamp}.json"
    os.makedirs("./discriminative_results", exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nPreset evaluation summary saved to: {summary_file}")
    return results_summary

def run_parameter_grid_search(audio_root_dir="./audiocaps", max_samples=100, datasets=None, 
                            param_subset=None, max_combinations=None):
    """
    Run evaluation across a grid of parameters
    
    Args:
        audio_root_dir: Path to audio files
        max_samples: Maximum samples per dataset (-1 for all)
        datasets: List of datasets to evaluate
        param_subset: Dictionary to override default parameter grids
        max_combinations: Maximum number of parameter combinations to test
    """
    if datasets is None:
        datasets = DISCRIMINATIVE_DATASETS
    
    # Use subset of parameters if provided
    param_grids = param_subset if param_subset else PARAMETER_GRIDS
    
    print("=== Running Parameter Grid Search ===")
    print(f"Parameter grids: {param_grids}")
    
    # Generate all parameter combinations
    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())
    all_combinations = list(itertools.product(*param_values))
    
    if max_combinations and len(all_combinations) > max_combinations:
        print(f"Limiting to first {max_combinations} combinations out of {len(all_combinations)}")
        all_combinations = all_combinations[:max_combinations]
    
    print(f"Total parameter combinations to test: {len(all_combinations)}")
    
    results_summary = []
    
    for i, combination in enumerate(all_combinations):
        params = dict(zip(param_names, combination))
        
        print(f"\n--- Combination {i+1}/{len(all_combinations)} ---")
        print(f"Parameters: {params}")
        
        # Create model suffix
        model_suffix = f"grid_{i+1:03d}"
        
        combination_results = {
            "combination_id": i + 1,
            "parameters": params,
            "datasets": {}
        }
        
        for dataset_name in datasets:
            try:
                print(f"Evaluating on {dataset_name}")
                run_discriminative_evaluation(
                    gamma=params["gamma"],
                    w=params["w"],
                    lam=params["lam"],
                    fisher_scale=params["fisher_scale"],
                    model_suffix=model_suffix,
                    dataset_name=dataset_name,
                    audio_root_dir=audio_root_dir,
                    max_samples=max_samples
                )
                combination_results["datasets"][dataset_name] = "completed"
            except Exception as e:
                print(f"Error in combination {i+1}, dataset {dataset_name}: {e}")
                combination_results["datasets"][dataset_name] = f"error: {str(e)}"
        
        results_summary.append(combination_results)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"./discriminative_results/grid_search_summary_{timestamp}.json"
    os.makedirs("./discriminative_results", exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nGrid search summary saved to: {summary_file}")
    return results_summary

def run_custom_evaluation(config_file, audio_root_dir="./audiocaps", max_samples=-1, datasets=None):
    """
    Run evaluation with custom configuration from JSON file
    
    Config file format:
    {
        "configurations": [
            {
                "name": "custom_config_1",
                "description": "Description of this configuration",
                "gamma": 0.3,
                "w": 6,
                "lam": 0.08,
                "fisher_scale": 0.7
            },
            ...
        ]
    }
    """
    if datasets is None:
        datasets = DISCRIMINATIVE_DATASETS
    
    print(f"=== Running Custom Evaluations from {config_file} ===")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return []
    
    configurations = config_data.get("configurations", [])
    print(f"Found {len(configurations)} custom configurations")
    
    results_summary = []
    
    for i, config in enumerate(configurations):
        name = config.get("name", f"custom_{i+1}")
        description = config.get("description", "No description")
        
        print(f"\n--- Running custom config: {name} ---")
        print(f"Description: {description}")
        
        required_params = ["gamma", "w", "lam", "fisher_scale"]
        missing_params = [p for p in required_params if p not in config]
        
        if missing_params:
            print(f"Skipping config {name}: missing parameters {missing_params}")
            continue
        
        model_suffix = f"custom_{name}"
        
        config_results = {
            "config_name": name,
            "description": description,
            "parameters": {p: config[p] for p in required_params},
            "datasets": {}
        }
        
        for dataset_name in datasets:
            try:
                print(f"Evaluating on {dataset_name}")
                run_discriminative_evaluation(
                    gamma=config["gamma"],
                    w=config["w"],
                    lam=config["lam"],
                    fisher_scale=config["fisher_scale"],
                    model_suffix=model_suffix,
                    dataset_name=dataset_name,
                    audio_root_dir=audio_root_dir,
                    max_samples=max_samples
                )
                config_results["datasets"][dataset_name] = "completed"
            except Exception as e:
                print(f"Error evaluating {dataset_name} with config {name}: {e}")
                config_results["datasets"][dataset_name] = f"error: {str(e)}"
        
        results_summary.append(config_results)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"./discriminative_results/custom_evaluation_summary_{timestamp}.json"
    os.makedirs("./discriminative_results", exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nCustom evaluation summary saved to: {summary_file}")
    return results_summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter sweep for discriminative evaluation")
    parser.add_argument("--mode", choices=["preset", "grid", "custom"], default="preset",
                        help="Evaluation mode: preset configurations, grid search, or custom config file")
    parser.add_argument("--audio_root_dir", type=str, default="./audiocabs",
                        help="Root directory containing audio files")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of samples to process per dataset (-1 for all)")
    parser.add_argument("--datasets", nargs='+', default=DISCRIMINATIVE_DATASETS,
                        help="List of discriminative datasets to evaluate")
    parser.add_argument("--config_file", type=str,
                        help="JSON config file for custom mode")
    parser.add_argument("--max_combinations", type=int, default=None,
                        help="Maximum number of parameter combinations for grid search")
    parser.add_argument("--quick_grid", action="store_true",
                        help="Use a smaller parameter grid for faster testing")
    
    args = parser.parse_args()
    
    print("=== Discriminative Parameter Sweep ===")
    print(f"Mode: {args.mode}")
    print(f"Audio root: {args.audio_root_dir}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Datasets: {len(args.datasets)} datasets")
    
    if args.mode == "preset":
        run_preset_evaluations(
            audio_root_dir=args.audio_root_dir,
            max_samples=args.max_samples,
            datasets=args.datasets
        )
    
    elif args.mode == "grid":
        if args.quick_grid:
            # Smaller parameter grid for quick testing
            param_subset = {
                "gamma": [0.0, 0.25],
                "w": [4],
                "lam": [0.0, 0.05],
                "fisher_scale": [0.0, 0.5]
            }
        else:
            param_subset = None
        
        run_parameter_grid_search(
            audio_root_dir=args.audio_root_dir,
            max_samples=args.max_samples,
            datasets=args.datasets,
            param_subset=param_subset,
            max_combinations=args.max_combinations
        )
    
    elif args.mode == "custom":
        if not args.config_file:
            print("Error: --config_file required for custom mode")
            return
        
        run_custom_evaluation(
            config_file=args.config_file,
            audio_root_dir=args.audio_root_dir,
            max_samples=args.max_samples,
            datasets=args.datasets
        )
    
    print("=== Parameter Sweep Complete ===")

if __name__ == "__main__":
    main()