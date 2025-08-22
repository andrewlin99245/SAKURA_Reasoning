#!/usr/bin/env python3
"""
Grid search for oscillator.py parameters to find optimal configuration.
"""
import os
import sys
import subprocess
import json
import itertools
from pathlib import Path
import time

# Parameter grid definitions
PARAMETER_GRID = {
    'template_last_k': [3, 4, 5, 6],
    'beta1': [0.05, 0.1, 0.2, 0.3],
    'beta2': [0.05, 0.1, 0.2, 0.3, 0.5],
    'beta3': [0.05, 0.1, 0.2, 0.3],
    'gamma_mf': [0.0, 0.1, 0.2, 0.3],
    'gamma_Q': [0.0, 0.05, 0.1, 0.15],
    'gamma_alpha': [0.0, 0.05, 0.1, 0.15],
    'gamma_amp': [0.0, 0.02, 0.05, 0.1],
    'gamma_energy': [0.0, 0.02, 0.05, 0.1],
    'gamma_maha': [0.0, 0.05, 0.1, 0.15],
    'flip_threshold': [0.4, 0.5, 0.6]
}

# Fixed parameters
FIXED_PARAMS = {
    'bos_csv': 'BOS_features_BOS_features.csv',
    'lda_stats': 'BOS_features_BOS_stats.json',
    'use_augmented': True
}

def run_oscillator_evaluation(params, run_id):
    """Run single oscillator evaluation with given parameters."""
    print(f"🔍 Running evaluation {run_id} with params: {params}")
    
    # Create templates for this k value
    k = params['template_last_k']
    corr_template = f"correlation_template_k{k}.json" 
    resonant_template = f"resonant_template_k{k}.json"
    output_dir = f"./grid_search_results/run_{run_id:04d}"
    
    # Fit templates if they don't exist
    if not os.path.exists(corr_template):
        print(f"  📐 Fitting correlation template for k={k}")
        cmd_corr = [
            'python', 'src/models/oscillator.py',
            '--fit_corr_template_from', FIXED_PARAMS['bos_csv'],
            '--corr_template_out', corr_template,
            '--template_last_k', str(k)
        ]
        subprocess.run(cmd_corr, capture_output=True, text=True)
    
    if not os.path.exists(resonant_template):
        print(f"  🧪 Fitting resonant template for k={k}")
        cmd_resonant = [
            'python', 'src/models/oscillator.py', 
            '--fit_resonant_from', FIXED_PARAMS['bos_csv'],
            '--resonant_out', resonant_template,
            '--resonant_last_k', str(k)
        ]
        subprocess.run(cmd_resonant, capture_output=True, text=True)
    
    # Run evaluation
    print(f"  🎯 Running evaluation...")
    cmd_eval = [
        'python', 'src/models/oscillator.py',
        '--bos_csv', FIXED_PARAMS['bos_csv'],
        '--lda_stats', FIXED_PARAMS['lda_stats'],
        '--corr_template', corr_template,
        '--resonant_template', resonant_template,
        '--use_augmented',
        '--beta1', str(params['beta1']),
        '--beta2', str(params['beta2']),
        '--beta3', str(params['beta3']),
        '--gamma_mf', str(params['gamma_mf']),
        '--gamma_Q', str(params['gamma_Q']),
        '--gamma_alpha', str(params['gamma_alpha']),
        '--gamma_amp', str(params['gamma_amp']),
        '--gamma_energy', str(params['gamma_energy']),
        '--gamma_maha', str(params['gamma_maha']),
        '--flip_threshold', str(params['flip_threshold']),
        '--output_dir', output_dir
    ]
    
    result = subprocess.run(cmd_eval, capture_output=True, text=True)
    
    # Parse results
    results_file = os.path.join(output_dir, 'lda_correction_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        return {
            'run_id': run_id,
            'params': params,
            'results': results,
            'success': True
        }
    else:
        print(f"  ❌ Failed to find results file for run {run_id}")
        print(f"  stderr: {result.stderr}")
        return {
            'run_id': run_id,
            'params': params,
            'results': None,
            'success': False,
            'error': result.stderr
        }

def smart_grid_search():
    """
    Smart grid search - start with reasonable ranges and explore promising areas.
    """
    os.makedirs('grid_search_results', exist_ok=True)
    
    # Phase 1: Coarse search with smaller parameter space
    print("🚀 Phase 1: Coarse grid search")
    coarse_grid = {
        'template_last_k': [4, 5],
        'beta1': [0.1, 0.2],
        'beta2': [0.1, 0.2, 0.3],
        'beta3': [0.1, 0.2],
        'gamma_mf': [0.1, 0.2],
        'gamma_Q': [0.05, 0.1],
        'gamma_alpha': [0.05, 0.1],
        'gamma_amp': [0.02, 0.05],
        'gamma_energy': [0.02, 0.05],
        'gamma_maha': [0.05, 0.1],
        'flip_threshold': [0.5]
    }
    
    all_results = []
    run_id = 0
    
    # Generate parameter combinations
    keys = list(coarse_grid.keys())
    values = list(coarse_grid.values())
    
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    
    print(f"📊 Total combinations in Phase 1: {total_combinations}")
    
    start_time = time.time()
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        
        result = run_oscillator_evaluation(params, run_id)
        all_results.append(result)
        
        if result['success']:
            acc = result['results']['augmented_accuracy']
            improvement = result['results']['improvement']
            print(f"  ✅ Run {run_id}: Accuracy={acc:.2f}%, Improvement={improvement:.2f}pts")
        
        run_id += 1
        
        # Save intermediate results
        if run_id % 10 == 0:
            with open('grid_search_results/intermediate_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Save Phase 1 results
    phase1_results_file = 'grid_search_results/phase1_results.json'
    with open(phase1_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"⏱️ Phase 1 completed in {elapsed/60:.1f} minutes")
    
    # Analyze Phase 1 results
    successful_results = [r for r in all_results if r['success']]
    if not successful_results:
        print("❌ No successful runs in Phase 1")
        return all_results
    
    # Sort by augmented accuracy
    successful_results.sort(key=lambda x: x['results']['augmented_accuracy'], reverse=True)
    
    print(f"\n🏆 Top 5 Phase 1 Results:")
    for i, result in enumerate(successful_results[:5]):
        acc = result['results']['augmented_accuracy']
        improvement = result['results']['improvement']
        params = result['params']
        print(f"  {i+1}. Accuracy: {acc:.2f}% ({improvement:+.2f}pts)")
        print(f"     k={params['template_last_k']}, β=({params['beta1']},{params['beta2']},{params['beta3']})")
        print(f"     γ=({params['gamma_mf']},{params['gamma_Q']},{params['gamma_alpha']},{params['gamma_amp']},{params['gamma_energy']},{params['gamma_maha']})")
    
    return all_results

def main():
    """Main grid search execution."""
    print("🔬 Oscillator Parameter Grid Search")
    print("=" * 50)
    
    # Run smart grid search
    results = smart_grid_search()
    
    # Final analysis
    print(f"\n📊 Final Analysis")
    print("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['results']['augmented_accuracy'])
        
        print(f"🎯 Best Configuration:")
        print(f"  Accuracy: {best_result['results']['augmented_accuracy']:.2f}%")
        print(f"  Improvement: {best_result['results']['improvement']:+.2f}pts")
        print(f"  Parameters: {best_result['params']}")
        
        # Save best configuration
        with open('grid_search_results/best_config.json', 'w') as f:
            json.dump(best_result, f, indent=2)
        
        print(f"💾 Results saved to grid_search_results/")
    else:
        print("❌ No successful configurations found")

if __name__ == "__main__":
    main()