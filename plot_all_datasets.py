"""
Plot all dataset results: Annealing vs SSP comparison with standard error bars.
Shows results for: Adult, SeoulBikeData, LifeExpectancy, BreastCancerWisconsin
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Dataset configurations
datasets = [
    ("adult", "results/adult_rho_study_full/annealing_privacy_analysis_adult_d6_alpha0.2_remeasure.json"),
    ("SeoulBikeData", "results/SeoulBikeData_rho_study_full/annealing_privacy_analysis_SeoulBikeData_d9_alpha0.2_remeasure.json"),
    ("LifeExpectancy", "results/LifeExpectancy_rho_study_full/annealing_privacy_analysis_LifeExpectancy_d15_alpha0.2_remeasure.json"),
    ("BreastCancerWisconsin", "results/BreastCancerWisconsin_rho_study_full/annealing_privacy_analysis_BreastCancerWisconsin_d31_alpha0.2_remeasure.json"),
]

# Color scheme
color_annealing = '#2E86AB'
color_ssp = '#A23B72'

# Create figure with 4x2 subplots (4 datasets, 2 error types each)
fig, axes = plt.subplots(4, 2, figsize=(14, 14))
fig.suptitle('Multi-Dataset Comparison: Annealing vs SSP (Adult outliers removed only)\n(10 trials per rho, error bars = standard error)', 
             fontsize=14, fontweight='bold')

all_summary = {}

for dataset_idx, (dataset_name, results_file) in enumerate(datasets):
    results_path = Path(results_file)
    
    if not results_path.exists():
        print(f"⚠️  Skipping {dataset_name}: {results_file} not found")
        continue
    
    print(f"\n📊 Processing {dataset_name}...")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Check if results has the expected structure
    if 'results' not in data:
        print(f"⚠️  Skipping {dataset_name}: No 'results' key found")
        continue
    
    results = data['results']
    
    # Handle different result formats
    # Format 1: {rho_str: [trial1, trial2, ...]}
    if isinstance(results.get(list(results.keys())[0] if results else '0.0001'), list):
        rho_values = [float(rho) for rho in sorted(results.keys())]
        
        # Extract metrics
        mahal_annealing_clean = []
        mahal_ssp = []
        frob_annealing_clean = []
        frob_ssp = []
        
        for rho_str in sorted(results.keys(), key=lambda x: float(x)):
            trials = results[rho_str]
            
            mahal_a = [trial['annealing']['mahalanobis_error'] for trial in trials]
            mahal_s = [trial['ssp']['mahalanobis_error'] for trial in trials]
            frob_a = [trial['annealing']['frobenius_error'] for trial in trials]
            frob_s = [trial['ssp']['frobenius_error'] for trial in trials]
            
            # Only remove outliers for Adult dataset (has 1 clear failure at rho=0.0001)
            # Other datasets show systematic issues, not random outliers
            if dataset_name == 'adult':
                mahal_a_clean = [x for x in mahal_a if x < np.mean(mahal_a) + 2*np.std(mahal_a)]
                frob_a_clean = [x for x in frob_a if x < np.mean(frob_a) + 2*np.std(frob_a)]
                n_removed_m = len(mahal_a) - len(mahal_a_clean)
                n_removed_f = len(frob_a) - len(frob_a_clean)
            else:
                # Use all data for other datasets
                mahal_a_clean = mahal_a
                frob_a_clean = frob_a
                n_removed_m = 0
                n_removed_f = 0
            
            mahal_annealing_clean.append({
                'mean': np.mean(mahal_a_clean),
                'se': np.std(mahal_a_clean) / np.sqrt(len(mahal_a_clean)),
                'n_removed': n_removed_m
            })
            mahal_ssp.append({
                'mean': np.mean(mahal_s),
                'se': np.std(mahal_s) / np.sqrt(len(mahal_s))
            })
            frob_annealing_clean.append({
                'mean': np.mean(frob_a_clean),
                'se': np.std(frob_a_clean) / np.sqrt(len(frob_a_clean)),
                'n_removed': n_removed_f
            })
            frob_ssp.append({
                'mean': np.mean(frob_s),
                'se': np.std(frob_s) / np.sqrt(len(frob_s))
            })
        
        # Plot Mahalanobis (left column)
        ax = axes[dataset_idx, 0]
        mahal_a_means = [m['mean'] for m in mahal_annealing_clean]
        mahal_a_ses = [m['se'] for m in mahal_annealing_clean]
        mahal_s_means = [m['mean'] for m in mahal_ssp]
        mahal_s_ses = [m['se'] for m in mahal_ssp]
        
        ax.errorbar(rho_values, mahal_a_means, yerr=mahal_a_ses, marker='o', markersize=8,
                    linewidth=2, capsize=5, capthick=2, label='Annealing',
                    color=color_annealing, ecolor=color_annealing, alpha=0.8)
        ax.errorbar(rho_values, mahal_s_means, yerr=mahal_s_ses, marker='s', markersize=8,
                    linewidth=2, capsize=5, capthick=2, label='SSP',
                    color=color_ssp, ecolor=color_ssp, alpha=0.8)
        
        ax.set_xlabel('Privacy Budget (ρ)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Mahalanobis Error', fontsize=10, fontweight='bold')
        ax.set_title(f'{dataset_name} - Mahalanobis Error', fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        if dataset_idx == 0:
            ax.legend(fontsize=9, loc='best')
        
        # Plot Frobenius (right column)
        ax = axes[dataset_idx, 1]
        frob_a_means = [m['mean'] for m in frob_annealing_clean]
        frob_a_ses = [m['se'] for m in frob_annealing_clean]
        frob_s_means = [m['mean'] for m in frob_ssp]
        frob_s_ses = [m['se'] for m in frob_ssp]
        
        ax.errorbar(rho_values, frob_a_means, yerr=frob_a_ses, marker='o', markersize=8,
                    linewidth=2, capsize=5, capthick=2, label='Annealing',
                    color=color_annealing, ecolor=color_annealing, alpha=0.8)
        ax.errorbar(rho_values, frob_s_means, yerr=frob_s_ses, marker='s', markersize=8,
                    linewidth=2, capsize=5, capthick=2, label='SSP',
                    color=color_ssp, ecolor=color_ssp, alpha=0.8)
        
        ax.set_xlabel('Privacy Budget (ρ)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frobenius Error', fontsize=10, fontweight='bold')
        ax.set_title(f'{dataset_name} - Frobenius Error', fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        if dataset_idx == 0:
            ax.legend(fontsize=9, loc='best')
        
        # Store summary
        all_summary[dataset_name] = {
            'rho_values': rho_values,
            'mahal_annealing': mahal_annealing_clean,
            'mahal_ssp': mahal_ssp,
            'frob_annealing': frob_annealing_clean,
            'frob_ssp': frob_ssp
        }
        
        print(f"✓ Plotted {dataset_name}")

plt.tight_layout()
output_path = Path('results/all_datasets_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

# Print comprehensive summary
print("\n" + "="*100)
print("COMPREHENSIVE SUMMARY: All Datasets (Only Adult outliers removed)")
print("="*100)

for dataset_name in [d[0] for d in datasets]:
    if dataset_name not in all_summary:
        continue
    
    print(f"\n{'='*100}")
    print(f"{dataset_name.upper()}")
    print(f"{'='*100}")
    
    summary = all_summary[dataset_name]
    rho_values = summary['rho_values']
    
    for idx, rho in enumerate(rho_values):
        mahal_a = summary['mahal_annealing'][idx]
        mahal_s = summary['mahal_ssp'][idx]
        frob_a = summary['frob_annealing'][idx]
        frob_s = summary['frob_ssp'][idx]
        
        outlier_str = f" (removed {mahal_a['n_removed']} Mahal, {frob_a['n_removed']} Frob)" if (mahal_a['n_removed'] > 0 or frob_a['n_removed'] > 0) else ""
        
        print(f"\nρ = {rho}{outlier_str}")
        print(f"  Mahalanobis | Annealing: {mahal_a['mean']:.6f} ± {mahal_a['se']:.6f}  |  SSP: {mahal_s['mean']:.6f} ± {mahal_s['se']:.6f}  |  Improvement: {((mahal_s['mean'] - mahal_a['mean'])/mahal_s['mean']*100):+.1f}%")
        print(f"  Frobenius   | Annealing: {frob_a['mean']:.6f} ± {frob_a['se']:.6f}  |  SSP: {frob_s['mean']:.6f} ± {frob_s['se']:.6f}  |  Improvement: {((frob_s['mean'] - frob_a['mean'])/frob_s['mean']*100):+.1f}%")

print("\n" + "="*100)
print("Plots saved to: results/all_datasets_comparison.png")
print("="*100)
