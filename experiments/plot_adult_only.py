#!/usr/bin/env python
"""Plot adult dataset results from new_solvers folder"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path

# Point to the new_solvers results directory
results_dir = Path(__file__).parent.parent / 'results' / 'new_solvers'

# Find the adult results file
adult_file = results_dir / 'annealing_privacy_analysis_adult_d6_alpha0.2_remeasure.json'

if not adult_file.exists():
    print(f"Error: Could not find {adult_file}")
    exit(1)

print(f"Loading results from {adult_file}")

with open(adult_file, 'r') as f:
    data = json.load(f)

# Privacy budgets to plot
rho_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]

# Extract data for each rho value
annealing_frob_means = []
annealing_frob_stds = []
annealing_mah_means = []
annealing_mah_stds = []

ssp_frob_means = []
ssp_frob_stds = []
ssp_mah_means = []
ssp_mah_stds = []

valid_rhos = []

for rho in rho_values:
    rho_key = str(rho)
    if rho_key in data['results']:
        trials = data['results'][rho_key]
        
        if isinstance(trials, list):
            # Annealing data
            ann_frob = [t['annealing']['frobenius_error'] for t in trials]
            ann_mah = [t['annealing']['mahalanobis_error'] for t in trials]
            
            # SSP data
            ssp_frob = [t['ssp']['frobenius_error'] for t in trials]
            ssp_mah = [t['ssp']['mahalanobis_error'] for t in trials]
            
            # Compute means and stds
            annealing_frob_means.append(np.mean(ann_frob))
            annealing_frob_stds.append(np.std(ann_frob))
            annealing_mah_means.append(np.mean(ann_mah))
            annealing_mah_stds.append(np.std(ann_mah))
            
            ssp_frob_means.append(np.mean(ssp_frob))
            ssp_frob_stds.append(np.std(ssp_frob))
            ssp_mah_means.append(np.mean(ssp_mah))
            ssp_mah_stds.append(np.std(ssp_mah))
            
            valid_rhos.append(rho)

if len(valid_rhos) == 0:
    print("Error: No valid rho values found")
    exit(1)

# Convert to numpy arrays
valid_rhos = np.array(valid_rhos)
annealing_frob_means = np.array(annealing_frob_means)
annealing_frob_stds = np.array(annealing_frob_stds)
annealing_mah_means = np.array(annealing_mah_means)
annealing_mah_stds = np.array(annealing_mah_stds)

ssp_frob_means = np.array(ssp_frob_means)
ssp_frob_stds = np.array(ssp_frob_stds)
ssp_mah_means = np.array(ssp_mah_means)
ssp_mah_stds = np.array(ssp_mah_stds)

# Create figure with 2 subplots (Mahalanobis and Frobenius)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot 1: Mahalanobis Error
ax0 = axes[0]
# PrivateGGM in blue with fill_between for error bands (1 std dev)
ax0.plot(valid_rhos, annealing_mah_means, marker='o', label='PrivateGGM', 
         linewidth=2, markersize=6, color='#1f77b4')
ax0.fill_between(valid_rhos, annealing_mah_means - annealing_mah_stds, 
                annealing_mah_means + annealing_mah_stds, alpha=0.2, color='#1f77b4')
# SSP in red with fill_between for error bands (1 std dev)
ax0.plot(valid_rhos, ssp_mah_means, marker='s', label='SSP', 
         linewidth=2, markersize=6, color='#d62728')
ax0.fill_between(valid_rhos, ssp_mah_means - ssp_mah_stds, 
                ssp_mah_means + ssp_mah_stds, alpha=0.2, color='#d62728')
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_ylabel('Mahalanobis Error', fontsize=11, fontweight='bold')
ax0.set_xlabel('Privacy Budget (ρ)', fontsize=11, fontweight='bold')
ax0.set_title('Adult Dataset - Mahalanobis Error', fontsize=12, fontweight='bold')
ax0.legend(loc='best', fontsize=10, framealpha=0.9)
ax0.grid(True, alpha=0.3, linestyle='--')
ax0.tick_params(labelsize=9)

# Plot 2: Frobenius Error
ax1 = axes[1]
# PrivateGGM in blue with fill_between for error bands (1 std dev)
ax1.plot(valid_rhos, annealing_frob_means, marker='o', label='PrivateGGM', 
         linewidth=2, markersize=6, color='#1f77b4')
ax1.fill_between(valid_rhos, annealing_frob_means - annealing_frob_stds, 
                annealing_frob_means + annealing_frob_stds, alpha=0.2, color='#1f77b4')
# SSP in red with fill_between for error bands (1 std dev)
ax1.plot(valid_rhos, ssp_frob_means, marker='s', label='SSP', 
         linewidth=2, markersize=6, color='#d62728')
ax1.fill_between(valid_rhos, ssp_frob_means - ssp_frob_stds, 
                ssp_frob_means + ssp_frob_stds, alpha=0.2, color='#d62728')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('Frobenius Error', fontsize=11, fontweight='bold')
ax1.set_xlabel('Privacy Budget (ρ)', fontsize=11, fontweight='bold')
ax1.set_title('Adult Dataset - Frobenius Error', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=9)

plt.tight_layout()

# Save figures
output_pdf = results_dir / 'adult_results.pdf'
output_png = results_dir / 'adult_results.png'

plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f'\nPlots saved to:')
print(f'  {output_pdf.name}')
print(f'  {output_png.name}')
plt.show()
