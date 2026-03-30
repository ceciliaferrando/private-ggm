#!/usr/bin/env python3
"""
Generate gridplot with all datasets that have available results.
Output: gridplot_0323.png and gridplot_0323.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import os

# Find all result files in goldstandard_linesearch directory
results_dir = Path(__file__).parent.parent / 'results' / 'goldstandard_linesearch'

# Collect all datasets with annealing_privacy_analysis results
datasets = []
result_files = {}

for json_file in sorted(results_dir.glob('annealing_privacy_analysis*.json')):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        dataset_name = data['parameters'].get('dataset', json_file.name)
        d = data['parameters'].get('d', 'unknown')
        
        result_files[dataset_name] = {
            'file': json_file,
            'data': data,
            'd': d
        }
        datasets.append((dataset_name, d, f"{dataset_name}\n(d={d})"))
    except Exception as e:
        print(f"Error loading {json_file.name}: {e}")

datasets = sorted(datasets, key=lambda x: x[0])  # Sort by dataset name

print(f"Found {len(datasets)} datasets: {', '.join([d[0] for d in datasets])}")

# Privacy budgets to plot
rho_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]

# Calculate grid dimensions
n_datasets = len(datasets)
n_cols = 5
n_rows = 2
figsize_width = n_cols * 2.2
figsize_height = n_rows * 1.8

# Create figure with gridspec
fig = plt.figure(figsize=(figsize_width, figsize_height))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.4)

for col_idx, (dataset_name, d, display_name) in enumerate(datasets):
    if col_idx >= n_cols * n_rows:
        break  # Skip if more datasets than space
    
    row_idx = col_idx // n_cols
    col_in_grid = col_idx % n_cols
    
    try:
        data = result_files[dataset_name]['data']
        
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
            print(f"Warning: No valid rhos for {dataset_name}")
            continue
        
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
        
        # Check for NaN or non-positive values
        if (np.any(~np.isfinite(annealing_mah_means)) or np.any(annealing_mah_means <= 0) or
            np.any(~np.isfinite(ssp_mah_means)) or np.any(ssp_mah_means <= 0) or
            np.any(~np.isfinite(annealing_frob_means)) or np.any(annealing_frob_means <= 0) or
            np.any(~np.isfinite(ssp_frob_means)) or np.any(ssp_frob_means <= 0)):
            print(f"Warning: Invalid data (NaN or non-positive) for {dataset_name}")
            continue
        
        # Top row: Mahalanobis Error
        if row_idx == 0:
            ax = fig.add_subplot(gs[row_idx, col_in_grid])
            # PrivateGGM in blue with fill_between for error bands
            ax.plot(valid_rhos, annealing_mah_means, marker='o', label='PrivateGGM', 
                   linewidth=1.5, markersize=4, color='#1f77b4')
            ax.fill_between(valid_rhos, annealing_mah_means - annealing_mah_stds, 
                           annealing_mah_means + annealing_mah_stds, alpha=0.2, color='#1f77b4')
            # SSP in red with fill_between for error bands
            ax.plot(valid_rhos, ssp_mah_means, marker='s', label='SSP', 
                   linewidth=1.5, markersize=4, color='#d62728')
            ax.fill_between(valid_rhos, ssp_mah_means - ssp_mah_stds, 
                           ssp_mah_means + ssp_mah_stds, alpha=0.2, color='#d62728')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(labelsize=7)
            ax.set_title(display_name, fontsize=8, fontweight='bold')
            if col_in_grid == 0:
                ax.set_ylabel('Mahalanobis Error', fontsize=8, fontweight='bold')
            if col_idx == 0:
                ax.legend(loc='best', fontsize=7, framealpha=0.9)
            ax.grid(False)
        
        # Bottom row: Frobenius Error
        else:
            ax = fig.add_subplot(gs[row_idx, col_in_grid])
            # PrivateGGM in blue with fill_between for error bands
            ax.plot(valid_rhos, annealing_frob_means, marker='o', label='PrivateGGM', 
                   linewidth=1.5, markersize=4, color='#1f77b4')
            ax.fill_between(valid_rhos, annealing_frob_means - annealing_frob_stds, 
                           annealing_frob_means + annealing_frob_stds, alpha=0.2, color='#1f77b4')
            # SSP in red with fill_between for error bands
            ax.plot(valid_rhos, ssp_frob_means, marker='s', label='SSP', 
                   linewidth=1.5, markersize=4, color='#d62728')
            ax.fill_between(valid_rhos, ssp_frob_means - ssp_frob_stds, 
                           ssp_frob_means + ssp_frob_stds, alpha=0.2, color='#d62728')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(labelsize=7)
            ax.set_xlabel('ρ', fontsize=8, fontweight='bold')
            if col_in_grid == 0:
                ax.set_ylabel('Frobenius Error', fontsize=8, fontweight='bold')
            ax.grid(False)
    
    except Exception as e:
        print(f'Error processing {dataset_name}: {e}')
        ax = fig.add_subplot(gs[row_idx, col_in_grid])
        ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', fontsize=8)
        ax.set_title(display_name, fontsize=8, fontweight='bold')

plt.suptitle('PrivateGGM vs SSP: Mahalanobis and Frobenius Errors', 
             fontsize=11, fontweight='bold', y=0.995)

# Save the plot
output_dir = results_dir
output_png = output_dir / 'gridplot_0323.png'
output_pdf = output_dir / 'gridplot_0323.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')

print(f'Gridplot saved to:')
print(f'  PNG: {output_png}')
print(f'  PDF: {output_pdf}')
plt.close()
