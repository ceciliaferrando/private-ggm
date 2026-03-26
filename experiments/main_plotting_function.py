import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import os

# Find all result files and extract dataset info
results_dir = Path(__file__).parent.parent / 'results' / 'new_solvers'

datasets = []
for json_file in sorted(results_dir.glob('annealing_privacy_analysis*.json')):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        dataset_name = data['parameters'].get('dataset', json_file.name)
        d = data['parameters'].get('d', None)
        
        # If d not in parameters, try to extract from filename
        if d is None:
            try:
                d = int(json_file.name.split('_d')[1].split('_')[0])
            except:
                d = 0
        
        datasets.append((dataset_name, f'd{d}', d, f"{dataset_name}\n(d={d})", str(json_file)))
    except Exception as e:
        print(f"Warning: Could not load {json_file.name}: {e}")

# Sort by dimension d
datasets.sort(key=lambda x: x[2])

# Remove bankruptcy and volkert
datasets = [d for d in datasets if d[0] not in ['bankruptcy', 'volkert']]

print(f"Found {len(datasets)} datasets ordered by d: {[d[0] for d in datasets]}")

# Privacy budgets to plot
rho_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]

# Create 3 plots with up to 5 datasets each
n_datasets_per_plot = 5
n_rows = 2

# Group datasets into chunks
dataset_groups = [datasets[i:i+n_datasets_per_plot] for i in range(0, len(datasets), n_datasets_per_plot)]

for group_idx, dataset_group in enumerate(dataset_groups):
    n_cols = len(dataset_group)
    figsize_width = n_cols * 2.2
    figsize_height = n_rows * 1.8
    
    # Create figure for this group
    fig = plt.figure(figsize=(figsize_width, figsize_height))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.4)
    
    for plot_idx, (dataset_name, dim, d_val, display_name, filepath) in enumerate(dataset_group):
        col_idx = plot_idx % n_cols
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract data for each rho value
            annealing_frob_means = []
            annealing_frob_stds = []
            annealing_mah_means = []
            annealing_mah_stds = []
            
            baseline_frob_means = []
            baseline_frob_stds = []
            baseline_mah_means = []
            baseline_mah_stds = []
            
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
                        
                        # Baseline data (may be None)
                        baseline_frob = [t['baseline']['frobenius_error'] for t in trials if t['baseline']['frobenius_error'] is not None]
                        baseline_mah = [t['baseline']['mahalanobis_error'] for t in trials if t['baseline']['mahalanobis_error'] is not None]
                        
                        # SSP data
                        ssp_frob = [t['ssp']['frobenius_error'] for t in trials]
                        ssp_mah = [t['ssp']['mahalanobis_error'] for t in trials]
                        
                        # Compute means and stds
                        annealing_frob_means.append(np.mean(ann_frob))
                        annealing_frob_stds.append(np.std(ann_frob))
                        annealing_mah_means.append(np.mean(ann_mah))
                        annealing_mah_stds.append(np.std(ann_mah))
                        
                        if len(baseline_frob) > 0:
                            baseline_frob_means.append(np.mean(baseline_frob))
                            baseline_frob_stds.append(np.std(baseline_frob))
                            baseline_mah_means.append(np.mean(baseline_mah))
                            baseline_mah_stds.append(np.std(baseline_mah))
                        else:
                            baseline_frob_means.append(np.nan)
                            baseline_frob_stds.append(0)
                            baseline_mah_means.append(np.nan)
                            baseline_mah_stds.append(0)
                        
                        ssp_frob_means.append(np.mean(ssp_frob))
                        ssp_frob_stds.append(np.std(ssp_frob))
                        ssp_mah_means.append(np.mean(ssp_mah))
                        ssp_mah_stds.append(np.std(ssp_mah))
                        
                        valid_rhos.append(rho)
            
            if len(valid_rhos) == 0:
                raise ValueError(f"No valid rho values found for {dataset_name}")
            
            # Convert to numpy arrays
            valid_rhos = np.array(valid_rhos)
            annealing_frob_means = np.array(annealing_frob_means)
            annealing_frob_stds = np.array(annealing_frob_stds)
            annealing_mah_means = np.array(annealing_mah_means)
            annealing_mah_stds = np.array(annealing_mah_stds)
            
            baseline_frob_means = np.array(baseline_frob_means)
            baseline_frob_stds = np.array(baseline_frob_stds)
            baseline_mah_means = np.array(baseline_mah_means)
            baseline_mah_stds = np.array(baseline_mah_stds)
            
            ssp_frob_means = np.array(ssp_frob_means)
            ssp_frob_stds = np.array(ssp_frob_stds)
            ssp_mah_means = np.array(ssp_mah_means)
            ssp_mah_stds = np.array(ssp_mah_stds)
            
            # Check for valid data (all finite and positive)
            if (not np.all(np.isfinite(annealing_mah_means)) or not np.all(annealing_mah_means > 0) or
                not np.all(np.isfinite(ssp_mah_means)) or not np.all(ssp_mah_means > 0)):
                raise ValueError(f"Invalid Mahalanobis data for {dataset_name}")
            if (not np.all(np.isfinite(annealing_frob_means)) or not np.all(annealing_frob_means > 0) or
                not np.all(np.isfinite(ssp_frob_means)) or not np.all(ssp_frob_means > 0)):
                raise ValueError(f"Invalid Frobenius data for {dataset_name}")
            
            # Row 0: Mahalanobis Error
            ax0 = fig.add_subplot(gs[0, col_idx])
            # PACE-GGM in blue with fill_between for error bands (1 std dev)
            ax0.plot(valid_rhos, annealing_mah_means, marker='o', label='PACE-GGM', 
                     linewidth=1.5, markersize=4, color='#1f77b4')
            ax0.fill_between(valid_rhos, annealing_mah_means - annealing_mah_stds, 
                            annealing_mah_means + annealing_mah_stds, alpha=0.2, color='#1f77b4')
            # Baseline (diagonal-only) in green with dashes
            if not np.all(np.isnan(baseline_mah_means)):
                ax0.plot(valid_rhos, baseline_mah_means, marker='^', label='PACE-GGM (diagonal)', 
                         linewidth=1.5, markersize=4, color='#2ca02c', linestyle='--')
                ax0.fill_between(valid_rhos, baseline_mah_means - baseline_mah_stds, 
                                baseline_mah_means + baseline_mah_stds, alpha=0.2, color='#2ca02c')
            # SSP in red with fill_between for error bands (1 std dev)
            ax0.plot(valid_rhos, ssp_mah_means, marker='s', label='SSP', 
                     linewidth=1.5, markersize=4, color='#d62728')
            ax0.fill_between(valid_rhos, ssp_mah_means - ssp_mah_stds, 
                            ssp_mah_means + ssp_mah_stds, alpha=0.2, color='#d62728')
            ax0.set_xscale('log')
            ax0.set_yscale('log')
            ax0.tick_params(labelsize=8)
            if col_idx == 0:
                ax0.set_ylabel('Mahalanobis Error', fontsize=9, fontweight='bold')
            if plot_idx == 0:  # Legend in first plot
                ax0.legend(loc='best', fontsize=7, framealpha=0.9)
            ax0.set_title(f"{dataset_name} (d={d_val})", fontsize=10, fontweight='bold')
            ax0.grid(False)
            
            # Row 1: Frobenius Error
            ax1 = fig.add_subplot(gs[1, col_idx])
            # PACE-GGM in blue with fill_between for error bands (1 std dev)
            ax1.plot(valid_rhos, annealing_frob_means, marker='o', label='PACE-GGM', 
                     linewidth=1.5, markersize=4, color='#1f77b4')
            ax1.fill_between(valid_rhos, annealing_frob_means - annealing_frob_stds, 
                            annealing_frob_means + annealing_frob_stds, alpha=0.2, color='#1f77b4')
            # Baseline (diagonal-only) in green with dashes
            if not np.all(np.isnan(baseline_frob_means)):
                ax1.plot(valid_rhos, baseline_frob_means, marker='^', label='PACE-GGM (diagonal)', 
                         linewidth=1.5, markersize=4, color='#2ca02c', linestyle='--')
                ax1.fill_between(valid_rhos, baseline_frob_means - baseline_frob_stds, 
                                baseline_frob_means + baseline_frob_stds, alpha=0.2, color='#2ca02c')
            # SSP in red with fill_between for error bands (1 std dev)
            ax1.plot(valid_rhos, ssp_frob_means, marker='s', label='SSP', 
                     linewidth=1.5, markersize=4, color='#d62728')
            ax1.fill_between(valid_rhos, ssp_frob_means - ssp_frob_stds, 
                            ssp_frob_means + ssp_frob_stds, alpha=0.2, color='#d62728')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.tick_params(labelsize=8)
            ax1.set_xlabel('ρ', fontsize=9, fontweight='bold')
            if col_idx == 0:
                ax1.set_ylabel('Frobenius Error', fontsize=9, fontweight='bold')
            ax1.grid(False)
            
        except Exception as e:
            print(f'WARNING: Error processing {dataset_name}: {e}')
            # Add empty plots with error message
            ax0 = fig.add_subplot(gs[0, col_idx])
            ax0.text(0.5, 0.5, 'Data Error', ha='center', va='center', fontsize=8)
            ax0.set_title(f"{dataset_name} (d={d_val})", fontsize=10, fontweight='bold')
            
            ax1 = fig.add_subplot(gs[1, col_idx])
            ax1.text(0.5, 0.5, 'Data Error', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    output_dir = results_dir
    output_pdf = output_dir / 'adult_new_solver.pdf'
    output_png = output_dir / 'adult_new_solver.png'
    
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f'Group {group_idx+1} saved to:')
    print(f'  {output_pdf.name}')
    print(f'  {output_png.name}')
    plt.close()
