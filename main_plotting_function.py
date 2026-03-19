import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Dataset information - just adult for now
datasets = [
    ('adult', 'd6', 'Adult\n(d=6)')
]

# Privacy budgets to plot
rho_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]

# Create compact figure with 2:4 aspect ratio (2 rows instead of 3)
fig = plt.figure(figsize=(8, 4))  # Smaller height for 2 rows
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

for col_idx, (dataset_name, dim, display_name) in enumerate(datasets):
    filepath = f'results/adult_gold_standard_1/annealing_privacy_analysis_{dataset_name}_{dim}_alpha0.2_remeasure.json'
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
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
        
        # Row 0: Mahalanobis Error
        ax0 = fig.add_subplot(gs[0, col_idx])
        # PrivateGGM in blue with fill_between for error bands (1 std dev)
        ax0.plot(valid_rhos, annealing_mah_means, marker='o', label='PrivateGGM', 
                 linewidth=1.5, markersize=4, color='#1f77b4')
        ax0.fill_between(valid_rhos, annealing_mah_means - annealing_mah_stds, 
                        annealing_mah_means + annealing_mah_stds, alpha=0.2, color='#1f77b4')
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
        if col_idx == 0:  # Legend in upper right plot
            ax0.legend(loc='best', fontsize=7, framealpha=0.9)
        ax0.set_title(display_name, fontsize=9, fontweight='bold')
        ax0.grid(False)
        
        # Row 1: Frobenius Error
        ax1 = fig.add_subplot(gs[1, col_idx])
        # PrivateGGM in blue with fill_between for error bands (1 std dev)
        ax1.plot(valid_rhos, annealing_frob_means, marker='o', label='PrivateGGM', 
                 linewidth=1.5, markersize=4, color='#1f77b4')
        ax1.fill_between(valid_rhos, annealing_frob_means - annealing_frob_stds, 
                        annealing_frob_means + annealing_frob_stds, alpha=0.2, color='#1f77b4')
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
        
    except FileNotFoundError:
        print(f'WARNING: File not found: {filepath}')
        # Add empty plots with error message
        ax0 = fig.add_subplot(gs[0, col_idx])
        ax0.text(0.5, 0.5, 'Data Not Found', ha='center', va='center')
        ax0.set_title(display_name, fontsize=9, fontweight='bold')
        
        ax1 = fig.add_subplot(gs[1, col_idx])
        ax1.text(0.5, 0.5, 'Data Not Found', ha='center', va='center')

plt.tight_layout()
plt.savefig('grid_comparison_2row.pdf', dpi=300, bbox_inches='tight')
plt.savefig('grid_comparison_2row.png', dpi=300, bbox_inches='tight')
print('Plots saved as grid_comparison_2row.pdf and grid_comparison_2row.png')
plt.close()
