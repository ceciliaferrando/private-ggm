"""
Plot hyperparameter robustness: all alpha/beta combinations vs SSP baseline.
One subplot per dataset, one curve per (alpha, beta) combination.

Handles two directory layouts:
  <grid>/ipm_solver/<combo>/*.json     (older final_hyperparam runs)
  <grid>/<combo>/ipm_solver/*.json     (ab_grid runs)

and two combo namings: 'alpha0.1_beta0.1' and 'a01_b01'.

    python -m experiments.plot_hyperparam_robustness \
        --grid-dir results/cr_ab_grid_floor \
        --ssp-dir results/cr_floor_12datasets \
        --output-dir results/0_cr_final --output-tag abgrid
"""
import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from pathlib import Path

RHO_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]

ap = argparse.ArgumentParser()
ap.add_argument('--grid-dir', default='results/final_hyperparam',
                help='Folder holding the alpha/beta combo subdirectories')
ap.add_argument('--ssp-dir', default='results/a30_b05_final_pubcov/ipm_solver',
                help='Folder to read the SSP reference curve from')
ap.add_argument('--datasets', nargs='*', default=None,
                help='Datasets to plot (default: auto-discover from the grid)')
ap.add_argument('--output-dir', default=None, help='Default: --grid-dir')
ap.add_argument('--output-tag', default='hyperparam_robustness')
args = ap.parse_args()

GRID_DIR   = Path(args.grid_dir)
PUBCOV_DIR = Path(args.ssp_dir)
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else GRID_DIR


def _combo_dirs(root):
    """Return {combo_name: dir holding the jsons}, for either layout."""
    found = {}
    inner = root / 'ipm_solver'
    candidates = [p for p in (inner if inner.is_dir() else root).iterdir() if p.is_dir()]
    for p in candidates:
        if p.name == 'ipm_solver':
            continue
        leaf = p / 'ipm_solver' if (p / 'ipm_solver').is_dir() else p
        if list(leaf.glob('annealing_privacy_analysis_*.json')):
            found[p.name] = leaf
    return dict(sorted(found.items()))


def _ab(combo):
    """Pull (alpha, beta) out of 'alpha0.1_beta0.1' or 'a01_b01'."""
    m = re.match(r'alpha([\d.]+)_beta([\d.]+)$', combo)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'a(\d+)_b(\d+)$', combo)
    if m:
        # a01 -> 0.1, a05 -> 0.5
        return (f'0.{m.group(1).lstrip("0") or "0"}',
                f'0.{m.group(2).lstrip("0") or "0"}')
    return combo, combo


combo_dirs = _combo_dirs(GRID_DIR)
combos = list(combo_dirs)
if not combos:
    raise SystemExit(f'No alpha/beta combo directories found under {GRID_DIR}')
print(f"Hyperparameter combos: {combos}")

if args.datasets:
    DATASETS = args.datasets
else:
    names = set()
    for leaf in combo_dirs.values():
        for f in leaf.glob('annealing_privacy_analysis_*.json'):
            names.add(json.load(open(f))['parameters']['dataset'])
    order = {}
    for leaf in combo_dirs.values():
        for f in leaf.glob('annealing_privacy_analysis_*.json'):
            p = json.load(open(f))['parameters']
            order[p['dataset']] = p['d']
    DATASETS = sorted(names, key=lambda n: order.get(n, 0))
print(f"Datasets ordered by d: {DATASETS}")

# Color map: one color per combo
cmap = cm.get_cmap('tab10', len(combos))
combo_colors = {c: cmap(i) for i, c in enumerate(combos)}

# Linestyle by alpha, marker by beta
alphas = sorted(set(_ab(c)[0] for c in combos))
betas  = sorted(set(_ab(c)[1] for c in combos))
alpha_ls  = {a: ls for a, ls in zip(alphas, ['-', '--', ':'])}
beta_mk   = {b: mk for b, mk in zip(betas,  ['o', 's', '^'])}


def extract_series(data, method_key, metric='mahalanobis_error'):
    centers, spreads, valid_rhos = [], [], []
    for rho in RHO_VALUES:
        trials = data['results'].get(str(rho), [])
        vals = [t[method_key][metric] for t in trials
                if method_key in t and t[method_key].get(metric) is not None]
        if not vals:
            continue
        centers.append(np.mean(vals))
        spreads.append(np.std(vals) / np.sqrt(len(vals)))
        valid_rhos.append(rho)
    return np.array(centers), np.array(spreads), np.array(valid_rhos)


n_cols = len(DATASETS)
fig = plt.figure(figsize=(n_cols * 2.5, 4.4))
gs  = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.5, wspace=0.3)

for col_idx, dataset in enumerate(DATASETS):
    ax0 = fig.add_subplot(gs[0, col_idx])  # Mahalanobis
    ax1 = fig.add_subplot(gs[1, col_idx])  # Frobenius

    ssp_files = list(PUBCOV_DIR.glob(f'annealing_privacy_analysis_{dataset}_*.json'))
    ssp_data = json.load(open(ssp_files[0])) if ssp_files else None
    d_val = ssp_data['parameters']['d'] if ssp_data else '?'

    for ax, metric in [(ax0, 'mahalanobis_error'), (ax1, 'frobenius_error')]:

        # ── SSP ──────────────────────────────────────────────────────────────
        if ssp_data:
            ssp_c, ssp_s, ssp_rhos = extract_series(ssp_data, 'ssp', metric)
            if len(ssp_rhos):
                lo = np.maximum(ssp_c - ssp_s, ssp_c * 1e-3)
                ax.plot(ssp_rhos, ssp_c, color='#d62728', linewidth=2,
                        label='SSP', zorder=5, marker='s', markersize=3)
                ax.fill_between(ssp_rhos, lo, ssp_c + ssp_s,
                                alpha=0.15, color='#d62728', zorder=4)

        # ── One curve per (alpha, beta) combo ────────────────────────────────
        for combo in combos:
            files = list(combo_dirs[combo].glob(
                f'annealing_privacy_analysis_{dataset}_*.json'))
            if not files:
                continue
            data = json.load(open(files[0]))
            c, s, rhos = extract_series(data, 'annealing', metric)
            if not len(rhos):
                continue

            a_val, b_val = _ab(combo)
            color  = combo_colors[combo]
            ls     = alpha_ls[a_val]
            marker = beta_mk[b_val]
            label  = f'α={a_val} β={b_val}'

            lo = np.maximum(c - s, c * 1e-3)
            ax.plot(rhos, c, color=color, linewidth=1.2, linestyle=ls,
                    marker=marker, markersize=3, label=label, zorder=3)
            ax.fill_between(rhos, lo, c + s, alpha=0.08, color=color, zorder=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(False)

    ax0.set_title(f"{dataset}\n(d={d_val})", fontsize=7, fontweight='bold')
    ax1.set_xlabel('ρ', fontsize=8, fontweight='bold')
    if col_idx == 0:
        ax0.set_ylabel('Mahalanobis Error', fontsize=8, fontweight='bold')
        ax1.set_ylabel('Frobenius Error', fontsize=8, fontweight='bold')
        ax0.legend(loc='upper right', fontsize=5, framealpha=0.9,
                   ncol=2, handlelength=1.5)

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ['pdf', 'png']:
    out = OUTPUT_DIR / f'{args.output_tag}.{ext}'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close()
