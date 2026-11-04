"""
Plot hyperparameter robustness: all alpha/beta combinations vs SSP baseline.
One subplot per dataset, one curve per (alpha, beta) combination.

Handles two directory layouts:
  <grid>/ipm_solver/<combo>/*.json     (older final_hyperparam runs)
  <grid>/<combo>/ipm_solver/*.json     (ab_grid runs)

and two combo namings: 'alpha0.1_beta0.1' and 'a01_b01'.

Several grid folders can be given at once, so that a sweep run in pieces
(e.g. the beta extension to 0.7 and 0.9, which covers only alpha 0.1 and 0.3)
plots alongside the main grid. Combos are keyed by (alpha, beta), so a combo
present in more than one folder is taken from the last one given.

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
from matplotlib import ticker
from pathlib import Path

RHO_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]

ap = argparse.ArgumentParser()
ap.add_argument('--grid-dir', nargs='+', default=['results/final_hyperparam'],
                help='One or more folders holding alpha/beta combo subdirectories; '
                     'later folders win on a repeated (alpha, beta)')
ap.add_argument('--ssp-dir', default='results/a30_b05_final_pubcov/ipm_solver',
                help='Folder to read the SSP reference curve from')
ap.add_argument('--datasets', nargs='*', default=None,
                help='Datasets to plot (default: auto-discover from the grid)')
ap.add_argument('--output-dir', default=None, help='Default: --grid-dir')
ap.add_argument('--output-tag', default='hyperparam_robustness')
ap.add_argument('--datasets-per-row', type=int, default=None,
                help='Wrap datasets across multiple row-pairs (default: all in one row)')
ap.add_argument('--group-sizes', nargs='+', type=int, default=None,
                help='Explicit datasets per block, e.g. 5 5 3. Overrides '
                     '--datasets-per-row and allows an uneven last block')
ap.add_argument('--hspace', type=float, default=0.18,
                help='Vertical gap between the two metric rows within a block')
ap.add_argument('--block-gap', type=float, default=0.34,
                help='Vertical gap between dataset blocks (must clear titles)')
ap.add_argument('--wspace', type=float, default=0.12, help='Horizontal gap between columns')
ap.add_argument('--cell-size', type=float, default=2.2, help='Inches per panel')
ap.add_argument('--legend-size', type=float, default=9.0, help='Legend font size')
ap.add_argument('--legend-gap', type=float, default=0.6,
                help='Inches reserved below the last panel row for its tick labels, '
                     'the rho label and the legend. GridSpec otherwise reserves 11%% of '
                     'the figure height, a large hole on a tall multi-block figure')
ap.add_argument('--color-by', choices=['combo', 'beta'], default='combo',
                help="'combo' gives every curve its own colour (fine up to ~9); "
                     "'beta' colours by beta and distinguishes alpha by linestyle, "
                     "which stays readable once the beta extension is included")
args = ap.parse_args()

GRID_DIRS  = [Path(g) for g in args.grid_dir]
PUBCOV_DIR = Path(args.ssp_dir)
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else GRID_DIRS[0]


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


combo_dirs = {}
for g in GRID_DIRS:
    for combo, leaf in _combo_dirs(g).items():
        # Key on (alpha, beta) rather than the directory name, so the two combo
        # namings cannot produce duplicate curves for the same setting.
        combo_dirs[_ab(combo)] = leaf
combo_dirs = dict(sorted(combo_dirs.items(), key=lambda kv: (float(kv[0][0]),
                                                             float(kv[0][1]))))
combos = list(combo_dirs)
if not combos:
    raise SystemExit(f'No alpha/beta combo directories found under '
                     f'{[str(g) for g in GRID_DIRS]}')
print(f"Hyperparameter combos: {['a=%s b=%s' % c for c in combos]}")

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

alphas = sorted({a for a, _ in combos}, key=float)
betas  = sorted({b for _, b in combos}, key=float)
alpha_ls = {a: ls for a, ls in zip(alphas, ['-', '--', ':', '-.'])}
beta_mk  = {b: mk for b, mk in zip(betas,  ['o', 's', '^', 'D', 'v'])}

if args.color_by == 'beta':
    # Colour carries beta, linestyle carries alpha. With the beta extension the
    # grid runs to 13 curves, past what a categorical map can separate, and beta
    # is the axis that actually moves the error.
    # Stop short of viridis' top end: its yellow is too pale to read on white.
    bmap = cm.get_cmap('viridis')
    beta_colors = {b: bmap(0.82 * i / max(len(betas) - 1, 1))
                   for i, b in enumerate(betas)}
    combo_colors = {c: beta_colors[c[1]] for c in combos}
else:
    cmap = cm.get_cmap('tab10', len(combos))
    combo_colors = {c: cmap(i) for i, c in enumerate(combos)}


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


# Datasets wrap across blocks; each block is a (Mahalanobis, Frobenius) row pair.
if args.group_sizes:
    if sum(args.group_sizes) != len(DATASETS):
        raise SystemExit(f'--group-sizes sums to {sum(args.group_sizes)} but there '
                         f'are {len(DATASETS)} datasets')
    sizes = list(args.group_sizes)
else:
    per_row = args.datasets_per_row or len(DATASETS)
    per_row = min(per_row, len(DATASETS))
    sizes = [per_row] * (len(DATASETS) // per_row)
    if len(DATASETS) % per_row:
        sizes.append(len(DATASETS) % per_row)

# Blocks are laid out on a grid as wide as the widest block; a shorter block
# simply leaves its trailing cells empty rather than stretching its panels.
n_cols   = max(sizes)
n_blocks = len(sizes)

# Each panel spans two half-columns, so a block narrower than the widest can be
# centred by an offset of a whole or a half column rather than flush left.
slots = []
for b, size in enumerate(sizes):
    off = n_cols - size          # in half-columns; odd values centre on a half
    slots.extend((b, off + 2 * i, off + 2 * i + 2) for i in range(size))
# Trim the bottom margin without changing panel size: GridSpec's defaults put
# the axes region in [0.11, 0.88] of the height, and panels are square, so any
# extra vertical room inflates them sideways into their neighbours. Shrink the
# figure by exactly what the margin gives up, keeping the axes region fixed.
TOP_FRAC, DEFAULT_SPAN = 0.88, 0.77
axes_inches = DEFAULT_SPAN * 2 * n_blocks * args.cell_size
fig_height = axes_inches / DEFAULT_SPAN          # provisional, for the fraction
bottom_frac = args.legend_gap / fig_height
fig_height = axes_inches / (TOP_FRAC - bottom_frac)
bottom_frac = args.legend_gap / fig_height
fig = plt.figure(figsize=(n_cols * args.cell_size, fig_height))
# Nested grids: the outer one separates the (Mahalanobis, Frobenius) blocks and
# needs a wide gap so the next block's titles clear the rho labels above them;
# the inner one packs the two metric rows together. A single GridSpec cannot do
# both, since hspace is uniform.
outer = gridspec.GridSpec(n_blocks, 1, figure=fig, hspace=args.block_gap,
                          bottom=bottom_frac, top=TOP_FRAC)
inner = [gridspec.GridSpecFromSubplotSpec(2, 2 * n_cols, subplot_spec=outer[i],
                                          hspace=args.hspace, wspace=args.wspace)
         for i in range(n_blocks)]

first_in_block = {b for b, *_ in slots}
seen_blocks = set()
for plot_idx, dataset in enumerate(DATASETS):
    block, c0, c1 = slots[plot_idx]
    is_first = block not in seen_blocks
    seen_blocks.add(block)
    ax0 = fig.add_subplot(inner[block][0, c0:c1])  # Mahalanobis
    ax1 = fig.add_subplot(inner[block][1, c0:c1])  # Frobenius

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

            a_val, b_val = combo
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
        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=8))
        ax.yaxis.set_major_formatter(
            ticker.LogFormatterSciNotation(labelOnlyBase=False))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(False)

    # pad keeps the title clear of the rho label on the panel above it
    ax0.set_title(f"{dataset}\n(d={d_val})", fontsize=7, fontweight='bold', pad=8)
    ax1.set_xlabel('ρ', fontsize=8, fontweight='bold')
    if is_first:
        ax0.set_ylabel('Mahalanobis Error', fontsize=8, fontweight='bold')
        ax1.set_ylabel('Frobenius Error', fontsize=8, fontweight='bold')
    # One legend for the whole figure, outside the axes, so it cannot sit on
    # top of the curves or collide with the y-label.
    if plot_idx == 0:
        handles, labels = ax0.get_legend_handles_labels()
        # Anchor by the legend's TOP edge so it hangs below the figure instead
        # of growing up into the margin that holds the last row's tick labels;
        # bbox_inches='tight' pulls it back into the saved image.
        fig.legend(handles, labels, loc='upper center', fontsize=args.legend_size,
                   ncol=min(len(labels), 5), framealpha=0.9, handlelength=1.8,
                   bbox_to_anchor=(0.5, -0.004))

# No tight_layout: set_box_aspect(1) makes the axes incompatible with it, and it
# pads around each square panel, which is what produced the wide gutters.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ['pdf', 'png']:
    out = OUTPUT_DIR / f'{args.output_tag}.{ext}'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close()
