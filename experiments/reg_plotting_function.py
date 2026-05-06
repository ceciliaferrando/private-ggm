import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import argparse

def plot_results_from_folder(results_dir, exclude_datasets=None):
    if exclude_datasets is None:
        exclude_datasets = []

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    datasets = []
    for json_file in sorted(results_dir.glob('annealing_privacy_analysis*.json')):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            dataset_name = data['parameters'].get('dataset', json_file.name)
            d = data['parameters'].get('d', None)
            if d is None:
                try:
                    d = int(json_file.name.split('_d')[1].split('_')[0])
                except:
                    d = 0
            # Only include files that have regression results
            has_reg = False
            for rho_trials in data['results'].values():
                if rho_trials and 'mse' in rho_trials[0].get('annealing', {}):
                    has_reg = True
                    break
            if not has_reg:
                print(f"Skipping {dataset_name}: no regression results found")
                continue
            datasets.append((dataset_name, f'd{d}', d, f"{dataset_name}\n(d={d})", str(json_file)))
        except Exception as e:
            print(f"Warning: Could not load {json_file.name}: {e}")

    datasets.sort(key=lambda x: x[2])
    datasets = [d for d in datasets if d[0] not in exclude_datasets]
    print(f"Found {len(datasets)} datasets with regression results: {[d[0] for d in datasets]}")
    return results_dir, datasets


def _load_ihm_folder(ihm_folder, dataset_name):
    """Load IHM/CoinPress baseline JSON for a given dataset from ihm_folder.

    Returns dict: method_key -> list of per-rho MSE arrays, or None if not found.
    """
    if ihm_folder is None:
        return {}
    ihm_dir = Path(ihm_folder)
    # Try to find a matching JSON for this dataset
    matches = list(ihm_dir.glob(f'reg_baselines_{dataset_name}_*.json'))
    if not matches:
        return {}
    with open(matches[0]) as f:
        data = json.load(f)
    return data


def _extract_mse_series(data, rho_values, method_key, stat):
    """Extract (centers, spreads, valid_rhos) for a given method key from a results dict."""
    centers, spreads, valid_rhos = [], [], []
    for rho in rho_values:
        rho_key = str(rho)
        if rho_key not in data['results']:
            continue
        trials = data['results'][rho_key]
        if not isinstance(trials, list) or not trials:
            continue
        vals = [t[method_key]['mse'] for t in trials
                if method_key in t and t[method_key].get('mse') is not None]
        if not vals:
            continue
        if stat == 'median':
            centers.append(np.median(vals))
            spreads.append(np.subtract(*np.percentile(vals, [75, 25])))
        elif stat == 'se':
            centers.append(np.mean(vals))
            spreads.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            centers.append(np.mean(vals))
            spreads.append(np.std(vals))
        valid_rhos.append(rho)
    return np.array(centers), np.array(spreads), np.array(valid_rhos)


def _plot_line(ax, rhos, center, spread, color, label, marker='o', ls='-'):
    if len(rhos) == 0:
        return
    ax.plot(rhos, center, marker=marker, label=label,
            linewidth=1.5, markersize=4, color=color, linestyle=ls)
    lo = np.maximum(center - spread, center * 1e-3)
    ax.fill_between(rhos, lo, center + spread, alpha=0.2, color=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot regression MSE results')
    parser.add_argument('results_folder', type=str, help='Path to results folder')
    parser.add_argument('--exclude', type=str, nargs='+', default=[],
                        help='Dataset names to exclude')
    parser.add_argument('--n-per-plot', type=int, default=5,
                        help='Max datasets per plot group (default: 5)')
    parser.add_argument('--stat', type=str, default='mean', choices=['mean', 'median', 'se'],
                        help='Statistic to plot: mean±std, mean±SE, or median±IQR (default: mean)')
    parser.add_argument('--ihm-folder', type=str, default=None,
                        help='Folder with IHM/CoinPress regression baseline JSONs to overlay')
    parser.add_argument('--adassp', action='store_true', default=True,
                        help='Overlay AdaSSP from main results (default: on)')
    parser.add_argument('--no-adassp', action='store_false', dest='adassp')
    parser.add_argument('--ssp', action='store_true', default=True,
                        help='Overlay SSP from main results (default: on)')
    parser.add_argument('--no-ssp', action='store_false', dest='ssp')
    parser.add_argument('--suffix', type=str, default='',
                        help='Extra suffix appended to output filenames (e.g. _fixed)')
    parser.add_argument('--clip-ymax', type=float, default=None,
                        help='Cap y-axis at max(PACE-GGM center) * this factor')
    parser.add_argument('--n-rows', type=int, default=1,
                        help='Number of rows per figure (datasets flow left-to-right, top-to-bottom)')
    args = parser.parse_args()

    results_dir, datasets = plot_results_from_folder(args.results_folder, exclude_datasets=args.exclude)

    print(f"Plotting regression results from: {results_dir}")
    if args.ihm_folder:
        print(f"Overlaying IHM/CoinPress from: {args.ihm_folder}")

    rho_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]
    n_rows = args.n_rows
    stat = args.stat

    # Group datasets: each figure holds n_rows * n_per_plot panels
    n_per_fig = args.n_per_plot * n_rows
    dataset_groups = [datasets[i:i+n_per_fig] for i in range(0, len(datasets), n_per_fig)]

    for group_idx, dataset_group in enumerate(dataset_groups):
        n_cols = args.n_per_plot
        actual_rows = (len(dataset_group) + n_cols - 1) // n_cols
        figsize_width = n_cols * 2.8
        figsize_height = actual_rows * 2.4

        fig = plt.figure(figsize=(figsize_width, figsize_height))
        hspace = 0.6 if actual_rows > 1 else 0.4
        gs = gridspec.GridSpec(actual_rows, n_cols, figure=fig, hspace=hspace, wspace=0.15)

        for plot_idx, (dataset_name, dim, d_val, display_name, filepath) in enumerate(dataset_group):
            row_idx = plot_idx // n_cols
            col_idx = plot_idx % n_cols

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                pace_c, pace_s, valid_rhos = _extract_mse_series(data, rho_values, 'annealing', stat)
                ssp_c,  ssp_s,  _          = _extract_mse_series(data, rho_values, 'ssp',       stat)

                if len(valid_rhos) == 0:
                    raise ValueError(f"No valid rho values with MSE found for {dataset_name}")

                ax = fig.add_subplot(gs[row_idx, col_idx])

                _plot_line(ax, valid_rhos, pace_c, pace_s, '#1f77b4', 'PACE-GGM')
                if args.ssp:
                    _plot_line(ax, valid_rhos, ssp_c, ssp_s, '#d62728', 'SSP', marker='s')

                # AdaSSP from main results (if available)
                if args.adassp:
                    ada_c, ada_s, ada_rhos = _extract_mse_series(data, rho_values, 'adassp', stat)
                    if len(ada_rhos):
                        _plot_line(ax, ada_rhos, ada_c, ada_s, '#d62728', 'AdaSSP',
                                   marker='^', ls='--')

                # IHM / CoinPress overlay from --ihm-folder
                if args.ihm_folder:
                    ihm_data = _load_ihm_folder(args.ihm_folder, dataset_name)
                    if ihm_data:
                        ihm_results = ihm_data.get('results', {})
                        # Discover method keys from first available rho
                        method_keys = []
                        for rho_trials in ihm_results.values():
                            if rho_trials:
                                method_keys = list(rho_trials[0].keys())
                                break

                        method_style = {
                            'ihm':            ('#ff7f0e', 'IHM',          'D', '-.'),
                            'coinpress_t1':   ('#9467bd', 'CoinPress(t=1)', 's', ':'),
                            'coinpress_t2':   ('#8c564b', 'CoinPress(t=2)', 'v', ':'),
                            'coinpress_t4':   ('#17becf', 'CoinPress(t=4)', 'p', ':'),
                        }
                        for mk in method_keys:
                            style = method_style.get(mk, ('#7f7f7f', mk, 'x', '--'))
                            color, label, marker, ls = style
                            mc, ms, mr = _extract_mse_series(ihm_data, rho_values, mk, stat)
                            if len(mr):
                                _plot_line(ax, mr, mc, ms, color, label, marker=marker, ls=ls)

                ax.set_xscale('log')
                all_centers = [pace_c] + ([ssp_c] if args.ssp else [])
                all_pos = np.concatenate([c for c in all_centers if len(c) and np.nanmin(c) > 0])
                if len(all_pos):
                    ax.set_yscale('log')
                if args.clip_ymax is not None and len(pace_c) > 0:
                    ax.set_ylim(top=np.nanmax(pace_c) * args.clip_ymax)
                ax.yaxis.set_major_locator(plt.MaxNLocator(2))
                sparse_rhos = rho_values[::2]
                ax.set_xticks(sparse_rhos)
                ax.set_xticklabels([f'$10^{{{int(np.log10(r))}}}$' if np.log10(r) == int(np.log10(r)) else str(r)
                                    for r in sparse_rhos], rotation=45, ha='right')
                ax.tick_params(labelsize=8, which='both')
                ax.set_box_aspect(1)
                ax.set_xlabel('ρ', fontsize=8, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel('MSE', fontsize=8, fontweight='bold')
                if plot_idx == 0 and group_idx == 0:
                    ax.legend(loc='best', fontsize=6, framealpha=0.9)
                ax.set_title(f"{dataset_name}\n(d={d_val})", fontsize=7, fontweight='bold')
                ax.grid(False)

            except Exception as e:
                print(f'WARNING: Error processing {dataset_name}: {e}')
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, 'Data Error', ha='center', va='center', fontsize=8)
                ax.set_title(f"{dataset_name}\n(d={d_val})", fontsize=8, fontweight='bold')

        plt.tight_layout()
        folder_name = results_dir.name
        auto_suffix = '_ihm' if args.ihm_folder else ''
        extra_suffix = args.suffix if args.suffix else ''
        output_pdf = results_dir / f'{folder_name}_reg_{stat}{auto_suffix}{extra_suffix}_group{group_idx+1}.pdf'
        output_png = results_dir / f'{folder_name}_reg_{stat}{auto_suffix}{extra_suffix}_group{group_idx+1}.png'

        plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f'Group {group_idx+1} saved to:')
        print(f'  {output_pdf.name}')
        print(f'  {output_png.name}')
        plt.close()

    print(f"\nAll regression plots saved to: {results_dir}")
