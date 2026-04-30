import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import os
import argparse

def plot_results_from_folder(results_dir, exclude_datasets=None):
    if exclude_datasets is None:
        exclude_datasets = ['bankruptcy', 'volkert']
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
            datasets.append((dataset_name, f'd{d}', d, f"{dataset_name}\n(d={d})", str(json_file)))
        except Exception as e:
            print(f"Warning: Could not load {json_file.name}: {e}")
    datasets.sort(key=lambda x: x[2])
    datasets = [d for d in datasets if d[0] not in exclude_datasets]
    print(f"Found {len(datasets)} datasets ordered by d: {[d[0] for d in datasets]}")
    return results_dir, datasets


def load_rho_data(data, rho_values):
    """Extract per-rho means/stds for annealing, baseline, ssp from a loaded JSON."""
    ann_frob_m, ann_frob_s = [], []
    ann_mah_m,  ann_mah_s  = [], []
    bas_frob_m, bas_frob_s = [], []
    bas_mah_m,  bas_mah_s  = [], []
    ssp_frob_m, ssp_frob_s = [], []
    ssp_mah_m,  ssp_mah_s  = [], []
    valid_rhos = []

    for rho in rho_values:
        rho_key = str(rho)
        if rho_key not in data['results']:
            continue
        trials = data['results'][rho_key]
        if not isinstance(trials, list) or not trials:
            continue

        ann_frob = [t['annealing']['frobenius_error'] for t in trials]
        ann_mah  = [t['annealing']['mahalanobis_error'] for t in trials]
        bas_frob = [t['baseline']['frobenius_error'] for t in trials if t['baseline']['frobenius_error'] is not None]
        bas_mah  = [t['baseline']['mahalanobis_error'] for t in trials if t['baseline']['mahalanobis_error'] is not None]
        ssp_frob = [t['ssp']['frobenius_error'] for t in trials]
        ssp_mah  = [t['ssp']['mahalanobis_error'] for t in trials]

        ann_frob_m.append(np.mean(ann_frob)); ann_frob_s.append(np.std(ann_frob))
        ann_mah_m.append(np.mean(ann_mah));   ann_mah_s.append(np.std(ann_mah))
        ssp_frob_m.append(np.mean(ssp_frob)); ssp_frob_s.append(np.std(ssp_frob))
        ssp_mah_m.append(np.mean(ssp_mah));   ssp_mah_s.append(np.std(ssp_mah))

        if bas_frob:
            bas_frob_m.append(np.mean(bas_frob)); bas_frob_s.append(np.std(bas_frob))
            bas_mah_m.append(np.mean(bas_mah));   bas_mah_s.append(np.std(bas_mah))
        else:
            bas_frob_m.append(np.nan); bas_frob_s.append(0)
            bas_mah_m.append(np.nan);  bas_mah_s.append(0)

        valid_rhos.append(rho)

    return (np.array(valid_rhos),
            np.array(ann_frob_m), np.array(ann_frob_s),
            np.array(ann_mah_m),  np.array(ann_mah_s),
            np.array(bas_frob_m), np.array(bas_frob_s),
            np.array(bas_mah_m),  np.array(bas_mah_s),
            np.array(ssp_frob_m), np.array(ssp_frob_s),
            np.array(ssp_mah_m),  np.array(ssp_mah_s))


def load_ablation_data(data, rho_values):
    """Extract per-rho mahalanobis error, frobenius error, total solver runtime, and mean per-round solver runtime."""
    mah_m, mah_s, frob_m, frob_s, rt_m, rt_s, rt_per_m, rt_per_s, valid_rhos = [], [], [], [], [], [], [], [], []
    for rho in rho_values:
        rho_key = str(rho)
        if rho_key not in data['results']:
            continue
        trials = data['results'][rho_key]
        if not isinstance(trials, list) or not trials:
            continue
        mahs, frobs, rts, rts_per = [], [], [], []
        for t in trials:
            ann = t.get('annealing', {})
            if 'mahalanobis_error' not in ann:
                continue
            mahs.append(ann['mahalanobis_error'])
            frobs.append(ann['frobenius_error'])
            timings = ann.get('round_timings', [])
            ipm_times = [r.get('ipm_time', r.get('round_time', 0)) for r in timings]
            rts.append(sum(ipm_times))
            rts_per.append(np.mean(ipm_times) if ipm_times else 0.0)
        if not mahs:
            continue
        mah_m.append(np.mean(mahs));       mah_s.append(np.std(mahs))
        frob_m.append(np.mean(frobs));     frob_s.append(np.std(frobs))
        rt_m.append(np.mean(rts));         rt_s.append(np.std(rts))
        rt_per_m.append(np.mean(rts_per)); rt_per_s.append(np.std(rts_per))
        valid_rhos.append(rho)
    return (np.array(valid_rhos),
            np.array(mah_m),    np.array(mah_s),
            np.array(frob_m),   np.array(frob_s),
            np.array(rt_m),     np.array(rt_s),
            np.array(rt_per_m), np.array(rt_per_s))


def load_mse_data(data, rho_values):
    pace_means, pace_ses, ssp_means, ssp_ses, valid_rhos = [], [], [], [], []
    diag_means, diag_ses, adassp_means, adassp_ses = [], [], [], []
    for rho in rho_values:
        rho_key = str(rho)
        if rho_key not in data['results']:
            continue
        trials = data['results'][rho_key]
        pace_mse   = [t['annealing']['mse'] for t in trials if 'mse' in t.get('annealing', {})]
        ssp_mse    = [t['ssp']['mse']       for t in trials if 'mse' in t.get('ssp', {})]
        diag_mse   = [t['diag_xty']['mse']  for t in trials if 'mse' in t.get('diag_xty', {})]
        adassp_mse = [t['adassp']['mse']    for t in trials if 'mse' in t.get('adassp', {})]
        if not pace_mse or not ssp_mse:
            continue
        pace_means.append(np.mean(pace_mse));   pace_ses.append(np.std(pace_mse)   / np.sqrt(len(pace_mse)))
        ssp_means.append(np.mean(ssp_mse));     ssp_ses.append(np.std(ssp_mse)     / np.sqrt(len(ssp_mse)))
        diag_means.append(np.mean(diag_mse)   if diag_mse   else np.nan)
        diag_ses.append(np.std(diag_mse)   / np.sqrt(len(diag_mse))   if diag_mse   else 0)
        adassp_means.append(np.mean(adassp_mse) if adassp_mse else np.nan)
        adassp_ses.append(np.std(adassp_mse) / np.sqrt(len(adassp_mse)) if adassp_mse else 0)
        valid_rhos.append(rho)
    return (np.array(valid_rhos),
            np.array(pace_means),   np.array(pace_ses),
            np.array(ssp_means),    np.array(ssp_ses),
            np.array(diag_means),   np.array(diag_ses),
            np.array(adassp_means), np.array(adassp_ses))


def plot_line(ax, rhos, means, stds, color, label, linestyle='-', marker='o',
             clip_log=False):
    ax.plot(rhos, means, marker=marker, label=label,
            linewidth=1.5, markersize=4, color=color, linestyle=linestyle)
    lo = means - stds
    if clip_log:
        lo = np.maximum(lo, means * 1e-3)
    ax.fill_between(rhos, lo, means + stds, alpha=0.2, color=color)


def load_privcovariance_data(data, rho_values, method_key):
    """Extract per-rho mean/std for a PrivateCovariance method (adaptive_cov or coinpress).

    Args:
        data:       parsed JSON dict from a privcovariance_*.json result file
        rho_values: list of rho floats to query
        method_key: 'adaptive_cov' or 'coinpress'

    Returns:
        valid_rhos, mah_means, mah_stds, frob_means, frob_stds  (all np.arrays)
    """
    valid_rhos, mah_m, mah_s, frob_m, frob_s = [], [], [], [], []
    for rho in rho_values:
        trials = data['results'].get(str(rho), [])
        if not trials:
            continue
        mahs  = [t[method_key]['mahalanobis_error'] for t in trials
                 if t.get(method_key, {}).get('mahalanobis_error') is not None]
        frobs = [t[method_key]['frobenius_error']   for t in trials
                 if t.get(method_key, {}).get('frobenius_error')   is not None]
        if not mahs:
            continue
        mah_m.append(np.mean(mahs));  mah_s.append(np.std(mahs))
        frob_m.append(np.mean(frobs)); frob_s.append(np.std(frobs))
        valid_rhos.append(rho)
    return (np.array(valid_rhos),
            np.array(mah_m),  np.array(mah_s),
            np.array(frob_m), np.array(frob_s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot results from a results folder')
    parser.add_argument('results_folder', type=str, help='Path to results folder')
    parser.add_argument('--exclude', type=str, nargs='+',
                        default=['bankruptcy', 'volkert', 'communities_crime', 'madeline'])
    parser.add_argument('--n-per-plot', type=int, default=5)
    parser.add_argument('--group-sizes', type=int, nargs='+', default=None,
                        help='Custom group sizes e.g. --group-sizes 5 6')
    parser.add_argument('--regression', action='store_true', default=False,
                        help='Add MSE row')
    parser.add_argument('--mse-only', action='store_true', default=False,
                        help='Show only MSE row (implies --regression)')
    parser.add_argument('--compare-folder', type=str, default=None,
                        help='Second folder to overlay (dashed lines)')
    parser.add_argument('--compare-label', type=str, default='private mean',
                        help='Label suffix for compare folder lines')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Where to save plots (default: results_folder)')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Tag for output filenames')
    parser.add_argument('--rho-values', nargs='+', type=float,
                        default=[0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0])
    parser.add_argument('--ablation-ipm-vs-pgd', action='store_true', default=False,
                        help='Ablation plot: compare ipm_nocc vs pgd_nocc (mahalanobis + runtime)')
    parser.add_argument('--ablation-datasets', nargs='+', type=str, default=None,
                        help='Dataset names to include in ablation plot')
    parser.add_argument('--ablation-labels', nargs=2, type=str,
                        default=['IPM (no CC)', 'PGD (no CC)'],
                        help='Labels for the two methods in ablation plot')
    parser.add_argument('--clip-ymax', type=float, default=None,
                        help='Cap y-axis at max(SSP) * this factor on Mahalanobis and Frobenius rows')
    parser.add_argument('--privcovariance-folder', type=str, default=None,
                        help='Folder with privcovariance_*.json files to overlay '
                             'AdaptiveCov and CoinPress lines on Mahalanobis/Frobenius plots')
    parser.add_argument('--ablation-fill', action='store_true', default=False,
                        help='Ablation plot: fill_unobserved comparison. results_folder must be '
                             'the parent dir containing fu0/fu1/fu10/fu100/fu1000/ipm subfolders.')
    args = parser.parse_args()

    if args.mse_only:
        args.regression = True

    results_dir, datasets = plot_results_from_folder(args.results_folder, exclude_datasets=args.exclude)
    compare_dir = Path(args.compare_folder) if args.compare_folder else None

    rho_values     = args.rho_values
    n_per_plot     = args.n_per_plot
    show_mse       = args.regression
    mse_only       = args.mse_only
    has_compare    = compare_dir is not None
    privcov_dir    = Path(args.privcovariance_folder) if args.privcovariance_folder else None
    has_privcov    = privcov_dir is not None
    clip_ymax      = args.clip_ymax

    if mse_only:
        n_rows = 1
    elif show_mse:
        n_rows = 3
    else:
        n_rows = 2

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_name = Path(args.results_folder).name
    tag = args.output_tag or folder_name
    if has_compare:
        tag = tag + '_vs_' + Path(args.compare_folder).name

    if args.group_sizes:
        dataset_groups = []
        idx = 0
        for sz in args.group_sizes:
            dataset_groups.append(datasets[idx:idx+sz])
            idx += sz
        if idx < len(datasets):
            dataset_groups.append(datasets[idx:])
    else:
        dataset_groups = [datasets[i:i+n_per_plot] for i in range(0, len(datasets), n_per_plot)]

    for group_idx, dataset_group in enumerate(dataset_groups):
        n_cols = len(dataset_group)
        cell_size = 2.2 if not mse_only else 2.8
        fig = plt.figure(figsize=(n_cols * cell_size, n_rows * cell_size))
        gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.4)

        for plot_idx, (dataset_name, dim, d_val, display_name, filepath) in enumerate(dataset_group):
            col_idx = plot_idx % n_cols

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Load compare data if available
                cmp_data = None
                if has_compare:
                    cmp_files = list(compare_dir.glob(f'annealing_privacy_analysis_{dataset_name}_*.json'))
                    if cmp_files:
                        with open(cmp_files[0]) as f:
                            cmp_data = json.load(f)

                (rhos, af_m, af_s, am_m, am_s,
                 bf_m, bf_s, bm_m, bm_s,
                 sf_m, sf_s, sm_m, sm_s) = load_rho_data(data, rho_values)

                if len(rhos) == 0:
                    raise ValueError("No valid rho values")

                # Validate covariance data
                if not mse_only:
                    for arr, name in [(am_m,'ann_mah'),(sm_m,'ssp_mah'),(af_m,'ann_frob'),(sf_m,'ssp_frob')]:
                        if not np.all(np.isfinite(arr)) or not np.all(arr > 0):
                            raise ValueError(f"Invalid data: {name}")

                row_offset = 0

                # ── Mahalanobis row ──────────────────────────────────────────
                if not mse_only:
                    ax0 = fig.add_subplot(gs[0, col_idx])
                    plot_line(ax0, rhos, am_m, am_s, '#1f77b4', 'PACE-GGM', clip_log=True)
                    if not np.all(np.isnan(bm_m)):
                        plot_line(ax0, rhos, bm_m, bm_s, '#2ca02c', 'Independent', linestyle='--', marker='^', clip_log=True)
                    plot_line(ax0, rhos, sm_m, sm_s, '#d62728', 'SSP', marker='s', clip_log=True)

                    if has_compare and cmp_data is not None:
                        (rhos2, _, _, am2_m, am2_s, _, _, _, _, _, _, _, _) = load_rho_data(cmp_data, rho_values)
                        if len(rhos2) > 0:
                            plot_line(ax0, rhos2, am2_m, am2_s, '#1f77b4',
                                      f'PACE-GGM ({args.compare_label})', linestyle='--', clip_log=True)

                    if has_privcov:
                        pc_files = list(privcov_dir.glob(f'privcovariance_{dataset_name}_*.json'))
                        if pc_files:
                            with open(pc_files[0]) as f:
                                pc_data = json.load(f)
                            for method_key, color, label_suffix, ls in [
                                ('adaptive_cov', '#ff7f0e', 'AdaptiveCov', '-.'),
                                ('coinpress',    '#17becf', 'CoinPress',   ':'),
                            ]:
                                rpc, mm, ms, _, _ = load_privcovariance_data(pc_data, rho_values, method_key)
                                if len(rpc) > 0:
                                    plot_line(ax0, rpc, mm, ms, color, label_suffix,
                                              linestyle=ls, marker='D', clip_log=True)

                    ax0.set_xscale('log'); ax0.set_yscale('log')
                    ax0.set_box_aspect(1)
                    ax0.tick_params(labelsize=8)
                    if col_idx == 0:
                        ax0.set_ylabel('Mahalanobis Error', fontsize=9, fontweight='bold')
                    if plot_idx == 0:
                        ax0.legend(loc='best', fontsize=6, framealpha=0.9)
                    ax0.set_title(f"{dataset_name}\n(d={d_val})", fontsize=8, fontweight='bold')
                    if clip_ymax is not None and np.any(np.isfinite(sm_m)):
                        ax0.set_ylim(top=np.nanmax(sm_m) * clip_ymax)
                    ax0.grid(False)
                    row_offset = 1

                # ── Frobenius row ─────────────────────────────────────────────
                if not mse_only:
                    ax1 = fig.add_subplot(gs[1, col_idx])
                    plot_line(ax1, rhos, af_m, af_s, '#1f77b4', 'PACE-GGM', clip_log=True)
                    if not np.all(np.isnan(bf_m)):
                        plot_line(ax1, rhos, bf_m, bf_s, '#2ca02c', 'Independent', linestyle='--', marker='^', clip_log=True)
                    plot_line(ax1, rhos, sf_m, sf_s, '#d62728', 'SSP', marker='s', clip_log=True)

                    if has_compare and cmp_data is not None:
                        (rhos2, af2_m, af2_s, _, _, _, _, _, _, _, _, _, _) = load_rho_data(cmp_data, rho_values)
                        if len(rhos2) > 0:
                            plot_line(ax1, rhos2, af2_m, af2_s, '#1f77b4',
                                      f'PACE-GGM ({args.compare_label})', linestyle='--', clip_log=True)

                    if has_privcov:
                        pc_files = list(privcov_dir.glob(f'privcovariance_{dataset_name}_*.json'))
                        if pc_files:
                            with open(pc_files[0]) as f:
                                pc_data = json.load(f)
                            for method_key, color, label_suffix, ls in [
                                ('adaptive_cov', '#ff7f0e', 'AdaptiveCov', '-.'),
                                ('coinpress',    '#17becf', 'CoinPress',   ':'),
                            ]:
                                rpc, _, _, fm, fs = load_privcovariance_data(pc_data, rho_values, method_key)
                                if len(rpc) > 0:
                                    plot_line(ax1, rpc, fm, fs, color, label_suffix,
                                              linestyle=ls, marker='D', clip_log=True)

                    ax1.set_xscale('log'); ax1.set_yscale('log')
                    ax1.set_box_aspect(1)
                    ax1.tick_params(labelsize=8)
                    if not show_mse:
                        ax1.set_xlabel('ρ', fontsize=9, fontweight='bold')
                    if col_idx == 0:
                        ax1.set_ylabel('Frobenius Error', fontsize=9, fontweight='bold')
                    if clip_ymax is not None and np.any(np.isfinite(sf_m)):
                        ax1.set_ylim(top=np.nanmax(sf_m) * clip_ymax)
                    ax1.grid(False)

                # ── MSE row ───────────────────────────────────────────────────
                if show_mse:
                    mse_row = 0 if mse_only else 2
                    ax2 = fig.add_subplot(gs[mse_row, col_idx])

                    rhos_m, pm, ps, sm_v, ss, dm, ds_v, am, as_v = load_mse_data(data, rho_values)
                    if len(rhos_m) > 0:
                        plot_line(ax2, rhos_m, pm,   ps,   '#1f77b4', 'PACE-GGM', marker='o')
                        plot_line(ax2, rhos_m, sm_v, ss,   '#d62728', 'SSP',      marker='s')
                        if not np.all(np.isnan(am)):
                            plot_line(ax2, rhos_m, am, as_v, '#9467bd', 'AdaSSP',   marker='D', linestyle=':')

                        if has_compare and cmp_data is not None:
                            rhos_m2, pm2, ps2, _, _, _, _, _, _ = load_mse_data(cmp_data, rho_values)
                            if len(rhos_m2) > 0:
                                plot_line(ax2, rhos_m2, pm2, ps2, '#1f77b4',
                                          f'PACE-GGM ({args.compare_label})', linestyle='--', marker='*')

                        ax2.set_xscale('log')
                        if np.nanmin(pm) > 0:
                            ax2.set_yscale('log')
                        finite_vals = [v for v in list(pm) + list(am) if np.isfinite(v) and v > 0]
                        if finite_vals:
                            ax2.set_ylim(bottom=min(finite_vals) * 0.5,
                                         top=min(np.nanmax(sm_v) * 1.05 if not np.all(np.isnan(sm_v)) else np.inf,
                                                 np.nanmax(finite_vals) * 10))
                        if mse_only and plot_idx == 0:
                            ax2.legend(loc='best', fontsize=7, framealpha=0.9,
                                       handlelength=2, borderpad=0.5)
                        if mse_only:
                            ax2.set_title(f"{dataset_name}\n(d={d_val})", fontsize=8, fontweight='bold')
                    else:
                        ax2.text(0.5, 0.5, 'No MSE', ha='center', va='center', fontsize=8)

                    ax2.tick_params(labelsize=8)
                    ax2.set_xlabel('ρ', fontsize=9, fontweight='bold')
                    if col_idx == 0:
                        ax2.set_ylabel('Regression MSE\n(mean±SE)', fontsize=9, fontweight='bold')
                    ax2.grid(False)
                    if mse_only:
                        ax2.set_box_aspect(1)

            except Exception as e:
                print(f'WARNING: Error processing {dataset_name}: {e}')
                for row in range(n_rows):
                    ax = fig.add_subplot(gs[row, col_idx])
                    ax.text(0.5, 0.5, 'Data Error', ha='center', va='center', fontsize=8)
                    if row == 0:
                        ax.set_title(f"{dataset_name}\n(d={d_val})", fontsize=8, fontweight='bold')

        plt.tight_layout()
        out_pdf = output_dir / f'{tag}_group{group_idx+1}.pdf'
        out_png = output_dir / f'{tag}_group{group_idx+1}.png'
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f'Group {group_idx+1}: {out_png.name}')
        plt.close()

    # ── Ablation: ipm_nocc vs pgd_nocc ──────────────────────────────────────────
    if args.ablation_ipm_vs_pgd:
        ipm_dir = results_dir         # results_folder = ablation_ipm_vs_pgd/ipm_nocc
        pgd_dir = compare_dir         # --compare-folder = ablation_ipm_vs_pgd/pgd
        if pgd_dir is None:
            print("ERROR: --ablation-ipm-vs-pgd requires --compare-folder pointing to pgd dir")
        else:
            abl_datasets = args.ablation_datasets or [d[0] for d in datasets]
            label_a, label_b = args.ablation_labels
            ipm_color = '#1f77b4'   # blue  (IPM no CC)
            pgd_color = '#d62728'   # red   (IPM CC / PGD)

            n_cols = len(abl_datasets)
            fig, axes = plt.subplots(3, n_cols,
                                     figsize=(n_cols * 2.8, 3 * 2.8),
                                     squeeze=False)
            fig.subplots_adjust(hspace=0.45, wspace=0.4)

            for col_idx, ds_name in enumerate(abl_datasets):
                ipm_files = list(ipm_dir.glob(f'annealing_privacy_analysis_{ds_name}_*.json'))
                pgd_files = list(pgd_dir.glob(f'annealing_privacy_analysis_{ds_name}_*.json'))
                if not ipm_files or not pgd_files:
                    print(f"WARNING: missing files for {ds_name}")
                    continue

                with open(ipm_files[0]) as f:
                    ipm_data = json.load(f)
                with open(pgd_files[0]) as f:
                    pgd_data = json.load(f)

                d_val = ipm_data['parameters'].get('d', '?')

                rhos_i, mah_i_m, mah_i_s, frob_i_m, frob_i_s, rt_i_m, rt_i_s, rtp_i_m, rtp_i_s = load_ablation_data(ipm_data, rho_values)
                rhos_p, mah_p_m, mah_p_s, frob_p_m, frob_p_s, rt_p_m, rt_p_s, rtp_p_m, rtp_p_s = load_ablation_data(pgd_data, rho_values)

                # Row 0: Mahalanobis error
                ax0 = axes[0, col_idx]
                if len(rhos_i) > 0:
                    plot_line(ax0, rhos_i, mah_i_m, mah_i_s, ipm_color, label_a,
                              clip_log=True)
                if len(rhos_p) > 0:
                    plot_line(ax0, rhos_p, mah_p_m, mah_p_s, pgd_color, label_b,
                              linestyle='--', marker='s', clip_log=True)
                ax0.set_xscale('log'); ax0.set_yscale('log')
                ax0.set_box_aspect(1)
                ax0.tick_params(labelsize=8)
                ax0.set_title(f"{ds_name}\n(d={d_val})", fontsize=8, fontweight='bold')
                if col_idx == 0:
                    ax0.set_ylabel('Mahalanobis Error', fontsize=9, fontweight='bold')
                    ax0.legend(loc='best', fontsize=6, framealpha=0.9)
                ax0.grid(False)

                # Row 1: Frobenius error
                ax1 = axes[1, col_idx]
                if len(rhos_i) > 0:
                    plot_line(ax1, rhos_i, frob_i_m, frob_i_s, ipm_color, label_a,
                              clip_log=True)
                if len(rhos_p) > 0:
                    plot_line(ax1, rhos_p, frob_p_m, frob_p_s, pgd_color, label_b,
                              linestyle='--', marker='s', clip_log=True)
                ax1.set_xscale('log'); ax1.set_yscale('log')
                ax1.set_box_aspect(1)
                ax1.tick_params(labelsize=8)
                if col_idx == 0:
                    ax1.set_ylabel('Frobenius Error', fontsize=9, fontweight='bold')
                ax1.grid(False)

                # Row 2: Total solver runtime
                ax2 = axes[2, col_idx]
                if len(rhos_i) > 0:
                    plot_line(ax2, rhos_i, rt_i_m, rt_i_s, ipm_color, label_a)
                if len(rhos_p) > 0:
                    plot_line(ax2, rhos_p, rt_p_m, rt_p_s, pgd_color, label_b,
                              linestyle='--', marker='s')
                ax2.set_xscale('log')
                ax2.set_box_aspect(1)
                ax2.tick_params(labelsize=8)
                ax2.set_xlabel('ρ', fontsize=9, fontweight='bold')
                if col_idx == 0:
                    ax2.set_ylabel('Total Solver Time (s)', fontsize=9, fontweight='bold')
                ax2.grid(False)

            abl_tag = args.output_tag or 'ablation_ipm_vs_pgd'
            out_pdf = output_dir / f'{abl_tag}.pdf'
            out_png = output_dir / f'{abl_tag}.png'
            plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            print(f"Ablation plot saved: {out_png}")
            plt.close()

    # ── Ablation: fill_unobserved (pure PGD vs IPM) ──────────────────────────────
    if args.ablation_fill:
        parent = Path(args.results_folder)
        fill_configs = [
            ('ipm_nocc',     'IPM (no CC)',  '#000000', '-',  '*'),
            ('pgd_nocc_fu0', 'PGD fill=0',  '#1f77b4', '--', 'o'),
            ('pgd_nocc_fu1', 'PGD fill=1',  '#d62728', '--', 's'),
        ]
        abl_datasets = args.ablation_datasets or ['adult', 'LifeExpectancy', 'bank32nh', 'spambase', 'communities_crime']

        n_cols = len(abl_datasets)
        fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2.8, 3 * 2.8), squeeze=False)
        fig.subplots_adjust(hspace=0.45, wspace=0.4)

        for col_idx, ds_name in enumerate(abl_datasets):
            d_val = '?'
            for fu, label, color, ls, marker in fill_configs:
                folder = parent / fu
                files = list(folder.glob(f'annealing_privacy_analysis_{ds_name}_*.json'))
                if not files:
                    continue
                with open(files[0]) as f:
                    data = json.load(f)
                d_val = data['parameters'].get('d', '?')
                rhos_i, mah_m, mah_s, frob_m, frob_s, rt_m, rt_s, rtp_m, rtp_s = \
                    load_ablation_data(data, rho_values)
                if len(rhos_i) == 0:
                    continue
                for ax_row, (ym, ys) in enumerate([(mah_m, mah_s), (frob_m, frob_s), (rt_m, rt_s)]):
                    ax = axes[ax_row, col_idx]
                    ax.plot(rhos_i, ym, marker=marker, label=label, linewidth=1.5,
                            markersize=5, color=color, linestyle=ls)
                    lo = ym - ys
                    if ax_row < 2:  # log-scale rows: clip lower bound to avoid artifacts
                        lo = np.maximum(lo, ym * 1e-3)
                    ax.fill_between(rhos_i, lo, ym + ys, alpha=0.15, color=color)

            row_labels = ['Mahalanobis Error', 'Frobenius Error', 'Total Solver Time (s)']
            for ax_row in range(3):
                ax = axes[ax_row, col_idx]
                ax.set_xscale('log')
                if ax_row < 2:
                    ax.set_yscale('log')
                ax.set_box_aspect(1)
                ax.tick_params(labelsize=8)
                ax.grid(False)
                if ax_row == 0:
                    ax.set_title(f"{ds_name}\n(d={d_val})", fontsize=9, fontweight='bold')
                if ax_row == 2:
                    ax.set_xlabel('ρ', fontsize=9, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(row_labels[ax_row], fontsize=9, fontweight='bold')
                    if ax_row == 0:
                        ax.legend(loc='best', fontsize=6, framealpha=0.9)

        abl_tag = args.output_tag or 'ablation_fill'
        out_pdf = output_dir / f'{abl_tag}.pdf'
        out_png = output_dir / f'{abl_tag}.png'
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"Fill ablation plot saved: {out_png}")
        plt.close()

    print(f"\nAll plots saved to: {output_dir}")
