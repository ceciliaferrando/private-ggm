"""Plot the sparse-vs-dense scalability stress test.

Reads the pickle written by benchmark_sparse_vs_dense.py and produces two
panels: wall-clock time against d (one line per average degree, dense and
sparse overlaid) and the dense/sparse speedup against d.

The benchmark's obj_dense / obj_sparse fields are NOT plotted: solve() with
decompose='blocks' reports hist from the largest connected component only, so
those objectives are not comparable across solvers on sparse graphs.
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_extra(paths):
    """Load additional pickles holding single records or lists of records."""
    out = []
    for path in paths:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        out.extend(obj if isinstance(obj, list) else [obj])
    return out


def merge(results, extra):
    """Fill in missing timings from extra records, keyed by (d, avg_degree, seed)."""
    by = {(r['d'], r['avg_degree'], r['seed']): r for r in results}
    for e in extra:
        key = (e['d'], e['avg_degree'], e['seed'])
        if key in by:
            for field in ('time_dense', 'time_sparse', 'obj_dense', 'obj_sparse'):
                if by[key].get(field) is None and e.get(field) is not None:
                    by[key][field] = e[field]
        else:
            results.append(e)
            by[key] = e
    return results


def plot_dense(results, out_dir, tag, cell_size=3.4):
    """Single-panel dense-solver runtime, with fitted scaling exponents."""
    degrees = sorted({r['avg_degree'] for r in results})
    cmap = plt.cm.viridis(np.linspace(0.08, 0.88, len(degrees)))

    fig, ax = plt.subplots(1, 1, figsize=(cell_size * 1.45, cell_size))

    for color, deg in zip(cmap, degrees):
        cells = sorted((r for r in results
                        if r['avg_degree'] == deg and r['time_dense'] is not None),
                       key=lambda r: r['d'])
        if not cells:
            continue
        ds = np.array([r['d'] for r in cells], dtype=float)
        ts = np.array([r['time_dense'] for r in cells], dtype=float)

        fit = ds >= 128
        label = f'{deg:g}'
        if fit.sum() >= 2:
            slope = np.polyfit(np.log(ds[fit]), np.log(ts[fit]), 1)[0]
            label = f'{deg:g}  ($p$={slope:.2f})'

        ax.plot(ds, ts, marker='o', linewidth=1.5, markersize=4,
                color=color, label=label)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('$d$', fontsize=9, fontweight='bold')
    ax.set_ylabel('Wall-clock time (s)', fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc='upper left', fontsize=6, framealpha=0.9,
              title='avg. degree', title_fontsize=6)
    ax.tick_params(labelsize=7)
    ax.set_box_aspect(1)

    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f'{tag}.pdf')
    out_png = os.path.join(out_dir, f'{tag}.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_pdf, out_png


def plot(results, out_dir, tag, cell_size=3.4):
    degrees = sorted({r['avg_degree'] for r in results})
    cmap = plt.cm.viridis(np.linspace(0.08, 0.88, len(degrees)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2 * cell_size * 1.25, cell_size))

    for color, deg in zip(cmap, degrees):
        cells = sorted((r for r in results if r['avg_degree'] == deg),
                       key=lambda r: r['d'])

        ds = [r['d'] for r in cells if r['time_dense'] is not None]
        td = [r['time_dense'] for r in cells if r['time_dense'] is not None]
        ss = [r['d'] for r in cells if r['time_sparse'] is not None]
        ts = [r['time_sparse'] for r in cells if r['time_sparse'] is not None]

        ax0.plot(ds, td, marker='o', linewidth=1.5, markersize=4,
                 color=color, label=f'degree {deg:g}')
        ax0.plot(ss, ts, marker='s', linewidth=1.5, markersize=4,
                 color=color, linestyle='--')

        both = [r for r in cells
                if r['time_dense'] is not None and r['time_sparse'] is not None]
        ax1.plot([r['d'] for r in both],
                 [r['time_dense'] / r['time_sparse'] for r in both],
                 marker='o', linewidth=1.5, markersize=4, color=color,
                 label=f'degree {deg:g}')

    ax0.set_xscale('log', base=2)
    ax0.set_yscale('log')
    ax0.set_xlabel('$d$', fontsize=9, fontweight='bold')
    ax0.set_ylabel('Wall-clock time (s)', fontsize=9, fontweight='bold')
    ax0.set_title('Solver runtime\n(solid: dense, dashed: sparse)',
                  fontsize=8, fontweight='bold')
    ax0.grid(alpha=0.3, linewidth=0.5)
    ax0.legend(loc='upper left', fontsize=6, framealpha=0.9, title='avg. degree',
               title_fontsize=6)

    ax1.axhline(1.0, color='0.4', linewidth=1.0, linestyle=':')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('$d$', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Speedup (dense / sparse)', fontsize=9, fontweight='bold')
    ax1.set_title('Sparse advantage\n(above 1: sparse is faster)',
                  fontsize=8, fontweight='bold')
    ax1.grid(alpha=0.3, linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=6, framealpha=0.9, title='avg. degree',
               title_fontsize=6)

    for ax in (ax0, ax1):
        ax.tick_params(labelsize=7)
        ax.set_box_aspect(1)

    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f'{tag}.pdf')
    out_png = os.path.join(out_dir, f'{tag}.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_pdf, out_png


def emit_table(results, path, dense_only=False):
    """Write the runtime grid as a LaTeX table (rows: d, columns: avg degree)."""
    degrees = sorted({r['avg_degree'] for r in results})
    dims = sorted({r['d'] for r in results})
    by = {(r['d'], r['avg_degree']): r for r in results}

    lines = [r'\begin{tabular}{r' + 'r' * len(degrees) + '}',
             r'\toprule',
             r'& \multicolumn{%d}{c}{average degree} \\' % len(degrees),
             r'\cmidrule(lr){2-%d}' % (len(degrees) + 1),
             '$d$ & ' + ' & '.join(f'{g:g}' for g in degrees) + r' \\',
             r'\midrule']
    solvers = [('ipm\\_chol', 'time_dense')]
    if not dense_only:
        solvers.append(('ipm\\_chol\\_sparse', 'time_sparse'))
    for solver, key in solvers:
        lines.append(r'\multicolumn{%d}{l}{\texttt{%s}} \\' % (len(degrees) + 1, solver))
        for d in dims:
            cells = []
            for g in degrees:
                v = by.get((d, g), {}).get(key)
                cells.append('--' if v is None else f'{v:.2f}')
            lines.append(f'{d} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the sparse-vs-dense stress test.')
    parser.add_argument('results', help='Pickle written by benchmark_sparse_vs_dense.py')
    parser.add_argument('--output-dir', default='results/0_cr_final')
    parser.add_argument('--output-tag', default='stress_test_runtime')
    parser.add_argument('--table', default=None, help='Also write a LaTeX runtime table here')
    parser.add_argument('--extra', nargs='*', default=[],
                        help='Additional pickles to merge (e.g. the dense d=2048 cells)')
    parser.add_argument('--dense-only', action='store_true',
                        help='Plot only ipm_chol, as a single panel')
    args = parser.parse_args()

    res = load(args.results)
    if args.extra:
        res = merge(res, load_extra(args.extra))

    if args.dense_only:
        pdf, png = plot_dense(res, args.output_dir, args.output_tag)
    else:
        pdf, png = plot(res, args.output_dir, args.output_tag)
    print(f'Wrote {pdf}\n      {png}')
    if args.table:
        print(f'      {emit_table(res, args.table, dense_only=args.dense_only)}')
