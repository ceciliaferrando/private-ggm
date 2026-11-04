"""Relative-error table for the alpha/beta grid.

For each (dataset, rho) cell, divides every configuration's mean error by the
best configuration in that cell, then reports the geometric mean of those
ratios across all cells. 1.000 means "always the best configuration".

Unlike an argmin-per-cell table this reports magnitude rather than a vote, so
it survives the per-cell noise that makes argmins flip at small trial counts.

    python -m experiments.make_grid_relerr_table results/ab_grid_lowalpha \
        --output results/0_cr_final/grid_relerr.tex
"""
import argparse
import json
import os
from glob import glob

import numpy as np

RHOS = ['0.0001', '0.001', '0.01', '0.1', '1.0', '2.0', '10.0']
METRICS = [('mahalanobis_error', 'Mahalanobis', 'mah'),
           ('frobenius_error', 'Frobenius', 'frob')]


def _ab(cell):
    return '0.' + cell[1:3].lstrip('0'), '0.' + cell[5:7].lstrip('0')


def collect(grid_dir):
    out = {}
    for cell in sorted(os.listdir(grid_dir)):
        leaf = os.path.join(grid_dir, cell, 'ipm_solver')
        if not os.path.isdir(leaf):
            leaf = os.path.join(grid_dir, cell)
        if not os.path.isdir(leaf):
            continue
        try:
            ab = _ab(cell)
        except Exception:
            continue
        for f in glob(os.path.join(leaf, 'annealing_privacy_analysis_*.json')):
            data = json.load(open(f))
            ds = data['parameters']['dataset']
            for rho in RHOS:
                trials = data['results'].get(rho, [])
                for key, _, tag in METRICS:
                    v = [t['annealing'][key] for t in trials
                         if key in t.get('annealing', {})]
                    if v:
                        out.setdefault((tag, ds, rho), {})[ab] = np.array(v, float)
    # keep only cells where every configuration is present with equal trials
    full = {}
    for k, v in out.items():
        if len({len(x) for x in v.values()}) == 1:
            full[k] = v
    n_cfg = max((len(v) for v in full.values()), default=0)
    return {k: v for k, v in full.items() if len(v) == n_cfg}


def geo(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('grid_folder')
    ap.add_argument('--label', default='tab:grid_relerr')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    data = collect(args.grid_folder)
    if not data:
        raise SystemExit(f'No complete grid cells under {args.grid_folder}')
    alphas = sorted({a for v in data.values() for a, _ in v})
    betas = sorted({b for v in data.values() for _, b in v})
    n_tr = max(len(x) for v in data.values() for x in v.values())

    rel = {}   # (tag, alpha, beta) -> list of ratios
    n_cells = {}
    for (tag, ds, rho), cfgs in data.items():
        m = {ab: v.mean() for ab, v in cfgs.items()}
        best = min(m.values())
        n_cells[tag] = n_cells.get(tag, 0) + 1
        for ab, val in m.items():
            rel.setdefault((tag, *ab), []).append(val / best)

    L = [r'\begin{table}[ht]', r'\centering\small',
         r'\begin{tabular}{l' + 'c' * len(betas) + 'c' + 'c' * len(betas) + '}',
         r'\toprule',
         (' & \\multicolumn{%d}{c}{Mahalanobis} & & \\multicolumn{%d}{c}{Frobenius} \\\\'
          % (len(betas), len(betas))),
         (f'\\cmidrule(lr){{2-{1+len(betas)}}} '
          f'\\cmidrule(lr){{{3+len(betas)}-{2+2*len(betas)}}}'),
         (' & ' + ' & '.join(rf'$\beta={b}$' for b in betas) + ' & & '
          + ' & '.join(rf'$\beta={b}$' for b in betas) + r' \\'),
         r'\midrule']
    for a in alphas:
        cells = []
        for tag in ['mah', 'frob']:
            cells.append(' & '.join(f'{geo(rel[(tag, a, b)]):.3f}' for b in betas))
        L.append(rf'$\alpha={a}$ & ' + cells[0] + ' & & ' + cells[1] + r' \\')
    L.append(r'\midrule')
    # beta marginals (pooled over alpha)
    mb = []
    for tag in ['mah', 'frob']:
        mb.append(' & '.join(
            f'{geo(np.concatenate([rel[(tag, a, b)] for a in alphas])):.3f}'
            for b in betas))
    L.append(r'marginal & ' + mb[0] + ' & & ' + mb[1] + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']

    # alpha marginals go in the caption, they do not fit the column layout
    am = {}
    for tag in ['mah', 'frob']:
        am[tag] = {a: geo(np.concatenate([rel[(tag, a, b)] for b in betas]))
                   for a in alphas}
    a_txt = '; '.join(
        f"{nm}: " + ', '.join(rf'$\alpha={a}$: {am[tag][a]:.3f}' for a in alphas)
        for tag, nm in [('mah', 'Mahalanobis'), ('frob', 'Frobenius')])

    L += [
        rf'\caption{{Mean error relative to the best configuration in each '
        rf'dataset--$\rho$ cell, as a geometric mean over '
        rf'{n_cells.get("mah", 0)} cells ({n_tr} trials each); $1.000$ denotes a '
        rf'configuration that is always optimal. The bottom row pools over '
        rf'$\alpha$. Pooling instead over $\beta$ gives --- {a_txt} --- so the '
        rf'spread across $\alpha$ is an order of magnitude smaller than the '
        rf'spread across $\beta$.}}',
        rf'\label{{{args.label}}}',
        r'\end{table}',
    ]
    tex = '\n'.join(L)
    print(tex)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as fh:
            fh.write(tex + '\n')
        print(f'\nwritten to {args.output}')


if __name__ == '__main__':
    main()
