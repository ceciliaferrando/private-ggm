"""Mean trial time +/- SE per dataset and rho, as a LaTeX table.

A solver column is emitted only when the rows actually disagree on the solver;
once every dataset is on the dense ipm_chol it carries no information, so it is
dropped. --solver-column forces it on.

    python -m experiments.make_runtime_table results/cr_floor_12datasets \
        --output results/0_cr_final/main_runtime_table.tex
"""
import argparse
import json
import os
from glob import glob

import numpy as np

RHOS = ['0.0001', '0.01', '1.0', '10.0']
RHO_HEADERS = [r'$\rho=10^{-4}$', r'$\rho=10^{-2}$', r'$\rho=1$', r'$\rho=10$']
SOLVER_TEX = {'ipm_chol': r'\texttt{ipm\_chol}',
              'ipm_chol_sparse': r'\texttt{ipm\_chol\_sparse}'}


def fmt(mean, se):
    """Match the old table's precision: 2 dp below 10, 1 dp above."""
    if mean >= 100:
        return rf'${mean:.0f} \pm {se:.0f}$'
    if mean >= 10:
        return rf'${mean:.1f} \pm {se:.1f}$'
    return rf'${mean:.2f} \pm {se:.2f}$'


def collect(folders):
    """Collect rows from one or more folders; later folders override earlier ones.

    This lets a rerun of a few datasets (e.g. the two largest on the dense
    solver) be layered over the main results folder without copying files.
    """
    if isinstance(folders, str):
        folders = [folders]
    by_dataset = {}
    for folder in folders:
        for ds, row in _collect_one(folder).items():
            by_dataset[ds] = row
    return sorted(by_dataset.values(), key=lambda r: r[1])


def _collect_one(folder):
    rows = {}
    for f in glob(os.path.join(folder, 'annealing_privacy_analysis_*.json')):
        data = json.load(open(f))
        p, res = data['parameters'], data['results']
        cells, n_trials = [], None
        for rho in RHOS:
            v = [t['annealing'].get('trial_time_seconds') for t in res.get(rho, [])
                 if t.get('annealing', {}).get('trial_time_seconds') is not None]
            if not v:
                cells.append(None)
                continue
            n_trials = len(v)
            cells.append((np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))))
        rows[p['dataset']] = (p['dataset'], p['d'], p.get('optimizer', '?'),
                              cells, n_trials)
    return rows


def _solver_note(rows):
    """Caption clause flagging mixed solvers, omitted when every row is dense."""
    solvers = {opt for _, _, opt, _, _ in rows}
    if len(solvers) <= 1:
        return ''
    return (' Datasets marked with the sparse solver are not directly '
            'comparable with the dense rows.')


def render(rows, label, n_trials, solver_column=None):
    # Default: show the column only if it would distinguish anything.
    if solver_column is None:
        solver_column = len({opt for _, _, opt, _, _ in rows}) > 1
    body = []
    for ds, d, opt, cells, _ in rows:
        name = r'\texttt{' + ds.replace('_', r'\_') + '}'
        vals = [fmt(*c) if c else '--' for c in cells]
        sol = f'{SOLVER_TEX.get(opt, opt):<28} & ' if solver_column else ''
        body.append(f'{name:<34} & {d:<4}& ' + sol
                    + ' & '.join(f'{v:<18}' for v in vals) + r' \\')
    return '\n'.join([
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        rf'\caption{{Mean total trial time (seconds) $\pm$ SE across {n_trials} '
        rf'trials for selected $\rho$ values.{_solver_note(rows)}}}',
        rf'\label{{{label}}}',
        r'\renewcommand{\arraystretch}{1.2}',
        r'\begin{tabular}{@{}lr' + ('l' if solver_column else '')
        + 'c' * len(RHOS) + r'@{}}',
        r'\toprule',
        r'Dataset & $d$ & ' + ('Solver & ' if solver_column else '')
        + ' & '.join(RHO_HEADERS) + r' \\',
        r'\midrule',
        *body,
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_folder', nargs='+',
                    help='One or more folders; later ones override earlier by dataset')
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--label', default='tab:trial-times')
    ap.add_argument('--output', default=None)
    ap.add_argument('--solver-column', dest='solver_column',
                    action='store_true', default=None,
                    help='Force the Solver column on (default: only when the '
                         'rows use more than one solver)')
    ap.add_argument('--no-solver-column', dest='solver_column',
                    action='store_false',
                    help='Force the Solver column off')
    args = ap.parse_args()

    rows = collect(args.results_folder)
    if args.datasets:
        rows = [r for r in rows if r[0] in set(args.datasets)]
    if not rows:
        raise SystemExit(f'No results in {args.results_folder}')

    incomplete = [r[0] for r in rows if any(c is None for c in r[3])]
    if incomplete:
        print(f'  warning: missing timings for {incomplete}')
    n_trials = max((r[4] for r in rows if r[4]), default=0)

    tex = render(rows, args.label, n_trials, args.solver_column)
    print(tex)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as fh:
            fh.write(tex + '\n')
        print(f'\nwritten to {args.output}')


if __name__ == '__main__':
    main()
