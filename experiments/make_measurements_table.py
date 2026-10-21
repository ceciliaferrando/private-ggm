"""Generate the measurement-count LaTeX table (Table 1) from a results folder.

Reads the <dataset>_measurements.out files written by test_iterative_annealing
and emits rows of the form $d + X$, where d is the number of diagonal entries
measured and X the mean number of distinct off-diagonal entries per trial.

    python -m experiments.make_measurements_table \
        results/cr_floor_12datasets \
        --highlight adult thyroid_ann ibm bank32nh communities_crime \
        --output results/0_cr_final/main_measurements_table.tex
"""
import argparse
import json
import os
from glob import glob

import numpy as np

RHOS = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]
RHO_HEADERS = [r'$\rho=10^{-4}$', r'$\rho=10^{-3}$', r'$\rho=10^{-2}$',
               r'$\rho=10^{-1}$', r'$\rho=1$', r'$\rho=2$', r'$\rho=10$']


def latex_escape(name):
    return name.replace('_', r'\_')


def collect(results_dir):
    rows = []
    for path in sorted(glob(os.path.join(results_dir, '*_measurements.out'))):
        dataset = os.path.basename(path).replace('_measurements.out', '')
        with open(path) as fh:
            records = json.load(fh)
        if not records:
            continue
        d = records[0]['d']

        # n_diagonal should equal d: every diagonal entry is measured in round 0.
        diag = {r['n_diagonal'] for r in records}
        if diag != {d}:
            print(f"  warning: {dataset} has n_diagonal={diag}, expected {{{d}}}")

        entries, events = [], []
        for rho in RHOS:
            recs = [r for r in records if r['rho'] == rho]
            if not recs:
                entries.append(None)
                events.append(None)
                continue
            entries.append(np.mean([r['n_offdiagonal'] for r in recs]))
            # Measurement events: one Gaussian call per (entry, round) pair. Diverges
            # from the entry count because each off-diagonal round also re-measures
            # both of its diagonals for free, and entries may be remeasured.
            ev_d, ev_o = [], []
            for r in recs:
                e2r = r['entry_to_rounds']
                ev_d.append(sum(len(v) for k, v in e2r.items() if _is_diag(k)))
                ev_o.append(sum(len(v) for k, v in e2r.items() if not _is_diag(k)))
            events.append((np.mean(ev_d), np.mean(ev_o)))
        rows.append((dataset, d, entries, events, len(records)))
    return sorted(rows, key=lambda r: r[1])


def _is_diag(key):
    i, j = key.strip('()').split(',')
    return int(i) == int(j)


def render(rows, highlight, style):
    both = style == 'both'
    # columns: dataset, d, [label], d(d+1)/2, one per rho
    colspec = ('lll' if both else 'll') + 'r' * (1 + len(RHOS))
    head = 'Dataset & $d$ & ' + ('& ' if both else '') + \
           '$\\frac{d(d+1)}{2}$ & ' + ' & '.join(RHO_HEADERS) + r' \\'
    out = [
        r'\begin{table}[t]',
        r'\centering',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{' + colspec + '}',
        r'\toprule',
        head,
        r'\midrule',
    ]
    for idx, (dataset, d, entries, events, _) in enumerate(rows):
        name = latex_escape(dataset)
        tri = d * (d + 1) // 2
        ecells = [f'${d}+{m:.1f}$' if m is not None else '--' for m in entries]
        if not both:
            lines = [f'{name:<21} & {d:<3} & {tri:<6} & '
                     + ' & '.join(f'{c:<11}' for c in ecells) + r' \\']
        else:
            mcells = [f'${a:.1f}+{b:.1f}$' if v is not None else '--'
                      for v, (a, b) in ((v, v or (0, 0)) for v in events)]
            lines = [
                (f'\\multirow{{2}}{{*}}{{{name}}} & \\multirow{{2}}{{*}}{{{d}}} & entries'
                 f' & \\multirow{{2}}{{*}}{{{tri}}} & '
                 + ' & '.join(f'{c:<13}' for c in ecells) + r' \\'),
                ('  & & measurements & & '
                 + ' & '.join(f'{c:<13}' for c in mcells) + r' \\'),
            ]
            if idx != len(rows) - 1:
                lines.append(r'\addlinespace[2pt]')
        if highlight and dataset not in highlight:
            lines = ['% ' + ln for ln in lines]
        out += lines
    cap = (r'\caption{Per trial, averaged over trials: the number of distinct '
           r'covariance entries measured, and the number of measurements taken '
           r'(diagonal$+$off-diagonal in both cases). The two differ because '
           r'each off-diagonal measurement also re-measures the two '
           r'corresponding diagonal entries at no additional privacy cost, and '
           r'because entries may be remeasured across rounds. The column '
           r'$d(d+1)/2$ is the total number of unique entries in the covariance '
           r'matrix, i.e.\ the ceiling on the first row.}'
           if both else
           r'\caption{Mean number of distinct entries measured per trial '
           r'(diagonal$+$off-diagonal). The column $d(d+1)/2$ gives the total '
           r'number of unique entries in the covariance matrix.}')
    out += [
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        cap,
        r'\label{tab:allmeasurements}',
        r'\end{table}',
    ]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_folder')
    ap.add_argument('--highlight', nargs='*', default=None,
                    help='Datasets to leave uncommented; the rest are commented out')
    ap.add_argument('--output', default=None, help='Write LaTeX here as well as stdout')
    ap.add_argument('--style', choices=['entries', 'both'], default='both',
                    help="'entries' = one row per dataset; 'both' = two rows, "
                         "distinct entries and measurements (needs \\usepackage{multirow})")
    args = ap.parse_args()

    rows = collect(args.results_folder)
    if not rows:
        raise SystemExit(f'No *_measurements.out files in {args.results_folder}')

    trials = {n for *_, n in rows}
    print(f'{len(rows)} datasets, {trials} records each (trials x rho)\n')

    tex = render(rows, set(args.highlight) if args.highlight else None, args.style)
    print(tex)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as fh:
            fh.write(tex + '\n')
        print(f'\nwritten to {args.output}')


if __name__ == '__main__':
    main()
