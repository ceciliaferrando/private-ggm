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
import string
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


def render(rows, highlight, style, label='tab:allmeasurements',
           rhos=None, rho_headers=None, caption=None, resizebox=True,
           tabcolsep=None, sideways=False, fontsize=r'\small',
           merge_d=False, integers=False, full_width=False, keys=False,
           abbrev=False):
    both = style == 'both'
    RHOS = rhos if rhos is not None else globals()['RHOS']
    RHO_HEADERS = rho_headers if rho_headers is not None else globals()['RHO_HEADERS']
    # The dataset column is the widest in the table and nothing in it is
    # numeric, so replacing the names with (a), (b), ... and moving them to a
    # caption legend buys more width than any other single change. The header
    # goes blank too: "Dataset" alone would otherwise set the column width.
    keymap = [(f'({c})', ds) for (ds, *_), c in zip(rows, string.ascii_lowercase)] \
        if keys else []
    dshead = '' if keys else 'Dataset'
    # columns: dataset, d, [label], d(d+1)/2, one per rho
    # columns: dataset, d, d(d+1)/2, [entries/measurements label], one per rho
    if merge_d:
        colspec = 'l' + 'c' + ('l' if both else '') + 'r' * len(RHOS)
        head = dshead + ' & $d$ ($\\tfrac{d(d+1)}{2}$) & ' + ('& ' if both else '') + \
               ' & '.join(RHO_HEADERS) + r' \\'
    else:
        colspec = 'll' + 'r' + ('l' if both else '') + 'r' * len(RHOS)
        head = dshead + ' & $d$ & $\\frac{d(d+1)}{2}$ & ' + ('& ' if both else '') + \
               ' & '.join(RHO_HEADERS) + r' \\'
    # A landscape float gets ~9in of width against 5.5in in portrait, which is
    # what lets the full rho range set at \small with no \resizebox.
    env = 'sidewaystable' if sideways else 'table'
    out = [
        rf'\begin{{{env}}}' + ('' if sideways else '[t]'),
        r'\centering',
        _caption(both, caption, integers, merge_d, keymap, abbrev),
        rf'\label{{{label}}}',
    ]
    # Narrower column padding buys width. Without \resizebox that width goes
    # straight to the margins; with it, a narrower natural width means a scale
    # factor closer to 1, i.e. larger glyphs. Either way it is worth setting.
    if tabcolsep is not None:
        out.append(rf'\setlength{{\tabcolsep}}{{{tabcolsep}pt}}')
    if resizebox:
        out.append(r'\resizebox{\linewidth}{!}{%')
    else:
        out.append(fontsize)
    out += [
        # tabular* with \extracolsep{\fill} pushes the slack into the column
        # gaps, so companion tables come out exactly \linewidth wide and still
        # share one font size -- \resizebox would instead scale each table's
        # glyphs by a different factor.
        (r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}' + colspec + '@{}}'
         if full_width else r'\begin{tabular}{' + colspec + '}'),
        r'\toprule',
        head,
        r'\midrule',
    ]
    prec = 0 if integers else 1
    for idx, (dataset, d, entries, events, _) in enumerate(rows):
        name = keymap[idx][0] if keys else latex_escape(dataset)
        tri = d * (d + 1) // 2
        # One column reads "102 (5253)" instead of two columns of 102 and 5253.
        dcols = (f'{d} ({tri})',) if merge_d else (f'{d:<3}', f'{tri:<6}')
        ecells = [f'${d}+{m:.{prec}f}$' if m is not None else '--' for m in entries]
        if not both:
            lines = [f'{name:<21} & ' + ' & '.join(dcols) + ' & '
                     + ' & '.join(f'{c:<11}' for c in ecells) + r' \\']
        else:
            mcells = [f'${a:.{prec}f}+{b:.{prec}f}$' if v is not None else '--'
                      for v, (a, b) in ((v, v or (0, 0)) for v in events)]
            # One column reads "102 (5253)" instead of two separate columns.
            mr = (f'\\multirow{{2}}{{*}}{{{d} ({tri})}}' if merge_d else
                  f'\\multirow{{2}}{{*}}{{{d}}} & \\multirow{{2}}{{*}}{{{tri}}}')
            blank = '  & & ' if merge_d else '  & & & '
            lines = [
                (f'\\multirow{{2}}{{*}}{{{name}}} & ' + mr + ' & entries & '
                 + ' & '.join(f'{c:<13}' for c in ecells) + r' \\'),
                (blank + ('meas.        & ' if abbrev else 'measurements & ')
                 + ' & '.join(f'{c:<13}' for c in mcells) + r' \\'),
            ]
            if idx != len(rows) - 1:
                lines.append(r'\addlinespace[2pt]')
        if highlight and dataset not in highlight:
            lines = ['% ' + ln for ln in lines]
        out += lines
    out += [
        r'\bottomrule',
        (r'\end{tabular*}%' if full_width else r'\end{tabular}%'),
        r'}' if resizebox else '',
        rf'\end{{{env}}}',
    ]
    return '\n'.join(l for l in out if l)


def _caption(both, caption=None, integers=False, merge_d=False, keymap=(),
             abbrev=False):
    """Caption text. NeurIPS places table captions above the table.

    When counts are rounded, the caption has to say so: the reader must not
    read an integer as an exact count when it is a mean over trials.
    """
    legend = ''
    if keymap:
        # The legend carries the dataset names the first column no longer has,
        # in the table's own row order (ascending d).
        legend = (' Datasets, in row order: '
                  + ', '.join(f'{k}~\\texttt{{{latex_escape(ds)}}}' for k, ds in keymap)
                  + '.')
    if caption:
        return r'\caption{' + caption + legend + '}'
    rounded = (' and rounded to the nearest integer' if integers else '')
    return (r'\caption{Per trial, averaged over trials' + rounded + r': the number of distinct '
           r'covariance entries measured, and the number of measurements taken '
           r'(diagonal$+$off-diagonal in both cases)'
           + (r', in the rows labelled \texttt{entries} and \texttt{meas.}\ '
              r'respectively' if abbrev else '')
           + r'. The two differ because '
           r'each off-diagonal measurement also re-measures the two '
           r'corresponding diagonal entries at no additional privacy cost, and '
           r'because entries, diagonal ones included, may be remeasured across '
           r'rounds. '
           + (r'The second column gives $d$ and, in parentheses, $d(d+1)/2$, the '
              r'total number of unique entries in the covariance matrix and hence '
              r'the ceiling on the entries row.'
              if merge_d else
              r'The column $d(d+1)/2$ is the total number of unique entries in the '
              r'covariance matrix, i.e.\ the ceiling on the entries row.')
           + legend + '}'
           if both else
           r'\caption{Mean number of distinct entries measured per trial'
           + (r', rounded to the nearest integer' if integers else '') + r' '
           r'(diagonal$+$off-diagonal). The column $d(d+1)/2$ gives the total '
           r'number of unique entries in the covariance matrix.' + legend + '}')


def _tabular_only(tex):
    """Pull just the tabular out of a rendered table environment."""
    lines = tex.split('\n')
    i = next(k for k, l in enumerate(lines) if l.startswith(r'\begin{tabular'))
    j = next(k for k, l in enumerate(lines) if l.startswith(r'\end{tabular'))
    return '\n'.join(lines[i:j + 1])


def render_stacked(rows, highlight, style, label, rhos, headers, caption,
                   chunks, tabcolsep=None, fontsize=r'\small',
                   merge_d=False, integers=False):
    """One table environment holding several rho panels, stacked vertically.

    Keeps a single caption, label and table number -- the reader sees one table
    -- while each panel is narrow enough to set at \\small instead of being
    scaled down by \\resizebox.
    """
    out = [r'\begin{table}[t]', r'\centering',
           _caption(style == 'both', caption, integers, merge_d), rf'\label{{{label}}}']
    if tabcolsep is not None:
        out.append(rf'\setlength{{\tabcolsep}}{{{tabcolsep}pt}}')
    out.append(fontsize)

    start = 0
    for n, size in enumerate(chunks):
        idx = list(range(start, start + size))
        start += size
        sub = [(ds, d, [e[i] for i in idx], [v[i] for i in idx], cnt)
               for ds, d, e, v, cnt in rows]
        panel = render(sub, highlight, style, label,
                       [rhos[i] for i in idx], [headers[i] for i in idx],
                       caption, resizebox=False, merge_d=merge_d,
                       integers=integers)
        out.append(_tabular_only(panel))
        if n != len(chunks) - 1:
            out.append(r'\\[8pt]')

    out.append(r'\end{table}')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_folder')
    ap.add_argument('--highlight', nargs='*', default=None,
                    help='Datasets to leave uncommented; the rest are commented out')
    ap.add_argument('--datasets', nargs='*', default=None,
                    help='Restrict the table to these datasets (rows for others are omitted entirely)')
    ap.add_argument('--label', default='tab:allmeasurements',
                    help='LaTeX \\label for the table')
    ap.add_argument('--output', default=None, help='Write LaTeX here as well as stdout')
    ap.add_argument('--style', choices=['entries', 'both'], default='both',
                    help="'entries' = one row per dataset; 'both' = two rows, "
                         "distinct entries and measurements (needs \\usepackage{multirow})")
    ap.add_argument('--rhos', nargs='*', default=None,
                    help='Emit only these rho columns, e.g. 0.0001 0.001 0.01 0.1. '
                         'Splitting the rho range across two tables keeps every '
                         'dataset and every rho while letting each table set at '
                         'full size instead of being shrunk by \\resizebox')
    ap.add_argument('--no-resizebox', action='store_true', default=False,
                    help='Omit \\resizebox so the table sets at its natural size; '
                         'use once the column count is low enough to fit')
    ap.add_argument('--caption', default=None, help='Override the caption text')
    ap.add_argument('--stack', nargs='*', type=int, default=None,
                    help='Split the rho columns into stacked panels inside ONE '
                         'table, e.g. --stack 4 3. Keeps a single caption, label '
                         'and table number while letting each panel set at full '
                         'size instead of being shrunk by \\resizebox')
    ap.add_argument('--merge-d', action='store_true', default=False,
                    help='Fold d and d(d+1)/2 into one column reading "102 (5253)"')
    ap.add_argument('--integers', action='store_true', default=False,
                    help='Round counts to the nearest integer. A mean over 10 '
                         'trials carries no meaningful tenth, and it is worth '
                         'roughly 0.5in of width')
    ap.add_argument('--fontsize', default='small',
                    choices=['small', 'footnotesize', 'scriptsize'],
                    help='Table font when not using \\resizebox. NeurIPS sets no '
                         'numeric minimum for tables (only "legible"), so this is '
                         'a readability choice, not a compliance one')
    ap.add_argument('--sideways', action='store_true', default=False,
                    help='Emit a landscape sidewaystable (needs '
                         '\\usepackage{rotating}). ~9in of width, so the full rho '
                         'range fits at full size in one table')
    ap.add_argument('--abbrev', action='store_true', default=False,
                    help="Label the second row 'meas.' instead of "
                         "'measurements', and say so in the caption. Worth "
                         "~30pt of the label column")
    ap.add_argument('--keys', action='store_true', default=False,
                    help='Replace dataset names with (a), (b), ... and list '
                         'them in the caption. The name column is the widest '
                         'in the table, so this is the cheapest way to buy '
                         'width back')
    ap.add_argument('--full-width', action='store_true', default=False,
                    help='Set the table as tabular* spanning the full '
                         '\\linewidth, distributing the slack into the column '
                         'gaps. Companion tables then share an exact width and '
                         'an identical font size')
    ap.add_argument('--tabcolsep', type=float, default=None,
                    help='Column padding in pt (LaTeX default 6). Narrows the '
                         'table either way: without --resizebox it reclaims '
                         'margin, with it the smaller natural width means less '
                         'shrinkage and so a larger effective font')
    args = ap.parse_args()

    rows = collect(args.results_folder)
    if args.datasets:
        keep = set(args.datasets)
        missing = keep - {r[0] for r in rows}
        if missing:
            print(f'  warning: requested but not found: {sorted(missing)}')
        rows = [r for r in rows if r[0] in keep]
    if not rows:
        raise SystemExit(f'No *_measurements.out files in {args.results_folder}')

    trials = {n for *_, n in rows}
    print(f'{len(rows)} datasets, {trials} records each (trials x rho)\n')

    sel_rhos, sel_headers = RHOS, RHO_HEADERS
    if args.rhos:
        want = [float(r) for r in args.rhos]
        missing = [r for r in want if r not in RHOS]
        if missing:
            raise SystemExit(f'--rhos: no such rho column(s): {missing}; '
                             f'available: {RHOS}')
        idx = [RHOS.index(r) for r in want]
        sel_rhos = [RHOS[i] for i in idx]
        sel_headers = [RHO_HEADERS[i] for i in idx]
        rows = [(ds, d, [e[i] for i in idx], [v[i] for i in idx], n)
                for ds, d, e, v, n in rows]
        print(f'rho columns: {sel_rhos}')

    hl = set(args.highlight) if args.highlight else None
    if args.stack:
        if sum(args.stack) != len(sel_rhos):
            raise SystemExit(f'--stack sums to {sum(args.stack)} but there are '
                             f'{len(sel_rhos)} rho columns')
        tex = render_stacked(rows, hl, args.style, args.label, sel_rhos,
                             sel_headers, args.caption, args.stack, args.tabcolsep,
                             '\\' + args.fontsize, args.merge_d, args.integers)
    else:
        tex = render(rows, hl, args.style, args.label, sel_rhos, sel_headers,
                     args.caption, resizebox=not args.no_resizebox,
                     tabcolsep=args.tabcolsep, sideways=args.sideways,
                     fontsize='\\' + args.fontsize, merge_d=args.merge_d,
                     integers=args.integers, full_width=args.full_width,
                     keys=args.keys, abbrev=args.abbrev)
    print(tex)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as fh:
            fh.write(tex + '\n')
        print(f'\nwritten to {args.output}')


if __name__ == '__main__':
    main()
