"""run_benchmark.py — run solvers on all PROBLEMS and save results to benchmark_results.pkl.

Usage
-----
    cd experiments
    python run_benchmark.py                          # run all solvers
    python run_benchmark.py --solvers pgd ipm_chol  # run a subset

Output
------
    experiments/benchmark_results.pkl

    Dict with a top-level 'solvers_run' key listing the solvers that were run,
    plus per-problem entries keyed by (d, noise_level):
        {
            'solvers_run': ['pgd', 'ipm_chol', ...],
            (d, noise_level): {
                'problem': prob dict from make_problem(),
                'solvers': {
                    'pgd':      {'Sigma_hat': ..., 'hist': ...},
                    'ipm_chol': {'Sigma_hat': ..., 'hist': ...},
                    ...
                }
            }
        }
    A solver entry is None if it raised an exception.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

# Add solvers/ dir (for problems.py) and project root (for privateggm/)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from problems import PROBLEMS, make_problem
from privateggm.solve import solve

ALL_SOLVERS = [
    'pgd',
    'pgd_maxent',
    'ipm_chol',
    'ipm_chol_sparse',
    'clarabel',
    'dr_splitting',
]

def run_all(solver_names, problems=PROBLEMS):
    all_results = {}

    for spec in problems:
        d, noise_level = spec['d'], spec['noise_level']
        key = (d, noise_level)
        prob = make_problem(**spec)
        print(f"\nd={d:>2}  noise={noise_level:<4}  |obs|={len(prob['rows'])}")

        solver_results = {}
        for name in solver_names:
            print(f"  {name:<20}", end='', flush=True)
            t0 = time.perf_counter()
            try:
                S_hat, hist = solve(prob['rows'], prob['cols'], prob['vals'],
                                    prob['tau'], prob['d'], solver=name)
                elapsed = time.perf_counter() - t0
                last = hist[-1]
                print(f"  {elapsed:6.2f}s  {len(hist):4d} iters  "
                      f"obj={last['obj']:.3e}  ent={last['entropy']:.2f}")
                solver_results[name] = {'Sigma_hat': S_hat, 'hist': hist}
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  {elapsed:6.2f}s  FAILED: {e}")
                solver_results[name] = None

        all_results[key] = {'problem': prob, 'solvers': solver_results}

    all_results['solvers_run'] = solver_names
    return all_results


if __name__ == '__main__':
    valid_names = ALL_SOLVERS
    valid_d     = sorted({s['d'] for s in PROBLEMS})

    parser = argparse.ArgumentParser(description='Run benchmark solvers.')
    parser.add_argument(
        '--solvers', nargs='+', default=valid_names,
        metavar='SOLVER',
        help=f'Solvers to run (default: all). Choices: {valid_names}',
    )
    parser.add_argument(
        '--d', nargs='+', type=int, default=None,
        metavar='D',
        help=f'Restrict to these d values (default: all). Choices: {valid_d}',
    )
    parser.add_argument(
        '--output', default='benchmark_results.pkl',
        metavar='FILE',
        help='Output pickle filename (default: benchmark_results.pkl)',
    )
    args = parser.parse_args()

    unknown = [s for s in args.solvers if s not in valid_names]
    if unknown:
        parser.error(f"Unknown solver(s): {unknown}. Valid choices: {valid_names}")

    solver_names = [name for name in ALL_SOLVERS if name in args.solvers]

    problems = PROBLEMS
    if args.d is not None:
        unknown_d = [d for d in args.d if d not in valid_d]
        if unknown_d:
            parser.error(f"Unknown d value(s): {unknown_d}. Valid choices: {valid_d}")
        problems = [s for s in PROBLEMS if s['d'] in args.d]

    out_path = os.path.join(_HERE, args.output)
    results = run_all(solver_names, problems)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved → {out_path}")
