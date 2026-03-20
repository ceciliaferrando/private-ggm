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
                    'pgd':      {'Sigma_hat': ..., 'hist': ..., 'compile_time': None},
                    'ipm_chol': {'Sigma_hat': ..., 'hist': ..., 'compile_time': None},
                    ...
                }
            }
        }
    compile_time is a float (seconds) for JAX solvers whose first iter is stripped,
    None otherwise.  A solver entry is None if it raised an exception.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import PROBLEMS, make_problem
from solvers import ipm_chol, pgd, pgd_maxent, pgd_maxent_jax, sparse_prec_ipm, sparse_prec_nobarrier, clarabel_solver, strip_compile_iter

ALL_SOLVERS = [
    ('pgd',                   pgd),
    ('pgd_maxent',            pgd_maxent),
    ('pgd_maxent_jax',        pgd_maxent_jax),
    ('ipm_chol',              ipm_chol),
    ('sparse_prec_ipm',       sparse_prec_ipm),
    ('sparse_prec_nobarrier', sparse_prec_nobarrier),
    ('clarabel',              clarabel_solver),
]

# Solvers whose first iteration is dominated by JAX JIT compilation.
STRIP_JIT = {'sparse_prec_ipm', 'pgd_maxent_jax', 'sparse_prec_nobarrier'}
# NOTE: strip_compile_iter is correct for sparse_prec_ipm (first outer IPM iter ≈ compile).
# For pgd_maxent_jax it is WRONG: hist[0] is the first Stage A PGD step (pure Python/NumPy),
# not the JAX compile. Actual JIT compile happens at the Stage A→B transition, somewhere
# in the middle of the combined history. We strip anyway to be consistent, but the
# "compile_time" label in output is misleading for pgd_maxent_jax.
# For sparse_prec_nobarrier, hist entries are per record_every inner steps, so stripping
# hist[0] removes the compile-dominated first record_every steps.

# NOTE: pgd_maxent_jax is very slow at large d — ~686s for d=128/med, likely longer
# for d=128/high (hits the 2000-iter limit with O(d^2) per step). At d>=128, consider
# running with:  --solvers pgd pgd_maxent ipm_chol sparse_prec_ipm clarabel


def run_all(solvers, problems=PROBLEMS):
    all_results = {}

    for spec in problems:
        d, noise_level = spec['d'], spec['noise_level']
        key = (d, noise_level)
        prob = make_problem(**spec)
        print(f"\nd={d:>2}  noise={noise_level:<4}  |inds|={len(prob['inds'])}")

        solver_results = {}
        for name, solver in solvers:
            print(f"  {name:<20}", end='', flush=True)
            t0 = time.perf_counter()
            try:
                S_hat, hist = solver(prob['A'], prob['inds'], prob['tau'])
                compile_time = None
                if name in STRIP_JIT:
                    hist, compile_time = strip_compile_iter(hist)
                elapsed = time.perf_counter() - t0
                last = hist[-1]
                compile_str = f"  compile={compile_time:.2f}s" if compile_time is not None else ""
                print(f"  {elapsed:6.2f}s  {len(hist):4d} iters  "
                      f"obj={last['obj']:.3e}  ent={last['entropy']:.2f}{compile_str}")
                solver_results[name] = {'Sigma_hat': S_hat, 'hist': hist, 'compile_time': compile_time}
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  {elapsed:6.2f}s  FAILED: {e}")
                solver_results[name] = None

        all_results[key] = {'problem': prob, 'solvers': solver_results}

    all_results['solvers_run'] = [name for name, _ in solvers]
    return all_results


if __name__ == '__main__':
    valid_names = [name for name, _ in ALL_SOLVERS]
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

    unknown = [s for s in args.solvers if s not in dict(ALL_SOLVERS)]
    if unknown:
        parser.error(f"Unknown solver(s): {unknown}. Valid choices: {valid_names}")

    solvers = [(name, fn) for name, fn in ALL_SOLVERS if name in args.solvers]
    # Preserve the canonical order from ALL_SOLVERS regardless of CLI order.

    problems = PROBLEMS
    if args.d is not None:
        unknown_d = [d for d in args.d if d not in valid_d]
        if unknown_d:
            parser.error(f"Unknown d value(s): {unknown_d}. Valid choices: {valid_d}")
        problems = [s for s in PROBLEMS if s['d'] in args.d]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    results = run_all(solvers, problems)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved → {out_path}")
