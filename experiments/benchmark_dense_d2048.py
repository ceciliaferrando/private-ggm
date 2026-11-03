"""Extend the stress test with the dense solver at d=2048.

benchmark_sparse_vs_dense.py skips ipm_chol above d=1024, so the dense curve
stops one point short of the grid. This runs those five cells only and writes
them in the same record format, to be merged at plot time.
"""

import argparse
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from benchmark_sparse_vs_dense import AVG_DEGREES, make_problem_by_degree, time_solver


def run(output_path, d=2048, seeds=1):
    results = []
    for avg_degree in AVG_DEGREES:
        for seed in range(seeds):
            prob = make_problem_by_degree(d, avg_degree, seed=seed)
            print(f"d={d}  avg_deg={avg_degree:.1f}  |off|={prob['n_off']}", flush=True)
            print("    ipm_chol       ", end='', flush=True)
            t_dense, obj_dense = time_solver('ipm_chol', prob)
            print(f"  {t_dense:.2f}s" if t_dense is not None else "  FAIL")

            results.append({
                'd':           d,
                'avg_degree':  avg_degree,
                'seed':        seed,
                'n_edges':     prob['n_off'],
                'time_dense':  t_dense,
                'time_sparse': None,
                'obj_dense':   obj_dense,
                'obj_sparse':  None,
            })
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense solver at d=2048 only.')
    parser.add_argument('--output', default='dense_d2048_results.pkl')
    parser.add_argument('--d', type=int, default=2048)
    parser.add_argument('--seeds', type=int, default=1)
    args = parser.parse_args()

    out = args.output if os.path.isabs(args.output) else os.path.join(_HERE, args.output)
    run(out, d=args.d, seeds=args.seeds)
    print(f"\nSaved -> {out}")
