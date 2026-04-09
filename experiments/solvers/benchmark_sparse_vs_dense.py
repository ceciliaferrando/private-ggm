"""benchmark_sparse_vs_dense.py — compare ipm_chol vs ipm_chol_sparse over a grid
of problem sizes and average graph degrees.

Usage
-----
    cd experiments
    python benchmark_sparse_vs_dense.py [--output FILE] [--seeds N]

Output
------
    sparse_vs_dense_results.pkl

    List of dicts, each with keys:
        d           : int — problem dimension
        avg_degree  : float — average degree used to generate the graph
        seed        : int
        n_edges     : int — number of off-diagonal observed edges
        time_dense  : float — wall time for ipm_chol (seconds), or None on failure
        time_sparse : float — wall time for ipm_chol_sparse (seconds), or None on failure
        obj_dense   : float or None
        obj_sparse  : float or None
"""

import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from privateggm.solvers.ipm_chol import ipm_chol
from privateggm.solvers.ipm_chol_sparse import ipm_chol_sparse


# ── Problem grid ──────────────────────────────────────────────────────────────

D_VALUES = [32, 64, 128, 256, 512, 1024, 2048]

# Average degree: expected number of observed off-diagonal neighbours per node.
# fraction_offdiag = avg_degree / (d - 1), so the grid is degree-based.
AVG_DEGREES = [0.5, 1.0, 1.5, 2.0, 2.5]

TAU_DIAG = 1.0   # fixed noise level (interior solution — avoids PSD-boundary issues)
N_SEEDS  = 1     # replication count per (d, avg_degree) cell


# ── Problem generation ────────────────────────────────────────────────────────

def make_problem_by_degree(d, avg_degree, seed=42, tau_diag=TAU_DIAG):
    """Generate a random covariance estimation problem with a given average degree.

    avg_degree controls how many off-diagonal neighbours each node has on average:
        n_off = round(avg_degree * d / 2)   (= avg_degree * d / 2 undirected edges)
    which gives fraction_offdiag = n_off / (d*(d-1)/2) = avg_degree / (d-1).
    """
    rng = np.random.default_rng(seed)

    # Ground-truth covariance
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigvals = np.clip(np.exp(rng.standard_normal(d)), 0.1, 10.0)
    Sigma_true = Q @ np.diag(eigvals) @ Q.T

    # Index set: all diagonals + random off-diagonal edges
    n_off = max(0, round(avg_degree * d / 2))
    n_off = min(n_off, d * (d - 1) // 2)

    diag_inds = [(i, i) for i in range(d)]
    pool      = [(i, j) for i in range(d) for j in range(i + 1, d)]
    chosen    = rng.choice(len(pool), n_off, replace=False)
    off_inds  = [pool[k] for k in sorted(chosen)]

    inds = np.array(diag_inds + off_inds)
    tau  = np.array([tau_diag] * d + [tau_diag * 2.0] * n_off)

    A = np.zeros((d, d))
    for k, (i, j) in enumerate(inds):
        noise   = rng.normal(0, tau[k])
        A[i, j] = Sigma_true[i, j] + noise
        A[j, i] = A[i, j]

    return {'A': A, 'inds': inds, 'tau': tau, 'd': d,
            'avg_degree': avg_degree, 'n_off': n_off, 'seed': seed}


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_solver(solver_fn, prob):
    """Run solver_fn on prob; return (elapsed_seconds, final_obj) or (None, None)."""
    try:
        t0 = time.perf_counter()
        _, hist = solver_fn(prob['A'], prob['inds'], prob['tau'])
        elapsed = time.perf_counter() - t0
        return elapsed, hist[-1]['obj']
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def run(output_path, seeds=N_SEEDS):
    results = []
    total = len(D_VALUES) * len(AVG_DEGREES) * seeds
    done  = 0

    for d in D_VALUES:
        for avg_degree in AVG_DEGREES:
            for seed in range(seeds):
                done += 1
                prob = make_problem_by_degree(d, avg_degree, seed=seed)
                n_edges = prob['n_off']
                actual_avg_deg = 2 * n_edges / d
                print(f"[{done}/{total}]  d={d:>4}  avg_deg={avg_degree:.1f}  "
                      f"(actual={actual_avg_deg:.2f})  |off|={n_edges}", flush=True)

                # Dense solver — skip for very large d to save time
                if d <= 1024:
                    print(f"    ipm_chol       ", end='', flush=True)
                    t_dense, obj_dense = time_solver(ipm_chol, prob)
                    print(f"  {t_dense:.2f}s" if t_dense is not None else "  FAIL")
                else:
                    print(f"    ipm_chol       skipped (d>{1024})")
                    t_dense, obj_dense = None, None

                print(f"    ipm_chol_sparse", end='', flush=True)
                t_sparse, obj_sparse = time_solver(ipm_chol_sparse, prob)
                print(f"  {t_sparse:.2f}s" if t_sparse is not None else "  FAIL")

                results.append({
                    'd':           d,
                    'avg_degree':  avg_degree,
                    'seed':        seed,
                    'n_edges':     n_edges,
                    'time_dense':  t_dense,
                    'time_sparse': t_sparse,
                    'obj_dense':   obj_dense,
                    'obj_sparse':  obj_sparse,
                })

                # Flush partial results after each cell
                with open(output_path, 'wb') as f:
                    pickle.dump(results, f)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark ipm_chol vs ipm_chol_sparse.')
    parser.add_argument('--output', default='sparse_vs_dense_results.pkl',
                        metavar='FILE', help='Output pickle filename')
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        metavar='N', help='Number of random seeds per (d, avg_degree) cell')
    args = parser.parse_args()

    out_path = os.path.join(_HERE, args.output)
    run(out_path, seeds=args.seeds)
    print(f"\nSaved → {out_path}")
