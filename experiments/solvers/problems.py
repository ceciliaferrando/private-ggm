"""problems.py — standard benchmark problem instances for covariance estimation.

Usage
-----
    from problems import make_problem, PROBLEMS, load_all

    prob = make_problem(d=32, noise_level='high', seed=42)
    S_hat, hist = solve(prob['rows'], prob['cols'], prob['vals'], prob['tau'], prob['d'])

    for spec in PROBLEMS:
        prob = make_problem(**spec)

Problem structure
-----------------
Each problem dict contains:
    Sigma_true   : (d, d) ndarray — ground-truth covariance
    rows         : (m,)   ndarray int — row indices of observed entries (upper-tri + diag)
    cols         : (m,)   ndarray int — col indices of observed entries (upper-tri + diag)
    vals         : (m,)   ndarray float — noisy observed values
    tau          : (m,)   ndarray float — per-entry noise std-devs
    d            : int
    noise_level  : str
    seed         : int

Graph structure
---------------
Random subset of off-diagonal entries (fraction_offdiag=0.2) plus all diagonals.
Off-diagonal tau = 2 * tau_diag (matching harness notebook convention).

Noise calibration
-----------------
Empirically (d=32), tau_diag=0.2 puts the problem near the PSD boundary (ipm_chol
gives min_eig(X*) ≈ 0). Three regimes:
    low  : tau_diag = 0.05  (X* strictly interior to PSD cone)
    med  : tau_diag = 0.2   (X* near/on the PSD boundary)
    high : tau_diag = 0.5   (X* on the PSD boundary)
These may need per-d tuning; run this file directly to check min_eig(X*) for all
PROBLEMS (requires solvers.py on the path).
"""

import numpy as np


# Tau values per noise level
TAU = {
    'low':  0.05,
    'med':  0.2,
    'high': 0.5,
}


def make_problem(d=32, noise_level='med', tau_diag=None, seed=42, fraction_offdiag=0.2):
    """Generate a random covariance estimation problem.

    Parameters
    ----------
    d               : int   — dimension
    noise_level     : str   — 'low', 'med', or 'high'
    seed            : int   — RNG seed
    fraction_offdiag: float — fraction of off-diagonal entries to observe

    Returns
    -------
    dict with keys: Sigma_true, rows, cols, vals, tau, d, noise_level, seed
    """

    # Set tau_diag based on noise_level if not provided
    tau_diag = TAU[noise_level] if tau_diag is None else tau_diag

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigvals = np.clip(np.exp(rng.standard_normal(d)), 0.1, 10.0)
    Sigma_true = Q @ np.diag(eigvals) @ Q.T

    n_off = int(fraction_offdiag * (d * (d - 1) // 2))
    diag_inds = [(i, i) for i in range(d)]
    pool = [(i, j) for i in range(d) for j in range(i + 1, d)]
    chosen = rng.choice(len(pool), min(n_off, len(pool)), replace=False)
    off_inds = [pool[k] for k in sorted(chosen)]

    inds = np.array(diag_inds + off_inds)
    tau  = np.array([tau_diag] * d + [tau_diag * 2.0] * n_off)

    vals = np.array([Sigma_true[i, j] + rng.normal(0, tau[k])
                     for k, (i, j) in enumerate(inds)])

    return {
        'Sigma_true':   Sigma_true,
        'rows':         inds[:, 0],
        'cols':         inds[:, 1],
        'vals':         vals,
        'tau':          tau,
        'd':            d,
        'noise_level':  noise_level,
        'seed':         seed,
    }


# Canonical benchmark suite
PROBLEMS = [
    {'d': 16, 'noise_level': 'low',  'seed': 42},
    {'d': 16, 'noise_level': 'med',  'seed': 42},
    {'d': 16, 'noise_level': 'high', 'seed': 42},
    {'d': 32, 'noise_level': 'low',  'seed': 42},
    {'d': 32, 'noise_level': 'med',  'seed': 42},
    {'d': 32, 'noise_level': 'high', 'seed': 42},
    {'d': 64,  'noise_level': 'low',  'seed': 42},
    {'d': 64,  'noise_level': 'med',  'seed': 42},
    {'d': 64,  'noise_level': 'high', 'seed': 42},
    {'d': 128, 'noise_level': 'low',  'seed': 42},
    {'d': 128, 'noise_level': 'med',  'seed': 42},
    {'d': 128, 'noise_level': 'high', 'seed': 42},
]


def load_all():
    """Return list of problem dicts for all entries in PROBLEMS."""
    return [make_problem(**spec) for spec in PROBLEMS]


if __name__ == '__main__':
    # Calibration check: run ipm_chol on all PROBLEMS and print min_eig(X*).
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from privateggm.solve import solve
    import numpy as np

    print(f"{'d':>4}  {'noise':>5}  {'|obs|':>6}  {'min_eig(X*)':>12}  {'obj':>10}  {'entropy':>10}")
    print('-' * 60)
    for spec in PROBLEMS:
        prob = make_problem(**spec)
        S_hat, hist = solve(prob['rows'], prob['cols'], prob['vals'], prob['tau'], prob['d'],
                            solver='ipm_chol')
        last = hist[-1]
        min_eig = float(np.linalg.eigvalsh(S_hat).min())
        print(f"{prob['d']:>4}  {prob['noise_level']:>5}  {len(prob['rows']):>6}  "
              f"{min_eig:>12.2e}  {last['obj']:>10.3e}  {last['entropy']:>10.3f}")
