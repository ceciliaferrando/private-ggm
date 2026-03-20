# Project: Private GGM — Covariance Estimation

## What this project is

Privacy-preserving covariance estimation via sparse Gaussian graphical models. The core problem is fitting a covariance matrix to noisy partial observations while enforcing a max-entropy (GGM sparsity) structure. See `notes/` for detailed math.

## Core optimization problem

Given noisy observations $a_{ij}$ at index set $E$ (including all diagonals), noise scales $\tau_{ij}$:

$$\min_{X \succeq 0} \; f(X_E) = \sum_{(i,j)\in E} \frac{1}{\tau_{ij}^2}(X_{ij} - a_{ij})^2$$

The max-entropy solution has $K = X^{-1}$ with sparsity pattern $E$ (GGM structure). Two approaches:

**Two-stage:** Stage A (PGD) finds a PSD matrix fitting the observations → Stage B (max-entropy completion given Stage A output as hard constraints)

**Unified:** Directly minimize $f(X_E)$ subject to $X \succeq 0$ and max-entropy structure, using an IPM that handles the PSD constraint via log-barrier $-\mu \log|X|$

## Notes (mathematical reference)

- `notes/MaxEnt Stage (Stage B).md` — dual formulation, KKT conditions for Stage B
- `notes/Interior Point Method for the PSD Nearness Problem.md` — IPM with Cholesky parameterization
- `notes/MaxEnt via Sparse Precision.md` — K-space parameterization + IPM extension
- `notes/Development Plan.md` — current roadmap, status, next steps, benchmark results
- `notes/Ideas Backlog.md` — speculative ideas, open hypotheses, theory questions
- `notes/dev_log.md` — chronological record of key development steps

## Dev log

Keep `notes/dev_log.md` up to date. Add an entry at the end of any session where a meaningful decision was made, a solver milestone was reached, or a non-obvious change was implemented. Each entry should include:
- The date (YYYY-MM-DD) and a short headline
- What changed and why — especially non-obvious decisions, tradeoffs considered, or dead ends
- Any benchmark numbers if relevant

Do not log routine edits (typos, reformatting). One entry per working session is the right cadence.

## Code layout

```
experiments/
  solvers.py                # all four solvers with uniform interface
  pgd_maxent_harness.ipynb  # comparison harness (uses solvers.py)
  pgd_maxent_simple.ipynb   # original notebook — keep as reference
  problems.py               # (planned) standard benchmark problem instances
privateggm/                 # existing privacy pipeline (don't modify casually)
  model.py                  # IterativePrivateGGM (contains covariance centering fix)
```

## Current algorithms (in `experiments/solvers.py`)

| Name               | Approach                                                              | Status                           |
| ------------------ | --------------------------------------------------------------------- | -------------------------------- |
| `pgd`              | PGD with Armijo backtracking, projection onto PSD cone                | Working                          |
| `pgd_maxent`       | Two-stage: Stage A PGD + Stage B max-entropy completion               | Working; fragile at PSD boundary |
| `ipm_chol`         | IPM with $X = LL^T$ parameterization, scipy L-BFGS-B                 | Best solver so far               |
| `sparse_prec_ipm`  | IPM in K-space ($K = X^{-1}$ sparse), JAX/optax L-BFGS               | Convergence issues               |
| `pgd_maxent_jax`   | Wrapper around `privateggm/model.py` pgd_maxent (student's version): Stage A uses BB line search + non-monotone backtracking; Stage B uses JAX/optax L-BFGS | Added to benchmark               |

## Current focus

**Benchmark development** — comparing all five solvers across the standard problem suite. Active work is in `experiments/pgd_maxent_harness.ipynb` (interactive) and `experiments/benchmark_plots.ipynb` + `experiments/run_benchmark.py` (full benchmark grid).

## Key difficulty

When noise is large, the optimal $X^*$ lies on the **boundary of the PSD cone** (minimum eigenvalue = 0). Methods that only reject non-PD steps (Stage B, naive K-space) have poor control approaching the boundary. The IPM central path ($-\mu \log|X|$ barrier) handles this more robustly.

## Development priorities

See `notes/Development Plan.md` for full status and next steps. Short version:
1. Fix `sparse_prec_ipm` convergence (or decide on an alternative K-space approach)
2. Build `problems.py` benchmark suite (d ∈ {32, 64, 128} × noise ∈ {low, high})
3. Try Clarabel (off-the-shelf conic IPM)
4. Finish mirror descent derivation

## Conventions

- All weights normalized: $\tilde w_{ij} = w_{ij} / \max w$ so objective is O(1) and step sizes are comparable across problems
- `inds` is always lower-triangle including diagonal, shape `(m, 2)`
- `tau` has same length as `inds`
- Solver return: `(Sigma_hat, hist)` where `hist` is list of per-iteration dicts
