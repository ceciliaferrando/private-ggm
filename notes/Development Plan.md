# Development Plan

## Goal

Find a principled, robust optimizer for the max-entropy covariance fitting problem that performs well across a range of noise levels — including the hard case where the optimal solution lies on the boundary of the PSD cone.

---

## Status

| Task | Status |
|------|--------|
| `ipm_chol` working and robust | ✅ Done |
| `sparse_prec_ipm` initial implementation | ✅ Done (convergence issues, see below) |
| `sparse_prec_ipm` timing / JIT separation | ✅ Done (`strip_compile_iter` postprocessor) |
| Float64 in JAX confirmed | ✅ Done |
| Cholesky-based PD check (replace slogdet) | ✅ Done |
| Zoom linesearch for L-BFGS | ✅ Done |
| Refactor to `solvers.py` | ✅ Done |
| Uniform diagnostics with wall time | ✅ Done |
| Harness notebook (`pgd_maxent_harness.ipynb`) | ✅ Done |
| Standard benchmark problems (`problems.py`) | ✅ Done |
| Clarabel solver | ✅ Done |
| Mirror descent derivation | ⬜ In progress (partial) |
| `sparse_prec_ipm` convergence fix | ⬜ Not started |

---

## Next Steps

### Immediate (next session)

- [x] **`sparse_prec_ipm` timing**: warmup-before-`t0` doesn't separate compile time (warmup costs ~2.7s real compute ≈ first outer iter). Decision: use `strip_compile_iter` postprocessor to drop first iter and re-zero the clock. Compile time is treated as inseparable and excluded from per-step comparisons.

- [x] **Reduce O(d³) operations in diagnostics**: eliminated redundant decompositions:
  - `pgd`: `project_psd` now optionally returns clipped eigenvalues; entropy computed from these instead of a separate `slogdet`.
  - `ipm_chol`: removed `min_eig_X` field (X = LL^T is always PD; the check was meaningless).
  - `pgd_maxent` stage B `loss_and_grad`: replaced `slogdet(K)` + `inv(K)` with a single Cholesky; logdet from `2 Σ log diag(L)`, Sigma from Cholesky solve.
  - `pgd_maxent` stage B callback: replaced `slogdet(Sigma)` with `-logdet_K` (identity log|X| = −log|K|).

### Near term

- [x] **`problems.py`**: create `experiments/problems.py` with `make_problem(d, noise_level, seed)` covering the standard suite (d ∈ {32, 64, 128} × noise ∈ {low, high}). Empirically calibrate `tau_diag` so low-noise gives min_eig(X*) > 0 and high-noise gives min_eig(X*) = 0.

- [ ] **Fix `sparse_prec_ipm` convergence**: see hypotheses in `Ideas Backlog.md`. Most promising first steps: (1) try gradient descent or Adam in K-space as a sanity check on the non-convexity hypothesis; (2) profile where time is spent (compilation vs. per-step cost vs. linesearch evaluations).

- [x] **Clarabel**: formulate the problem as a conic QP and implement a `clarabel_solver(A, inds, tau)` wrapper. See reference section below.

### Ongoing

- [ ] **Mirror descent**: finish derivation of the closed-form update and check whether sparse K structure is preserved. See `Ideas Backlog.md`.

---

## Benchmark Results

### d=128, all noise levels (seed=42)

| Solver | noise | obj | entropy | time (s) |
|--------|-------|-----|---------|----------|
| `pgd` | low | 3.9e-16 | -74.1 | 0.48 |
| `pgd_maxent` | low | 1.9e-14 | 35.55 | 0.39 |
| `ipm_chol` | low | 4.3e-15 | 35.35 | 1.20 |
| `clarabel` | low | 1.4e-12 | 35.48 | 36.91 |
| `sparse_prec_ipm` | low | 2.2e-11 | 35.55 | 18.78† |
| `pgd_maxent_jax` | low | 9.7e-09 | 35.55 | 63.60† |
| `pgd` | med | 1.225 | -1685.3 | 5.44 |
| `pgd_maxent` | med | 1.761 | -524.4 | 7.55 |
| `pgd_maxent_jax` | med | 1.402 | -425.7 | 686† |
| `ipm_chol` | med | 1.222 | -1425.8 | 10.13 |
| `clarabel` | med | 1.222 | -1585.1 | 61.72 |
| `sparse_prec_ipm` | med | 2.733 | -98.7 | 27.84† |
| `pgd` | high | 1.236e+02 | -1867.7 | 3.75 |
| `pgd_maxent` | high | 1.832e+02 | -208.5 | 4.18 |
| `pgd_maxent_jax` | high | 1.243e+02 | -534.9 | 912† |
| `ipm_chol` | high | 1.236e+02 | -1842.1 | 6.88 |
| `clarabel` | high | 1.236e+02 | -1716.3 | 51.94 |
| `sparse_prec_ipm` | high | 1.369e+02 | -133.8 | 24.84† |

† compile time stripped. `pgd_maxent_jax` times include compile (~0.6–0.9s) but are dominated by 686s/912s wall time.

**Key findings at d=128:**
- `ipm_chol` remains fastest correct solver (1–10s). `clarabel` correct but 5–6× slower — O(d³) Newton steps with no sparsity to exploit (see chordal note below).
- `pgd_maxent_jax` completely uncompetitive (686s med, 912s high); O(d²) per-step cost × 2000 iterations. Recommend excluding from d≥128 runs.
- `sparse_prec_ipm` still wrong at med/high noise.
- Chordal decomposition in Clarabel cannot help here: our formulation has a single full d×d PSD cone (X ≽ 0) with no sparsity. Chordal would require a sparse K-space (K = X⁻¹) formulation but that makes the objective non-quadratic.

### d=32, all noise levels (seed=42)

| Solver | noise | obj | entropy | time (s) | iters |
|--------|-------|-----|---------|----------|-------|
| `pgd` | low | 4.0e-29 | 8.93 | 0.00 | 1 |
| `pgd_maxent` | low | 8.3e-15 | 9.84 | 0.01 | 47 |
| `ipm_chol` | low | 1.8e-15 | 9.82 | 0.06 | 28 |
| `clarabel` | low | 7.1e-14 | 9.83 | 0.26 | 14 |
| `sparse_prec_ipm` | low | 3.2e-12 | 9.84 | 0.43† | 12 |
| `pgd` | med | 1.2e-3 | -102.3 | 0.25 | 2000 |
| `pgd_maxent` | med | 5.9e-3 | -10.67 | 0.27 | 2500 |
| `ipm_chol` | med | 1.2e-3 | -16.19 | 0.15 | 28 |
| `clarabel` | med | 1.2e-3 | -14.76 | 0.20 | 15 |
| `sparse_prec_ipm` | med | 6.5e-3 | -2.79 | 1.02† | 12 |
| `pgd` | high | 4.1e+0 | -413.0 | 0.28 | 2000 |
| `pgd_maxent` | high | 1.2e+1 | -39.14 | 0.20 | 2019 |
| `ipm_chol` | high | 4.1e+0 | -365.9 | 0.40 | 28 |
| `clarabel` | high | 4.1e+0 | -392.7 | 0.32 | 18 |
| `sparse_prec_ipm` | high | 4.9e+0 | -31.61 | 1.05† | 12 |

† compile time stripped

**Key findings:**
- `ipm_chol` and `clarabel` agree on objective across all regimes; both handle the PSD boundary correctly
- `pgd_maxent` breaks down at high noise (obj 3× worse than optimal)
- Low/med noise: `clarabel` entropy matches `ipm_chol` closely — Clarabel's cone barrier incidentally gives near-max-entropy behavior
- High noise: `clarabel` entropy is slightly *more negative* than `ipm_chol` (both on the optimal face, different barrier choices lead to different points)
- `clarabel` is faster than `ipm_chol` at small d (d=16: ~10ms vs ~100ms) but slower at d=64 (~4s vs ~1.2s)
- `sparse_prec_ipm` still not converging (obj worse than `ipm_chol`, entropy far from correct)

---

## Reference: Solver Interface

### Uniform signature

```python
def solver_name(A, inds, tau, **kwargs) -> tuple[np.ndarray, list[dict]]:
    """
    A         : (d, d) observation matrix (zero at unobserved entries)
    inds      : (m, 2) observed index pairs, lower-triangle including diagonal
    tau       : (m,)   noise std-dev for each entry in inds

    Returns (Sigma_hat, hist) where hist entries contain at minimum:
      time    — cumulative wall-clock seconds since solver start
      obj     — weighted objective f(X_E)  (bare, no barrier)
      entropy — log|X|
    """
```

### Uniform parameter names

| Name | Meaning |
|------|---------|
| `eps` | Final gradient tolerance |
| `eps_coarse` | Coarse inner tolerance for IPM outer loops |
| `max_outer` | Max outer (μ-reduction) iterations (IPM) |
| `max_inner` | Max inner L-BFGS steps per outer iteration (IPM) |
| `mu_init` | Initial barrier weight |
| `mu_factor` | Geometric decrease factor for μ |

### Hist fields

**Uniform (all solvers):**

| Key | Description |
|-----|-------------|
| `time` | Cumulative wall-clock seconds |
| `obj` | Weighted objective f(X_E) — primary comparison metric |
| `entropy` | log\|X\| — tracks max-entropy behavior |

**IPM-specific (outer-loop entry only):**

| Key | Description |
|-----|-------------|
| `mu` | Current barrier parameter |
| `inner_iters` | Inner L-BFGS steps at this μ level |

**Algorithm-specific debugging:**

| Algorithm | Extra fields |
|-----------|-------------|
| `pgd` | `delta`, `eta`; optionally `min_eig_X`, `min_eig_G`, `comp_slack` (if `track_kkt=True`) |
| `pgd_maxent` | `stage` ('A'/'B'); stage B adds `dual_loss`, `constraint_viol`, `grad_norm` |
| `ipm_chol` | `min_eig_G`, `comp_slack` |
| `sparse_prec_ipm` | `grad_norm` |

### Clarabel formulation (reference, not yet implemented)

The problem is a QP with PSD cone constraint, directly handled by Clarabel:

$$\min_{x} \tfrac{1}{2} x^T P x + q^T x \quad \text{s.t.} \quad \text{mat}(x) \succeq 0$$

where $x$ is the lower triangle of $X$, $P$ encodes the weighted squared-error objective, and the PSD constraint is expressed as a standard SDP cone constraint. Interface via the `clarabel` Python package.
