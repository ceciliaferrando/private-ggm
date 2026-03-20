# Development Log

Chronological record of key development steps. Add an entry whenever a meaningful decision is made, a solver milestone is reached, or a non-obvious change is implemented.

---

## 2026-03-19

**Reduced redundant O(d³) operations in diagnostics** (`experiments/solvers.py`)

- `project_psd` now accepts `return_evals=True`; pgd linesearch uses returned eigenvalues to compute entropy of the accepted step instead of a separate `slogdet`. Saves 1× `eigh` per PGD iteration.
- `pgd_maxent` Stage B `loss_and_grad`: replaced `slogdet(K)` + `inv(K)` with a single Cholesky — logdet from `2 Σ log diag(L)`, Sigma from Cholesky solve. Saves 1× O(d³) per L-BFGS step.
- `pgd_maxent` Stage B callback: `slogdet(Sigma)[1]` → `-logdet_K` using the identity log|X| = −log|K|. Trivial one-liner.
- `ipm_chol`: removed `min_eig_X` hist field — X = LL^T is always PD, the check was meaningless.

**Resolved sparse_prec_ipm timing decision**

Warmup-before-`t0` cannot separate JAX compile time (warmup ≈ first outer iter cost ~2.7s). Decision: `strip_compile_iter` postprocessor drops the first outer iter and re-zeros the clock. Compile time treated as inseparable; excluded from per-step comparisons.

**Standard benchmark suite (`problems.py`, `run_benchmark.py`, `benchmark_plots.ipynb`)**

- `problems.py`: `make_problem(d, noise_level, seed)` generates random covariance problems matching the harness notebook convention (random sparse off-diagonal subset, fraction_offdiag=0.2, tau_offdiag = 2×tau_diag). Three noise levels: low=0.05, med=0.2, high=0.5. Empirical calibration at d=32 confirms med puts X* near/on the PSD boundary; high is firmly on it for all d.
- `run_benchmark.py`: runs all 4 solvers × 9 problems (d ∈ {16,32,64} × {low,med,high}), saves to `benchmark_results.pkl`.
- `benchmark_plots.ipynb`: loads pickle and shows 3×3 convergence grids (obj vs time, entropy vs time) plus summary table.
- Key finding from first full benchmark: high-noise regime clearly separates solvers. `ipm_chol` and `pgd` agree on objective (both handle PSD boundary correctly); `pgd_maxent` breaks down in high noise (obj 3–5× worse). `sparse_prec_ipm` remains worse than `ipm_chol` across all regimes.

## 2026-03-19 (continued)

**Added `pgd_maxent_jax` to the benchmark**

Wrapped the student's `privateggm/model.py` implementation of `pgd_maxent` (Stage A: projected gradient descent with BB + non-monotone line search; Stage B: max-entropy L-BFGS via JAX/optax) as `pgd_maxent_jax` in `solvers.py` with the uniform benchmark interface. Key changes:

- Instrumented `projected_gradient_descent` and `optimize_lbfgs` in `model.py` to collect per-iteration `(time, obj, entropy)` hist using the same primal objective formula as other solvers.
- `pgd_maxent` in `model.py` now returns `full_convergence_info['hist']` as a combined Stage A + Stage B list tagged with `stage='A'/'B'`.
- `strip_compile_iter` updated to return `(hist, compile_time)` — compile time is now saved in benchmark results and shown in the summary table.
- `run_benchmark.py` now accepts `--solvers` CLI flag for running subsets; `STRIP_JIT` set covers both JAX solvers.
- `benchmark_plots.ipynb` reads `SOLVER_NAMES` from `results['solvers_run']` rather than a hardcoded list.

Key findings from first run with all 5 solvers (d ∈ {16,32,64} × {low,med,high}):

- `pgd_maxent_jax` Stage A is un-JIT'd NumPy, making it 10–30× slower than `pgd_maxent` (12–37s vs 0.1–0.8s). Stage B JAX compile time is only ~0.4s (vs ~5s for `sparse_prec_ipm`).
- In low noise, `pgd_maxent_jax` matches `pgd_maxent` solution quality (same entropy/frob_err) but takes much longer.
- In high noise, Stage A gets stuck: step size collapses to ~5e-7 and stops making progress. Same boundary fragility as `pgd_maxent`.
- `sparse_prec_ipm`'s longer compile time (~5s) explained by 2× linesearch steps (30 vs 15) and `jax.lax.cond` tracing both branches.

---

## 2026-03-19 (continued)

**Extended benchmark to d=128; notebook updated**

Added d=128 (low/med/high) to `PROBLEMS` in `problems.py` and updated `benchmark_plots.ipynb` (`D_VALS = [16, 32, 64, 128]`). Full benchmark run completed.

Key findings at d=128:

- `ipm_chol` remains the best solver: 1.2s (low), 10.1s (med), 6.9s (high), 28 iters, correct objective across all regimes.
- `clarabel` matches `ipm_chol` on objective but is ~5–6× slower (37s low, 62s med, 52s high). Consistent with O(d³) Newton step scaling; chordal decomposition does not help here — our formulation has a single full d×d PSD cone (X ≽ 0) with no sparsity, so Clarabel's chordal feature has nothing to decompose. To exploit chordal structure we'd need a K-space (sparse precision) formulation, but that makes the objective non-quadratic.
- `pgd_maxent_jax` completely uncompetitive at d=128: 686s (med), 912s (high), hits the 2000-iter limit with O(d²) per step and a stuck line search at high noise. Added a note to `run_benchmark.py` recommending `--solvers pgd pgd_maxent ipm_chol sparse_prec_ipm clarabel` for d≥128.
- `sparse_prec_ipm` still gives wrong objectives at high noise (e.g. d=128 high: 1.369e+02 vs correct 1.236e+02).

---

## 2026-03-19 (continued)

**Added `clarabel_solver` — off-the-shelf conic IPM via Clarabel**

Formulated the covariance fitting problem as a QP with PSD cone constraint and implemented `clarabel_solver` in `solvers.py`. Upgraded `clarabel` to 0.11.1 (callbacks were added after 0.10.0).

Key decisions and non-obvious details:

- **Variable encoding**: Clarabel uses upper-triangle column-major order. Our `inds` use upper-triangle off-diagonals (`i < j`, as generated by `make_problem`) despite CLAUDE.md saying "lower-triangle". The safe index formula is `k = max(i,j)*(max(i,j)+1)//2 + min(i,j)`. The initial formula `k = i*(i+1)//2 + j` was wrong for off-diagonal entries and caused completely wrong solutions (obj ~1.8 in the low-noise case instead of ~0).

- **Objective constant**: Clarabel minimizes `(1/2)x^T P x + q^T x`, which equals our `f(X_E) - C` where `C = sum(w * a^2)`. Computed once and added back at each callback iteration: `obj = info.cost_primal + C`.

- **Per-iteration history via callbacks**: `solver.set_termination_callback(fn)` fires at each IPM iteration. The callback `info` object exposes `cost_primal`, `mu`, `res_primal`, `res_dual`, `solve_time`, `iterations`, `status`. Notably `info.status` is always "Unsolved" mid-solve; we patch the final hist entry with `sol.status` after `solver.solve()` returns. Entropy is only computable after reconstructing `Sigma_hat` from `sol.x`, so it's `nan` for intermediate entries and filled in on the final entry.

- **Constraint matrix**: `A_clar = -D` where `D` is diagonal with entries 1 (diagonal of X) and √2 (off-diagonal), encoding `s = vec(X) ∈ PSDTriangleCone(d)`.

**Benchmark results (Clarabel vs ipm_chol):**

- Objective matches `ipm_chol` to solver tolerance across all d and noise levels.
- Low/med noise: entropy nearly identical to `ipm_chol` (both find the same interior solution).
- High noise (PSD boundary): Clarabel finds a *lower* entropy point than `ipm_chol` (e.g. d=64 high: -845 vs -810). Both are on the same optimal face; different IPM barriers find different points. `ipm_chol`'s explicit `-mu*log|X|` barrier is more entropy-biased.
- Speed: faster than `ipm_chol` for small d (d=16: ~10ms vs ~100ms, 9–18 iters vs 28); slower at d=64 (~4s vs ~1.2s).

---
