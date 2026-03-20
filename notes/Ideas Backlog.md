# Ideas Backlog

Speculative ideas, open hypotheses, and theory questions. Add freely; review when deciding what to work on next. Concrete action items belong in `Development Plan.md`, not here.

---

## Algorithmic Ideas

### `sparse_prec_ipm` convergence hypotheses

- **Hypothesis 1:** Non-convexity of $f(K^{-1}_E) + \mu \log|K|$ in K-space means L-BFGS Hessian approximation is poor despite Wolfe conditions. Try gradient descent or Adam in K-space as a sanity check.
- **Hypothesis 2:** The problem is ill-conditioned at the current iterate. Try diagonal preconditioning or normalizing K entries.
- **Hypothesis 3:** The barrier parameter $\mu$ is too small, so the log|K| term is negligible and we're effectively minimizing the non-convex $f(K^{-1}_E)$ naked.
- **Alternative:** Abandon K-space parameterization. Work in X-space directly (convex) and use a projection or barrier to enforce $X^{-1}$ sparse — or accept that sparse K structure is only enforced at convergence.

**What we know from benchmark data (2026-03-19):**

The inner L-BFGS loop behaviour differs sharply by noise level:
- **Low noise:** inner loop converges in 45–134 steps per outer iter, grad_norm reaches tolerance. Solver works.
- **High noise (d=32 and d=128):** inner loop hits `max_inner=200` on *every* outer iteration, grad_norm stays large (0.1–14), never decreases across outer iterations. This is not a backtracking / linesearch issue — the linesearch (zoom, `while_loop`) exits early when it finds a valid step. The L-BFGS is genuinely failing to make progress on the inner subproblem.

Per-iteration timing at d=128 high-noise is ~1.5–1.7s for 200 inner steps (~8ms/step), consistent with one Cholesky + solve per linesearch evaluation at d=128 — not anomalously slow.

**Implication:** The root failure is that L-BFGS cannot minimize $f(K^{-1}_E) + \mu\log|K|$ in K-space on high-noise problems. The Hessian approximation is likely wrong (non-convex landscape), not just slow.

### Ablate the log barrier (no-barrier K-space solver)

**Motivation:** The log-barrier $+\mu\log|K|$ was added to (a) keep $K \succ 0$ and (b) encourage max-entropy structure. But it couples with the outer $\mu$-annealing schedule and makes the objective non-convex in a more complex way. It's unclear whether it's helping or hurting convergence.

**Proposed experiment:** Implement `sparse_prec_nobarrier` — single-phase L-BFGS directly minimizing $f(K^{-1}_E)$ over $z = (K_{ij})_{(i,j)\in E}$, with non-PD rejection (return 1e30 on Cholesky failure) but no log-barrier term and no outer $\mu$ loop. Compare against `sparse_prec_ipm`, `ipm_chol`, and `pgd` at low $d$ (16, 32) across noise levels.

This isolates whether the barrier + annealing are the source of convergence issues, or whether the underlying K-space non-convexity is the fundamental problem.

If the no-barrier version also fails at high noise → non-convexity is the issue, and the barrier cannot fix it.
If the no-barrier version succeeds → the outer loop / barrier schedule is actively harmful.

### Coordinate descent in K-space

- For fixed $X_E$, the Stage B dual is separable by row/column with the Schur complement formula (GLASSO coordinate descent)
- May be more robust than L-BFGS for the non-convex K-space problem
- Reference: Friedman et al. 2007 (graphical lasso)

### ADMM for Stage B / unified problem

- ADMM can split the PSD constraint from the sparsity/fit constraint
- One subproblem: project onto PSD cone (cheap via eigendecomposition)
- Other subproblem: minimize fit term (quadratic, closed form)
- May be more robust at the boundary than line-search methods

### Warm-starting `ipm_chol` from Stage A

- Currently initializes from $X = \alpha I$, ignoring Stage A output
- Could warm-start from Stage A's $\tilde X$ by finding the nearest lower-triangular factor
- Would likely reduce outer iterations significantly

### Adaptive μ schedule

- Current geometric schedule $\mu_{k+1} = \rho \mu_k$ is fixed
- Could adapt: if inner loop converges quickly, decrease μ more aggressively; if it struggles, slow down
- Standard IPM theory: use μ proportional to duality gap

### Second-order (Newton) methods

- For small $d$ (≤ 32), a full Newton step on the barrier subproblem would converge quadratically
- The Hessian of $\phi_\mu(X) = f(X_E) - \mu \log|X|$ w.r.t. lower triangle of $X$ is dense but structured
- Could compute explicitly and use scipy's Newton-CG or direct Cholesky solve

---

## Entropy Regularization as a Statistical Estimator

**Observation (anecdotal):** Not fully minimizing $f(X_E)$ — stopping with a higher-entropy solution — gave lower error with respect to the true (unknown) covariance $\Sigma^*$ in several cases. The max-entropy bias acts as a useful regularizer.

**Hypothesis:** The right amount of regularization depends on the regime:

- **Interior case** (optimal $X^*$ strictly PSD, low noise): the unconstrained minimizer already has reasonable entropy; additional regularization may hurt.
- **Boundary case** (optimal $X^*$ on the PSD boundary, high noise): the strict minimizer collapses toward a degenerate matrix; entropy regularization pulls it toward a more statistically stable estimate.

**Conjecture:** There is an optimal $\mu^* > 0$ minimizing expected estimation error, larger in the boundary regime than the interior. The question is not just "how do we optimize?" but "where on the central path should we stop?"

**To explore (after solver robustness is settled):**
- Verify empirically: plot error vs $\Sigma^*$ as a function of $\mu$ for fixed problem
- Check whether optimal $\mu$ correlates with noise level / boundary proximity
- Held-out validation criterion for selecting $\mu$ (analogous to cross-validation)
- Connection to existing theory: Ledoit-Wolf shrinkage, James-Stein, empirical Bayes

---

## Implementation / Engineering

### `pgd_maxent_jax` Stage A: JIT-compile or replace the inner loop

Stage A (`projected_gradient_descent`) is pure Python/NumPy — up to 2000 iterations each
calling `jnp.linalg.eigh` (no JIT) for the PSD projection, plus `gold_standard_line_search`
in Python. This makes `pgd_maxent_jax` 10–30× slower than `pgd_maxent` despite doing
essentially the same Stage A work. Benchmark: 12–37s vs 0.1–0.8s at d=16–64.

Options:
- **JIT the projection loop:** Rewrite Stage A as a `jax.lax.while_loop` or `jax.lax.fori_loop`
  so the whole stage compiles. Would add compile time but drastically reduce per-iter cost.
- **Replace with `solvers.pgd`:** Stage A doesn't need to be the student's implementation.
  The existing Armijo-PGD (`solvers.pgd`) is faster for the same task; could call it from
  the wrapper and only use `optimize_lbfgs` for Stage B.
- **Accept the cost:** If the goal is benchmarking the student's algorithm as-is, slowness
  is informative. Just note it in comparisons.

### `sparse_prec_ipm` vs `pgd_maxent_jax` compile time gap (~5s vs ~0.4s)

`sparse_prec_ipm` compiles much longer because:
1. `max_linesearch_steps=30` (vs 15 in `optimize_lbfgs`) — zoom linesearch loop is unrolled
   into the XLA graph at compile time.
2. `jax.lax.cond` in the loss function — both branches (valid/invalid Cholesky) are fully
   traced, doubling the compiled graph size.
3. `mu` passed as a traced JAX array adds a dynamic dependency through the whole graph.

If compile time matters, reducing `max_linesearch_steps` or removing the `jax.lax.cond`
(replacing with a Python-level guard) would help.

### Sparse matrix operations

- Current `ipm_chol`: $O(d^3)$ per iteration (dense matrix multiply)
- For sparse $E$ and chordal graphs, can reduce to $O(|E| \cdot d)$ using sparse Cholesky
- Only matters at large $d$ (≥ 128)

### GPU acceleration

- JAX `sparse_prec_ipm` could run on GPU for large $d$ with minimal changes
- Worth testing once algorithm is working correctly on CPU

### Gradient checking

- Add numerical gradient check for all solver gradients
- Useful for debugging `sparse_prec_ipm` convergence
- `jax.test_util.check_grads` or manual finite differences

---

## Mathematical / Theory Questions

### Convexity of $f(K^{-1}_E)$

- Is $f(K^{-1}_E)$ convex in $K$ for any special case? (e.g. diagonal $K$, scalar case)
- What is the condition number of the Hessian as a function of $K$?

### Mirror descent derivation

- Update rule for mirror map $\phi(X) = -\log|X|$ (Bregman projection onto sparse-K manifold)
- Key question: is the sparse K structure preserved after each Bregman step?
- If not, what is the right projection back?

### Duality gap for the unified problem

- The unified problem (minimize $f(X_E)$ s.t. $X \succeq 0$, $K = X^{-1}$ sparse) has a dual
- Can we track the duality gap as a stopping criterion?

### When does the high-noise case land on the boundary?

- Characterize exactly when $\lambda_\min(X^*) = 0$
- Is there a closed-form condition on $A$ and $\tau$?
