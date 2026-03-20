# Clarabel Formulation

## Clarabel's standard form

$$\min_{x} \; \tfrac{1}{2} x^T P x + q^T x \quad \text{s.t.} \quad Ax + s = b, \quad s \in \mathcal{K}$$

where $x \in \mathbb{R}^n$ is the decision variable, $P \succeq 0$ encodes the quadratic objective, and $s \in \mathcal{K}$ is the slack in the cone $\mathcal{K}$.

---

## Our problem

$$\min_{X \succeq 0} \; f(X_E) = \sum_{(i,j) \in E} \frac{1}{\tau_{ij}^2} (X_{ij} - a_{ij})^2$$

$X$ is $d \times d$ symmetric PSD. $E$ is the observed index set (lower triangle including diagonal). $a_{ij}$ are noisy observations, $\tau_{ij}$ are noise standard deviations.

---

## Variable encoding

Clarabel works with the **upper triangle of $X$ in column-major order**, stored as $x \in \mathbb{R}^{d(d+1)/2}$.

Column-major upper-triangle ordering for a $d \times d$ matrix: column $c$ contributes rows $0, 1, \ldots, c$, so the global index for entry $(r, c)$ with $r \leq c$ is:

$$k(r, c) = \frac{c(c+1)}{2} + r$$

Our `inds` may contain either lower- or upper-triangle off-diagonal pairs (the solvers are symmetric in $A$). In practice `make_problem` generates upper-triangle off-diagonal entries ($i < j$). By symmetry $X_{ij} = X_{ji}$, so we use the upper-triangle element $(j, i)$:

$$k_\ell = \frac{\max(i,j)\,(\max(i,j)+1)}{2} + \min(i,j) \quad \text{for each } (i,j) \in E$$

This handles either triangle convention and is one-to-one.

---

## Objective: $P$ and $q$

Using normalized weights $\tilde{w}_{ij} = w_{ij} / \max_{E} w$ with $w_{ij} = 1/\tau_{ij}^2$, the objective expands to:

$$f(X_E) \propto \sum_{(i,j) \in E} \tilde{w}_{ij}(x_{k_\ell} - a_{ij})^2 = \underbrace{\sum_\ell \tilde{w}_\ell x_{k_\ell}^2 - 2\tilde{w}_\ell a_\ell x_{k_\ell}}_{\frac{1}{2}x^T P x + q^T x} + \underbrace{\sum_\ell \tilde{w}_\ell a_\ell^2}_{C}$$

Matching $\tfrac{1}{2} x^T P x + q^T x$:

$$P_{k_\ell, k_\ell} = 2\tilde{w}_\ell, \qquad q_{k_\ell} = -2\tilde{w}_\ell a_\ell$$

$P$ is **diagonal** (no cross-terms since each variable appears in at most one observation). $q_k = 0$ for unobserved entries.

The constant $C = \sum_\ell \tilde{w}_\ell a_\ell^2$ is dropped from the Clarabel objective but must be added back to recover our cost:

$$f(X_E) = \texttt{cost\_primal} + C$$

This allows us to track our objective from the per-iteration callback (which exposes `info.cost_primal`) without access to the current iterate $x$.

---

## PSD constraint: $A$, $b$, and cone

Clarabel's `PSDTriangleConeT(d)` uses the scaled vectorization:

$$\text{vec}(X)_k = \begin{cases} X_{rc} & r = c \\ \sqrt{2}\, X_{rc} & r \neq c \end{cases}$$

This scaling preserves inner products: $\langle A, B \rangle = \text{tr}(AB) = \text{vec}(A)^T \text{vec}(B)$.

The constraint $X \succeq 0 \iff \text{vec}(X) \in \mathcal{K}_{\text{tri}}^d$ is encoded as:

$$Ax + s = 0 \quad \Longrightarrow \quad s = -Ax = Dx = \text{vec}(X)$$

where $D = \operatorname{diag}(d_k)$ with:

$$d_k = \begin{cases} 1 & \text{entry } k \text{ is diagonal in } X \\ \sqrt{2} & \text{entry } k \text{ is off-diagonal in } X \end{cases}$$

So $A = -D$ (diagonal, $d(d+1)/2 \times d(d+1)/2$), $b = 0$, cone = `PSDTriangleConeT(d)`.

No equality constraints are needed (unlike the Clarabel SDP example which has $\langle A, X \rangle = 1$).

---

## Summary

| Quantity | Value |
|----------|-------|
| $n$ | $d(d+1)/2$ (upper-triangle variables) |
| $P$ | diagonal, $P_{kk} = 2\tilde{w}$ for observed entries, $0$ elsewhere |
| $q$ | $q_k = -2\tilde{w}\,a$ for observed entries, $0$ elsewhere |
| $A$ | $-D$, diagonal with $1$ (diagonal entries) and $\sqrt{2}$ (off-diagonal) |
| $b$ | $0$ |
| Cone | `PSDTriangleConeT(d)` |

---

## Max-entropy conjecture

We are not explicitly adding a log-determinant term. However, Clarabel uses a primal-dual interior-point method with a log-barrier for the PSD cone. Near the solution, the central-path equations enforce:

$$X S = \mu I, \quad S = X^{-1} \mu$$

where $S$ is the dual slack. If the optimal unconstrained minimizer lies on the PSD boundary, the barrier pushes $X$ into the interior along the central path. The limiting solution as $\mu \to 0$ may or may not coincide with the max-entropy completion — this is an empirical question to check against `ipm_chol`.
