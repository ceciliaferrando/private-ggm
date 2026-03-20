## Problem

Let $S$ be a set of observed index pairs (including all diagonals) and let $a_{ij}$ be the target values at those positions — in the pipeline these are the entries of the PSD matrix $\tilde\Sigma$ produced by Stage A.  Stage B solves the **maximum entropy covariance completion**:

$$
\max_{\Sigma \succ 0} \; \log|\Sigma|
\quad \text{s.t.} \quad
\Sigma_{ij} = a_{ij}, \quad (i,j) \in S.
$$

Maximising $\log|\Sigma|$ is equivalent to maximising the differential entropy of a zero-mean Gaussian with covariance $\Sigma$.

## Dual problem

Form the Lagrangian with a symmetric dual variable $Z_S$ supported on $S$:

$$
\mathcal{L}(\Sigma, Z_S) = \log|\Sigma| - \langle Z_S,\, \Sigma - A \rangle,
$$

where $A$ is the matrix equal to $a_{ij}$ on $S$ and zero elsewhere.  Setting the gradient to zero gives $\Sigma^{-1} = Z_S$, so the optimal precision matrix $K = \Sigma^{-1}$ is **supported on $S$** (the GGM sparsity pattern).  Substituting back yields the dual:

$$
\min_{K \succ 0,\; \operatorname{supp}(K) \subseteq S} \; -\log|K| + \operatorname{tr}(KA).
$$

## Parameterisation in code

The code parameterises $K$ by a vector $z \in \mathbb{R}^{|S|}$ via `get_K`:

$$
K(z)_{ij} = \begin{cases} 2z_k & i = j \\ z_k & i \neq j \end{cases}
\quad \text{for } (i,j) = \operatorname{inds}[k].
$$

The factor of 2 on the diagonal arises because `get_K` writes $z_k$ to both $K[i,j]$ and $K[j,i]$; for diagonal entries these coincide, so they are added twice.  The initialisation in `initialize_z` halves the diagonal entries to compensate.

With this parameterisation,

$$
\operatorname{tr}(K(z)\,A) = \sum_{k:\, i_k = j_k} 2z_k\, a_{i_k i_k}
+ \sum_{k:\, i_k \neq j_k} 2z_k\, a_{i_k j_k}
= 2\, z \cdot a_{\mathrm{vec}},
$$

so the loss function in `loss_fn` is exactly the dual objective:

$$
\ell(z) = -\log|K(z)| + 2\, z \cdot a_{\mathrm{vec}}.
$$

## Optimality condition

At the minimum $z^*$ the gradient vanishes:

$$
\nabla_z \ell = 0
\implies K(z^*)^{-1}_{ij} = a_{ij} \quad \forall\, (i,j) \in S.
$$

That is, the covariance estimate $\Sigma^* = K(z^*)^{-1}$ exactly interpolates the Stage A values on $S$.  Outside $S$, entries of $\Sigma^*$ are determined by the maximum-entropy principle (the precision matrix is zero there).

## KKT conditions

The problem has equality constraints $\Sigma_{ij} = a_{ij}$ for $(i,j)\in S$ and the implicit constraint $\Sigma \succ 0$ (active only on the boundary).  Form the Lagrangian with a symmetric multiplier $Z_S$ supported on $S$:

$$
\mathcal{L}(\Sigma, Z_S) = \log|\Sigma| - \langle Z_S,\, \Sigma - A \rangle.
$$

**Stationarity** ($\nabla_\Sigma \mathcal{L} = 0$):

$$
\Sigma^{-1} = Z_S \implies K := \Sigma^{-1} \text{ is supported on } S \quad (K_{ij} = 0 \text{ for } (i,j)\notin S).
$$

**Primal feasibility:**

$$
\hat\Sigma_{ij} = a_{ij} \quad \forall\,(i,j)\in S.
$$

Together the two KKT conditions are:

$$
\hat\Sigma_{ij} = a_{ij}\ \forall\,(i,j)\in S
\qquad\text{and}\qquad
\hat K_{ij} = (\hat\Sigma^{-1})_{ij} = 0\ \forall\,(i,j)\notin S.
$$

Because the problem is convex and Slater's condition holds (any $\Sigma \succ 0$ with $\Sigma_S = a_S$ is strictly feasible), these conditions are **necessary and sufficient**.  In the dual parameterisation, sparsity of $K$ is enforced by construction; `constraint_viol` in the code tracks the primal feasibility residual $\sqrt{\tfrac{1}{|S|}\sum_{(i,j)\in S}(\hat\Sigma_{ij}-a_{ij})^2}$.

## Role in the pipeline

Stage A (PGD) produces a PSD matrix $\tilde\Sigma$ that approximately fits the noisy observations.  Stage B then finds the maximum entropy matrix whose entries on $S$ match $\tilde\Sigma|_S$ exactly.  The two stages decouple the feasibility problem (PSD + observed-entry fit) from the completion problem (how to fill in the rest of the matrix).
