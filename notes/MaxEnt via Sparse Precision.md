
## Problem

$$ \min_{X \succeq 0} \quad f(X_E) = \sum_{i} \frac{1}{\tau_{ii}^2}(X_{ii} - a_{ii})^2 + \sum_{(i,j) \in E,, i \neq j} \frac{1}{2\tau_{ij}^2}(X_{ij} - a_{ij})^2 $$

## Reparameterization

The maximum entropy solution has precision matrix $K = X^{-1}$ with sparsity pattern $E$. We optimize over the nonzero entries of $K$:

$$ z = (K_{ij})_{(i,j) \in E} $$

with the constraint $K \succ 0$.

## Subroutines

**Forward pass:** $z \mapsto x_E$

Given $z$, form the sparse symmetric matrix $K$ from its nonzero entries. Compute the dense inverse $X = K^{-1}$ and extract:

$$ x_E = (X_{ij})_{(i,j) \in E} $$

**Loss:** $x_E \mapsto \ell$

$$ \ell = f(x_E) = \sum_{i} \frac{1}{\tau_{ii}^2}(X_{ii} - a_{ii})^2 + \sum_{(i,j) \in E,, i\neq j} \frac{1}{2\tau_{ij}^2}(X_{ij} - a_{ij})^2 $$

**Gradient of $f$ with respect to $x_E$:**

$$ \frac{\partial f}{\partial X_{ij}} = \begin{cases} \dfrac{2}{\tau_{ii}^2}(X_{ii} - a_{ii}) & i = j \[6pt] \dfrac{1}{\tau_{ij}^2}(X_{ij} - a_{ij}) & (i,j) \in E,, i \neq j \end{cases} $$

Collect these into a sparse symmetric matrix $G$ (zero outside $E$).

**Backward pass:** $G \mapsto \nabla_z \ell$

By the chain rule, the gradient of $\ell$ with respect to $K$ is:

$$ \frac{\partial \ell}{\partial K} = -X G X = -K^{-1} G K^{-1} $$

Extract the entries at positions $E$ to get $\nabla_z \ell$:

$$ (\nabla_z \ell)_{ij} = -(XGX)_{ij}, \qquad (i,j) \in E $$

## Algorithm

**Initialize:** $K = I$ (or $K = \operatorname{diag}(1/a_{11}, \ldots, 1/a_{nn})$), extract $z_0$.

**Iterate** (L-BFGS over $z$):

1. Forward pass: form $K$ from $z$, compute $X = K^{-1}$, extract $x_E$
2. Compute loss $\ell = f(x_E)$ and gradient $G$
3. Backward pass: compute $\nabla_z \ell = -(XGX)_E$
4. L-BFGS update of $z$, with line search enforcing $K \succ 0$ (checked via Cholesky)

**Terminate** when $|\nabla_z \ell|$ is below tolerance.

## Notes

- The forward and backward passes currently use dense $O(n^3)$ operations; both can later be replaced by sparse Cholesky / graphical model inference.
- $K \succ 0$ is maintained by checking the Cholesky factorization during line search and rejecting steps that fail.
- The solution $X = K^{-1}$ at convergence is the maximum entropy minimizer of $f$ by construction.

## Interior Point Extension

### Motivation

When the optimal $X^\star$ lies on the boundary of the PSD cone, a line search that merely rejects non-PD steps has poor control over the approach to the boundary. A log-barrier central path provides finer-grained control.

### Log-Barrier Regularization

The log barrier for $X \succ 0$ is $-\log|X| = \log|K|$. The regularized objective is:

$$\ell_\mu(z) = f(X_E) + \mu \log|K|$$

As $X \to$ PSD boundary, $|X| \to 0$ so $\log|K| \to +\infty$, keeping iterates interior. This also acts as max-entropy regularization (maximizing $\log|X|$ = maximizing Gaussian entropy), but that is redundant here since the sparse-K parameterization already guarantees $X = K^{-1}$ is the max-entropy completion of $X_E$.

**Attempted alternative — K-space barrier $-\mu\log|K|$ (does not work):**

One might expect that $-\log|K|$ is the natural barrier for $K \succ 0$ (repels K from 0). Empirically, this causes divergence: $-\mu\log|K|$ attracts K toward 0 (singular), so $X = K^{-1}$ explodes to huge values. The $f(X_E)$ term cannot counteract the barrier, and entropy goes to $-\infty$. The original $+\mu\log|K|$ is the correct barrier.

### Gradient

The gradient of $\mu\log|K|$ w.r.t. $K$ is $\mu K^{-1} = \mu X$. The combined gradient w.r.t. $z$ is:

$$\nabla_z \ell_\mu = \underbrace{-(XGX)_E}_{\text{from } f} + \underbrace{\mu \cdot X_E}_{\text{barrier}}$$

### Algorithm (Outer–Inner IPM)

**Outer loop:** schedule $\mu^{(0)} > \mu^{(1)} > \cdots \to 0$ (e.g., $\mu^{(k+1)} = \rho \mu^{(k)}$ with $\rho \in (0,1)$).

**Inner loop:** for each fixed $\mu^{(k)}$, run L-BFGS on $\ell_{\mu^{(k)}}(z)$ to convergence, warm-starting from the previous solution.