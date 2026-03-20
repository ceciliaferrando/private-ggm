
## Problem

$$ \min_{X \succeq 0} \quad f(X) = \sum_{i} \frac{1}{\tau_{ii}^2}(X_{ii} - a_{ii})^2 + \sum_{(i,j) \in E} \frac{1}{2\tau_{ij}^2} (X_{ij} - a_{ij})^2 $$

## Barrier Formulation

Replace the hard PSD constraint with a log-barrier, parametrized by $\mu > 0$:

$$ \phi_\mu(X) = f(X) - \mu \log |X| $$

As $\mu \to 0$, minimizers of $\phi_\mu$ converge to the solution of the original problem.

## Gradient

Cholesky parameterization: $X = L L^T$, where $L$ lower triangular, rewrite as function of $L$
* Cons: no longer convex AFAIK
* Pros: 
	* $X$ automatically PSD
	* Iterations $O(d^2)$ instead of $O(d^3)$
	* Example: $\log |X| = \log |LL^T| = \log|L|^2 = 2 \log |L| = \sum_{i} \log L_{ii}$

$$ \nabla \phi_\mu(X) = G - \mu X^{-1} $$

where $G = \nabla f(X)$ is the sparse matrix of weighted residuals:

$$ G_{ij} = \begin{cases} \dfrac{2}{\tau_{ii}^2}(X_{ii} - a_{ii}) & i = j \\[6pt] \dfrac{1}{\tau_{ij}^2}(X_{ij} - a_{ij}) & (i,j) \in S \\[6pt] 0 & \text{otherwise} \end{cases} $$

Both $G$ and $X^{-1}$ are symmetric, so $\nabla \phi_\mu$ is symmetric. Work with the $n(n+1)/2$ entries of the lower triangle.

## Algorithm

**Initialize:** $X_0 = \alpha I$ for $\alpha > 0$ large enough that $X_0 \succ 0$. Set $\mu_0$ large.

**Outer loop:** for $k = 0, 1, 2, \ldots$

1. Run L-BFGS on $\phi_{\mu_k}$, warm-started from $X_k$, until $|\nabla \phi_{\mu_k}(X)| \leq \epsilon_{\text{inner}}(\mu_k)$
2. Set $X_{k+1} \leftarrow$ result, $\mu_{k+1} \leftarrow \rho ,\mu_k$ for $\rho \in (0, 1)$

**Inner L-BFGS loop:** standard L-BFGS with Wolfe line search, with the additional requirement that each step maintains $X \succ 0$. Cap the step length $\alpha$ so that $X + \alpha , \Delta X \succ 0$ by checking the minimum eigenvalue along the search direction.

## Stopping Criteria

**Inner loop:** solve to tolerance $\epsilon_{\text{inner}}(\mu) = \min(\mu, \epsilon_{\text{outer}})$ — no need to solve tightly when $\mu$ is large.

**Outer loop:** stop when the KKT residual is small:

$$ |G - \mu X^{-1}|_F \leq \epsilon_{\text{outer}} $$

At convergence, $G \approx \mu X^{-1} \approx 0$ in the directions where $X$ is well-conditioned, consistent with the KKT conditions $G \succeq 0$ and $GX = 0$.