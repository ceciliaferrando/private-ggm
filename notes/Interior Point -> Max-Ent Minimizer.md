## Setup

Consider the convex problem

$$ \min_x \quad f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \quad i = 1, \ldots, m $$

with feasible set $\mathcal{F} = {x : g_i(x) \leq 0}$ and optimal set $X^\star = \arg\min_{x \in \mathcal{F}} f(x)$ with optimal value $f^\star$. Assume Slater's condition holds (nonempty interior) and $f$, $g_i$ are convex.

Define the log-barrier

$$ \phi(x) = -\sum_i \log(-g_i(x)) $$

and the barrier problem

$$ x(\mu) = \arg\min_x \quad f(x) + \mu \phi(x), \qquad \mu > 0. $$

The minimizer $x(\mu)$ is unique for each $\mu > 0$ since $\phi$ is strictly convex.

**Theorem.** As $\mu \to 0$,

$$ x(\mu) \to x^\infty \equiv \arg\min_{x \in X^\star} \phi(x). $$

That is, the central path converges to the minimizer of the barrier over the optimal set.

## Proof

### Step 1: Every limit point lies in $X^\star$

By optimality of $x(\mu)$, for any $y \in X^\star$:

$$ f(x(\mu)) + \mu\phi(x(\mu)) \leq f(y) + \mu\phi(y) = f^\star + \mu\phi(y) $$

Rearranging:

$$ 0 \leq f(x(\mu)) - f^\star \leq \mu(\phi(y) - \phi(x(\mu))) $$

The left inequality holds since $x(\mu)$ is feasible. As $\mu \to 0$ the right side vanishes (for any fixed $y \in X^\star$), so $f(x(\mu)) \to f^\star$. Every limit point of $x(\mu)$ therefore lies in $X^\star$.

### Step 2: Every limit point minimizes $\phi$ over $X^\star$

Let $x^\infty = \lim_{\mu_k \to 0} x(\mu_k)$ be any accumulation point; by Step 1, $x^\infty \in X^\star$. Suppose for contradiction that $x^\infty$ does not minimize $\phi$ over $X^\star$, so there exists $y \in X^\star$ with

$$ \phi(y) < \phi(x^\infty). $$

By continuity of $\phi$, there exists $\epsilon > 0$ such that for all large enough $k$:

$$ \phi(y) < \phi(x(\mu_k)) - \epsilon. $$

By optimality of $x(\mu_k)$:

$$ f(x(\mu_k)) + \mu_k \phi(x(\mu_k)) \leq f(y) + \mu_k \phi(y) $$

Rearranging:

$$ f(y) - f(x(\mu_k)) \geq \mu_k(\phi(x(\mu_k)) - \phi(y)) > \mu_k \epsilon > 0. $$

But $y \in X^\star$ implies $f(y) = f^\star \leq f(x(\mu_k))$, so $f(y) - f(x(\mu_k)) \leq 0$. Contradiction.

Therefore $\phi(x^\infty) \leq \phi(y)$ for all $y \in X^\star$, i.e., $x^\infty$ minimizes $\phi$ over $X^\star$.

### Step 3: The limit is unique

Since $\phi$ is strictly convex, it has a unique minimizer over the convex set $X^\star$. Every accumulation point of $x(\mu)$ equals this minimizer, so $x(\mu) \to x^\infty$. $\blacksquare$

## Application to Maximum Entropy PSD Nearness

In our problem the barrier is $\phi(X) = -\log|X|$, which is strictly convex and equals (up to sign and constants) the negative entropy of a Gaussian with covariance $X$. The theorem guarantees that the interior point method converges to

$$ x^\infty = \arg\min_{X \in X^\star} (-\log|X|) = \arg\max_{X \in X^\star} \log|X|, $$

the **maximum entropy minimizer** of the PSD nearness problem — without requiring a separate solution of the maximum determinant completion problem.