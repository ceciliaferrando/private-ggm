
## Problem

The general form of our problem is
$$
\min_{X \succeq 0} f(X)
$$
## Lagrangian

Introduce dual variable $\Lambda \succeq 0$ for the constraint $X \succeq 0$ and the Lagrangian is:

$$ \mathcal{L}(X, \Lambda) = f(X) - \langle \Lambda, X\rangle $$
Here, $\langle \Lambda, X \rangle = \sum_{ij} \Lambda_{ij} X_{ij}$. Another way to write this is $\langle \Lambda, X \rangle = \operatorname{tr}(\Lambda X)$. 

Justification: the Lagrangian encodes the constraint because when we maximize with respect to $\Lambda$ the maximum value of $-\langle \Lambda, X \rangle$ is $0$ if $X \succeq 0$ and $+ \infty$ otherwise. 

## KKT Conditions

The KKT conditions are 

$$\nabla_X \mathcal L(X, \Lambda) = 0, \quad X \succeq 0, \quad \Lambda \succeq 0, \quad \langle \Lambda, X \rangle = 0$$

In order, these are called: (1) stationarity, (2) primal feasibility, (3) dual feasibility, and (4) complementary slackness.

In our problem, stationarity becomes:
$$
\nabla f(X) - \Lambda = 0 \quad \iff \Lambda = \nabla f(X)
$$
so we can use this to eliminate $\Lambda$ and drop the stationarity condition to get:

**Simplified KKT conditions**
$$ X \succeq 0, \quad \nabla f(X) \succeq 0, \quad \langle \nabla f(X), X \rangle = 0 $$

These can be tracked numerically through:
1. The minimum eigenvalue of $X$ (should be $\geq 0$.)
2. The minimum eigenvalue of $\nabla f(X)$.
3. The absolute value of $\langle \nabla f(X), X \rangle$.

## Our problem

Our specific problem is:
$$ \min_{X \succeq 0} \quad f(X) = \sum_{i} \frac{1}{\tau_{ii}^2}(X_{ii} - a_{ii})^2 + \sum_{(i,j) \in S} \frac{1}{2\tau_{ij}^2} (X_{ij} - a_{ij})^2 $$
The gradient is:
$$[\nabla f(X)]_{ij} = \begin{cases} \dfrac{2}{\tau_{ii}^2}(X_{ii} - a_{ii}) & i = j \\[6pt] \dfrac{1}{\tau_{ij}^2}(X_{ij} - a_{ij}) & (i,j) \in S \\[6pt] 0 & \text{otherwise} \end{cases}$$


The complementary slackness condition becomes:
$$\langle \nabla f(X), X \rangle = \sum_i \frac{2}{\tau_{ii}^2}(X_{ii} - a_{ii}) X_{ii} + \sum_{(i,j)\in S} \frac{1}{\tau_{ij}^2}(X_{ij} - a_{ij}) X_{ij} = 0$$