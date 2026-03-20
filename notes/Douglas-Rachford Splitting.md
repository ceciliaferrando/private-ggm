$$
\newcommand{\psd}{\mathbf{S_+}}
\DeclareMathOperator{\prox}{prox}
\DeclareMathOperator*{\argmin}{argmin}
$$

We are given noisy observations of some entries of a PSD matrix $\Sigma$ including all of the diagonal entries:
$$
a_{ij} = \Sigma_{ij} + \mathcal{N}(0, \tau_{ij})^2, \quad (i,j) \in S
$$
where $S$ is a set of ordered pairs $(i,j)$ with $i \geq j$ and $(i, i) \in S$ for all $i$. That is, the diagonal entries are always observed, and we only record an "observation" for entries in the lower triangle of the matrix, since the others are determined by symmetry.

Our problem is to find the maximum entropy PSD matrix that minimizes the squared error to the observed entries. This can be split into two steps.

1. Find a PSD matrix $X$ whose observed entries minimize the log-likelihood of the observations:
	$$
	\begin{aligned}
	\text{minimize} \quad &\sum_{(i,j) \in S} \frac{1}{\tau_{ij}^2}(X_{ij} - a_{ij})^2 \\
	\text{s.t.} \quad &X \succeq 0
	\end{aligned}
	$$
	It can be shown the vector $z$ of the entries of $X$ in positions corresponding to $S$ is uniquely determined by this optimization problem. Clearly the objective depends only on $z$ and is strictly convex in $z$. The problem can be rewritten to minimize this objective over all vectors $z \in \{ \text{vec}\left((X_{ij})_{(i,j) \in S}: X \succeq 0 \right)\}$ that can be completed into a PSD matrix. The feasible set is the set of restrictions of a PSD matrix to a subset of its entries, which is convex. Therefore, there is a unique minimizer $z^*$.)

2. Find the maximum determinant PSD completion of the matrix with entries matching $X$ in positions corresponding to $S$. This gives the maximum-entropy minimizer of the problem in part 1. 
    $$
	\begin{aligned}
	\text{maximize} \quad & \log |\Sigma| \\
	\text{s.t.} \quad &\Sigma_{ij} = X_{ij} \quad (i, j) \in S, \\
	&\Sigma \succeq 0
	\end{aligned}
	$$

Problem 2 is well studied. Section 10.2 of the monograph by Vandenberghe and Anderson describes how to solve this efficiently by a sparse Cholesky factorization when the sparsity pattern $S$ is *chordal*. We have an L-BFGS implementation that uses some of the ideas from https://www.seas.ucla.edu/~vandenbe/publications/covsel1.pdf

# Finding the PSD matrix closest to observed entries 

We focus on first problem. To simplify the derivation, we rewrite the objective in a way that depends on both the upper and lower triangle:
$$
\begin{aligned}
f(X) &= \sum_{(i,j) \in S} \frac{1}{\tau_{ij}^2}(X_{ij} - a_{ij})^2 \\
&= \frac{1}{2}\sum_{(i,j) \in S}\frac{1}{\tau_{ij}^2}(X_{ij} - a_{ij})^2  + \frac{1}{2}\sum_{(j,i) \in S} \frac{1}{\tau_{ij}^2}(X_{ij} - a_{ij})^2 \\
&= \sum_{i} \frac{1}{\tau_{ii}^2}(X_{ii} - a_{ii})^2 + \frac{1}{2}\sum_{(i,j) \in E} \frac{1}{\tau_{ij}^2}(X_{ij} - a_{ij})^2.
\end{aligned}
$$
The expression on Line 2 is equal to the one on Line 1 by symmetry. In Line 3 we reorganize and notice that each diagonal entry is counted twice. The set $E = \{ (i, j): i \neq j, (i, j) \in S \text{ or } (j, i) \in S\}$ is the set of nonzero off-diagonal entries from both the upper and lower triangle.

The problem is now
$$
\begin{aligned}
\text{minimize} \quad  &f(X) = \sum_{i} \frac{1}{\tau_{ii}^2}(X_{ii} - a_{ii})^2 + \sum_{(i,j) \in E} \frac{1}{2\tau_{ij}^2} (X_{ij} - a_{ij})^2\\
\text{s.t.} \quad &X \succeq 0
\end{aligned}
$$
Note: A version of this problem that never builds a dense matrix $X$ has been studied in the literature as the projection of a symmetric sparse matrix $A$ onto the set of *PSD-completable sparse* matrices with the same sparsity pattern (cite: Decomposition Methods for Sparse Matrix Nearness Problems by Sun and Vandenberghe). We will focus on the version stated here where the optimization variable is a dense matrix $X$.
### Projected Gradient Descent
One method to solve this problem is projected gradient descent. The gradient $G = \nabla f(X)$ is given by:
$$
G_{ij} = \begin{cases}
\frac{2}{\tau_{ij}^2} (X_{ij} - a_{ij}) & i = j \\
\frac{1}{\tau_{ij}^2} (X_{ij} - a_{ij}) & (i, j) \in E \\
0 & \text{otherwise}
\end{cases}
$$
The algorithm is to take a gradient step then project onto the PSD cone.

Repeat
1. Gradient step: $X \gets X - \eta G$
2. Project to be PSD: $X \gets \Pi_{X \succeq 0}(X)$

**Note**: the factor of 2 in the gradient for the diagonal entries seems surprising but I verified it is correct in a different way: if we instead use the original non-symmetric objective we get gradient $H$ with $H_{ij} = \frac{2}{\tau_{ij}}(X_{ij} - a_{ij})$ for $(i,j) \in S$ and 0 otherwise. But this is not symmetric, so neither is $X' \gets X - \eta H$ and we need to project it onto the set of symmetric matrices as $X'' \gets (X' + X'^T)/2 = X - \eta(H + H^T)/2$. This gives the same update, since $G = (H + H^T)/2$.
## Douglas-Rachford Splitting

A second way to solve the problem is Douglas-Rachford splitting. Write the problem as
$$
\min tf(X) + tg(X)
$$
where $t>0$ is an arbitrary scaling factor and $g(X) = \delta_{\psd} (X)$ is the indicator function for the PSD cone, which takes value $0$ when $X \succeq 0$ and $+\infty$ when $X \not \succeq 0$. The Douglas-Rachford algorithm initializes $Y_0$ arbitrarily and iterates as
$$
\begin{aligned}
X_{k+1} &= \prox_{tf}(Y_k) \\
Y_{k+1} &= Y_k + \prox_{tg}(2X_{k+1}-Y_k)-X_{k+1}
\end{aligned}
$$

The prox operators are
$$
\begin{aligned}
\prox_{tf}(Y) &= X, &\text{where } X_{ij} &= \begin{cases}
\frac{2ta_{ij}+\tau_{ij}^2 Y_{ij}}{2t + \tau_{ij}^2} & i = j \\
\frac{ta_{ij}+\tau_{ij}^2 Y_{ij}}{t + \tau_{ij}^2} & (i, j) \in E \\
Y_{ij} & \text{else}
\end{cases}
\\
\prox_{tg}(Z) &= \Pi_{\psd}(Z)
\end{aligned}
$$
where $\Pi_{\psd}$ is the projection onto the PSD cone.

These can be derived as follows
$$
\begin{aligned}
\prox_{tf}(Y) &= \argmin_{X} \left\{ tf(X) + \frac{1}{2} \|X - Y\|_F^2\right\} \\
&= \argmin_{X} \left\{f(X) + \frac{1}{2t} \|X - Y\|_F^2 \right\} \\
&= \argmin_{X} \Bigg\{ \sum_{i}\bigg( \frac{1}{\tau_{ii}^2} (X_{ii} 
- a_{ii})^2 + \frac{1}{2t}(X_{ii}-y_{ii})^2 \bigg) \\
& \qquad \qquad + \sum_{(i,j) \in E} \bigg( \frac{1}{2\tau_{ij}^2} (X_{ij} - a_{ij})^2 + \frac{1}{2t}(X_{ij}-y_{ij})^2 \bigg) \\
& \qquad \qquad + \sum_{(i,j) \notin E, i \neq j} \frac{1}{2t}(X_{ij}-y_{ij})^2 \Bigg\} \\
\end{aligned}
$$
The final expression can be optimized independently across the entries of $X$ to give the result, using the following scalar formula, which is easily derived through calculus:
$$
\argmin_{x} \frac{1}{c}(x-a)^2 + \frac{1}{d}(x-y)^2 = \frac{da + cy}{c + d}
$$
For $\prox_{tg}(Z)$ we have
$$
\prox_{tg}(Z) = \argmin_{U} \left\{ tg(U) + \frac{1}{2}\|U-Z\|_F^2 \right\} = \argmin_{U \succeq 0} \frac{1}{2}\|U - Z\|_F^2
$$
which is the projection of $Z$ onto the PSD cone.

