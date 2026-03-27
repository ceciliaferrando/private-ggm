"""dr_splitting — Douglas-Rachford splitting for sparse PSD-nearness.

Solves the PSD-completable projection problem:

    min  Σ_{(i,j)∈E}  w_ij (X_ij - a_ij)²
    s.t. X ∈ Π_E(S^p_+)   (PSD-completable with sparsity pattern E)

using D-R splitting on the primal form (Sun & Vandenberghe, §4).  The
optimization variable x = vec_{E'}(X) lives on the chordal extension E' ⊇ E.
Clique submatrix constraints P_{β_k} X P_{β_k} ⪰ 0 are handled as the splitting
variable y[k], with PSD projection as the prox step.

This is essentially a better Stage A — returns a PSD-completable sparse matrix.
Max-entropy completion (Stage B) is a separate step if needed.

Reference
---------
Y. Sun, L. Vandenberghe, "Decomposition methods for sparse matrix nearness
problems", SIAM J. Matrix Anal. Appl., 36(4), 2015.
"""

import time

import networkx as nx
import numpy as np

from ._shared import _weights, project_psd


# ── Chordal completion (pluggable backend) ────────────────────────────────────

def chordal_completion_nx(inds, d):
    """Chordal completion and clique enumeration via networkx.

    Parameters
    ----------
    inds : (m, 2) int array — observed index pairs (including diagonal)
    d    : int — matrix dimension

    Returns
    -------
    edges = (rows, cols) 
        rows : (n,) int array — row indices of chordal pattern (row <= col)
        cols : (n,) int array — col indices of chordal pattern
    cliques   : list of sorted int arrays — maximal cliques β_k of the chordal graph
    """
    obs_rows, obs_cols = inds[:, 0], inds[:, 1]

    G = nx.Graph()
    G.add_nodes_from(range(d))
    for i, j in zip(obs_rows, obs_cols):
        if i != j:
            G.add_edge(i, j)

    H, _ = nx.complete_to_chordal_graph(G)
    cliques = [np.array(sorted(c), dtype=int) for c in nx.chordal_graph_cliques(H)]

    # upper-tri entries (row <= col), including diagonal — matches inds convention
    entries = set((i, i) for i in range(d))
    for i, j in H.edges():
        entries.add((min(i, j), max(i, j)))
    entries = sorted(entries)
    rows = np.array([r for r, c in entries], dtype=int)
    cols = np.array([c for r, c in entries], dtype=int)

    return (rows, cols), cliques


# ── vec / unvec ───────────────────────────────────────────────────────────────

_SQRT2 = np.sqrt(2)


def vec(Y, rows=None, cols=None):
    """Extract upper-triangle entries of Y into a √2-scaled vector.

    Off-diagonal entries are multiplied by √2 so that ‖vec(Y)‖² = ‖Y‖_F².

    vec(Y)             — dense: all upper-tri entries of s×s matrix Y
    vec(Y, rows, cols) — sparse: entries Y[rows, cols]
    """
    if rows is None:
        r, c = np.triu_indices(Y.shape[0])
    else:
        r, c = rows, cols
    v = Y[r, c].copy()
    v[r != c] *= _SQRT2
    return v


def unvec(v, s, rows=None, cols=None):
    """Reconstruct a symmetric matrix from a √2-scaled upper-triangle vector.

    Reverses the scaling applied by vec: off-diagonal entries are divided by √2.

    unvec(v, s)             — dense: s×s matrix from upper-tri entries v
    unvec(v, d, rows, cols) — sparse: d×d matrix, scatter v to (rows, cols)
    """
    if rows is None:
        r, c = np.triu_indices(s)
    else:
        r, c = rows, cols
    v = v.copy()
    v[r != c] /= _SQRT2
    Y = np.zeros((s, s))
    Y[r, c] = v
    Y[c, r] = v
    return Y


# ── Junction tree / separators ────────────────────────────────────────────────

def _build_separators(cliques):
    """Build the junction tree and extract separator information.

    Parameters
    ----------
    cliques : list of sorted int arrays — maximal cliques

    Returns
    -------
    sep_parent     : list of int — for each separator, which clique to extract it from
    sep_local_inds : list of int arrays — local row/col indices of the separator
                     within the parent clique (for use with np.ix_)
    """
    K = len(cliques)
    if K <= 1:
        return [], []

    # Weighted clique-intersection graph; max spanning tree = junction tree
    CG = nx.Graph()
    for i in range(K):
        for j in range(i + 1, K):
            sep = np.intersect1d(cliques[i], cliques[j])
            if len(sep) > 0:
                CG.add_edge(i, j, weight=len(sep))

    sep_parent, sep_local_inds = [], []
    for i, j in nx.maximum_spanning_tree(CG).edges():
        sep = np.intersect1d(cliques[i], cliques[j])
        sep_parent.append(i)
        sep_local_inds.append(np.searchsorted(cliques[i], sep))

    return sep_parent, sep_local_inds


# ── Entropy ───────────────────────────────────────────────────────────────────

def clique_entropy(y, sep_parent, sep_local_inds, evals=None):
    """Entropy (log det) of the max-entropy completion via the junction tree formula:

        log det X = Σ_k log det X_{β_k} - Σ_s log det X_{S_s}

    Parameters
    ----------
    y              : list of (s_k, s_k) clique submatrices
    sep_parent     : list of parent clique indices (from _build_separators)
    sep_local_inds : list of local index arrays into parent clique
    evals          : optional list of eigenvalue arrays for each y[k], e.g. from
                     project_psd(..., return_evals=True); avoids re-eigendecomposing
    """
    ent = 0.0
    for k, Y_k in enumerate(y):
        ev = evals[k] if evals is not None else np.linalg.eigvalsh(Y_k)
        ent += np.sum(np.log(np.maximum(ev, 1e-10)))
    for parent, local in zip(sep_parent, sep_local_inds):
        S = y[parent][np.ix_(local, local)]
        ent -= np.sum(np.log(np.maximum(np.linalg.eigvalsh(S), 1e-10)))
    return float(ent)


# ── Index geometry ────────────────────────────────────────────────────────────

def build_geometry(inds, tau, d, chordal_fn=None):
    """Precompute all index structures for D-R splitting.

    Parameters
    ----------
    inds       : (m, 2) int array — observed index pairs
    tau        : (m,) float array — noise std-devs
    d          : int — matrix dimension
    chordal_fn : callable(inds, d) -> ((rows, cols), cliques),
                 default chordal_completion_nx

    Returns
    -------
    geom : dict with fields:

        edges = (rows, cols) 
            rows : (n,) int array — row indices of chordal pattern (row <= col)
            cols : (n,) int array — col indices of chordal pattern
        cliques     list of sorted int arrays — maximal cliques β_k
        n           int — number of entries in x
        obs_inds    (m,) int — positions in x of observed entries
        clique_inds    list of (n_k,) int arrays — positions in x of upper-tri
                           entries of clique k, in triu row-major order
        n_covers       (n,) int — number of cliques covering each entry of x
        sep_parent     list of int — parent clique index for each junction-tree separator
        sep_local_inds list of int arrays — local row/col indices of separator within
                           parent clique (for use with np.ix_)
        w              (m,) float — normalized objective weights 1/τ²
    """
    if chordal_fn is None:
        chordal_fn = chordal_completion_nx

    # Normalize to row <= col once at the boundary
    obs_rows = np.minimum(inds[:, 0], inds[:, 1])
    obs_cols = np.maximum(inds[:, 0], inds[:, 1])

    edges, cliques = chordal_fn(inds, d)
    (rows, cols) = edges
    n = len(rows)

    # Mapping from (r,c) to index in edge list
    edge2ind = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}

    # Check that all diagonal entries are present (required for PSD-completable projection)
    for i in range(d):
        if (i, i) not in edge2ind:
            raise ValueError(f"Missing diagonal entry ({i}, {i}) in chordal pattern")

    # Positions in edge list of observed entries
    obs_inds = np.array([edge2ind[e] for e in zip(obs_rows, obs_cols)], dtype=int)

    # For each clique β_k, clique_inds[k] maps its triu entries to positions in x
    clique_inds = []
    for beta in cliques:
        s = len(beta)
        r, c = np.triu_indices(s)
        clique_edges = [(beta[r[i]], beta[c[i]]) for i in range(len(r))]
        clique_inds.append(np.array([edge2ind[edge] for edge in clique_edges], dtype=int))

    # Number of cliques covering each entry of x (always >= 1 for chordal graphs)
    n_covers = np.zeros(n, dtype=int)
    for cidx in clique_inds:
        n_covers[cidx] += 1

    sep_parent, sep_local_inds = _build_separators(cliques)

    return dict(
        edges=(rows, cols),
        cliques=cliques,
        n=n,
        obs_inds=obs_inds,
        clique_inds=clique_inds,
        n_covers=n_covers,
        sep_parent=sep_parent,
        sep_local_inds=sep_local_inds,
        w=_weights(tau),
    )


# ── Solver ────────────────────────────────────────────────────────────────────
def _objective(x, w, a, obs_inds):
    """Objective function f(x) = Σ_{i∈obs_inds} w_i (x_i - a_i)²."""
    r = x[obs_inds] - a
    return float(np.sum(w * r ** 2))

def dr_splitting(A, inds, tau,
                 gamma=1.0, rho=1.0,
                 max_iters=2000, eps_abs=1e-5,
                 chordal_fn=None):
    """Douglas-Rachford splitting for the sparse PSD-nearness problem.

    Parameters
    ----------
    A          : (d, d) ndarray — noisy observations (symmetric; only E entries used)
    inds       : (m, 2) int array — observed index pairs
    tau        : (m,) float array — noise std-devs
    gamma      : float — proximal step size (> 0)
    rho        : float — D-R relaxation parameter (0 < rho < 2)
    max_iters  : int — maximum iterations
    eps_abs    : float — absolute convergence tolerance on primal residual
    chordal_fn : callable(inds, d) -> (edges, cliques), default chordal_completion_nx

    Returns
    -------
    Sigma_hat : (d, d) ndarray — estimated covariance (sparse, zeros off E')
    hist      : list of per-iteration dicts with keys:
                    time, obj, r_primal, r_dual
    """
    d = A.shape[0]
    geom = build_geometry(inds, tau, d, chordal_fn)

    (rows, cols) = geom['edges']
    cliques      = geom['cliques']
    n            = geom['n']
    obs_inds     = geom['obs_inds']
    clique_inds    = geom['clique_inds']
    n_covers       = geom['n_covers']
    sep_parent     = geom['sep_parent']
    sep_local_inds = geom['sep_local_inds']
    w              = geom['w']
    n_cliques    = len(cliques)

    obs_rows, obs_cols = inds[:, 0], inds[:, 1]

    # This function uses the convention that vectorization scales off-diagonal entries by 
    # √2, so ‖vec(Y)‖² = ‖Y‖_F². To preserve the objective sum(w * x([obs_inds] - a)²) we 
    # need to scale w by 1/2 for off-diagonal entries. 
    a = vec(A, obs_rows, obs_cols)  # observed entries (√2-scaled off-diag)
    w = w.copy()                    # internal weights for prox step
    w[obs_rows != obs_cols] /= 2.0  # adjust for √2 scaling: off-diag counted once not twice

    # ── Initialization ────────────────────────────────────────────────────────
    # x stores scaled entries; initialise from scaled observations
    x = np.zeros(n)
    x[obs_inds] = a

    # y[k]: clique submatrix of initial X
    y = [unvec(x[clique_inds[k]], len(cliques[k]))
         for k in range(n_cliques)]

    # Dual variables (D-R running sums): u mirrors x, v[k] mirrors y[k]
    u = x.copy()
    v = [Y_k.copy() for Y_k in y]

    hist = []
    t0 = time.perf_counter()

    for _ in range(max_iters):
        # ── Step 1a: prox of weighted quadratic  (x update) ──────────────────
        # x_i = u_i                                for i ∉ obs_inds
        # x_i = (u_i + γ w_j ã_j) / (1 + γ w_j)    for i ∈ obs_inds
        x_new = u.copy()
        x_new[obs_inds] = (u[obs_inds] + gamma * w * a) / (1.0 + gamma * w)

        # ── Step 1b: prox of PSD indicator (y update) ───────────────────────
        # return_evals=True lets clique_entropy reuse these without re-doing eigh
        evals_new = [None] * n_cliques
        y_new     = [None] * n_cliques
        for k in range(n_cliques):
            y_new[k], evals_new[k] = project_psd(v[k], return_evals=True)

        # ── Step 2: project (x, y) onto V = {(x,y) : y_k = P_{β_k} x} ───────
        num = 2.0 * x_new - u
        for k in range(n_cliques):
            num[clique_inds[k]] += vec(2.0 * y_new[k] - v[k])

        p = num / (1.0 + n_covers)

        q = [unvec(p[clique_inds[k]], len(cliques[k]))
             for k in range(n_cliques)]

        # ── Step 3: dual update ───────────────────────────────────────────────
        u += rho * (p - x_new)
        for k in range(n_cliques):
            v[k] += rho * (q[k] - y_new[k])

        # ── Convergence diagnostics ───────────────────────────────────────────
        # Primal residual: ‖p - x_new‖  (gap between prox output and projection)
        # Dual residual:   ‖x_new - x_prev‖ (change in prox output)
        r_primal = float(np.linalg.norm(p - x_new))
        r_dual   = float(np.linalg.norm(x_new - x))

        obj     = float(np.sum(w * (p[obs_inds] - a) ** 2))
        entropy = clique_entropy(y_new, sep_parent, sep_local_inds, evals=evals_new)

        hist.append({
            'time':     time.perf_counter() - t0,
            'obj':      obj,
            'entropy':  entropy,
            'r_primal': r_primal,
            'r_dual':   r_dual,
        })

        x = x_new
        y = y_new

        if r_primal < eps_abs and r_dual < eps_abs:
            break

    Sigma_hat = unvec(p, d, rows, cols)
    return Sigma_hat, hist
