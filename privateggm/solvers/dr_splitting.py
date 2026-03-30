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

import functools
import itertools
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ._shared import _weights, project_psd


# ── vec / unvec ───────────────────────────────────────────────────────────────

_SQRT2 = np.sqrt(2)

@functools.lru_cache(maxsize=None)
def _triu_rc(s):
    return np.triu_indices(s)


def vec(Y, rows=None, cols=None):
    """Extract upper-triangle entries of Y into a √2-scaled vector.

    Off-diagonal entries are multiplied by √2 so that ‖vec(Y)‖² = ‖Y‖_F².

    vec(Y)             — dense: all upper-tri entries of sxs matrix Y
    vec(Y, rows, cols) — sparse: entries Y[rows, cols]
    """
    r, c = _triu_rc(Y.shape[0]) if rows is None else (rows, cols)
    v = Y[r, c]
    v[r != c] *= _SQRT2
    return v


def unvec(v, s, rows=None, cols=None):
    """Reconstruct a symmetric matrix from a √2-scaled upper-triangle vector.

    Reverses the scaling applied by vec: off-diagonal entries are divided by √2.

    unvec(v, s)             — dense: sxs matrix from upper-tri entries v
    unvec(v, d, rows, cols) — sparse: dxd matrix, scatter v to (rows, cols)
    """
    r, c = _triu_rc(s) if rows is None else (rows, cols)
    v = v.copy()
    v[r != c] /= _SQRT2
    Y = np.zeros((s, s))
    Y[r, c] = v
    Y[c, r] = v
    return Y


# ── Junction tree ─────────────────────────────────────────────────────────────

@dataclass
class JunctionTree:
    """Junction tree of a chordal graph.

    Cliques are stored in BFS traversal order from the root (cliques[0]).
    This guarantees parents[j] < j for all j > 0, so a root-to-leaves pass
    is forward iteration and a leaves-to-root pass is reverse iteration.

    Fields
    ------
    cliques : list of sorted int arrays — maximal cliques, traversal order
    parents : list — parents[0] = None; parents[j] = int index for j > 0
    seps    : list of int arrays — seps[j] = global node indices of separator
              between cliques[j] and its parent; seps[0] = empty array
    edges   : (rows, cols) — upper-tri entries of the chordal extension
    """
    cliques : list
    parents : list
    seps    : list
    edges   : tuple

    @property
    def clique_sizes(self):
        return [len(c) for c in self.cliques]


def get_elim_order_cholmod(rows, cols, d):
    """Fill-reducing elimination ordering via CHOLMOD AMD (symbolic analysis only).

    Parameters
    ----------
    rows, cols : (m,) int arrays — observed index pairs
    d          : int — matrix dimension

    Returns
    -------
    elim_order : (d,) int array — elim_order[step] = variable eliminated at that step
    """
    import scipy.sparse as sp
    from sksparse.amd import amd

    off = rows != cols
    r, c = rows[off].astype(int), cols[off].astype(int)
    A = sp.csc_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(d, d))
    return amd(A).astype(int)


def get_elim_order_nx(rows, cols, d):
    """Perfect elimination ordering via networkx MCS-M triangulation.

    Parameters
    ----------
    rows, cols : (m,) int arrays — observed index pairs
    d          : int — matrix dimension

    Returns
    -------
    elim_order : (d,) int array — elim_order[step] = variable eliminated at that step
    """
    G = nx.Graph()
    G.add_nodes_from(range(d))
    for r, c in zip(rows, cols):
        if r != c:
            G.add_edge(int(r), int(c))
    _, alpha = nx.complete_to_chordal_graph(G)
    # alpha[v] = position of v in the ordering (0 = eliminated first)
    return np.array(sorted(range(d), key=lambda v: alpha[v]), dtype=int)


def triangulate(rows, cols, d, elim_order):
    """Triangulate a graph by simulating variable elimination.

    Adds fill edges to make the graph chordal under the given ordering.

    Parameters
    ----------
    rows, cols : (m,) int arrays — observed index pairs
    d          : int — matrix dimension
    elim_order : (d,) int array — variable elimination order

    Returns
    -------
    H : nx.Graph — chordal graph (original edges + fill-in)
    """
    H = nx.Graph()
    H.add_nodes_from(range(d))
    H.add_edges_from((int(r), int(c)) for r, c in zip(rows, cols) if r != c)

    eliminated = np.zeros(d, dtype=bool)
    for v in elim_order:
        v = int(v)
        neighbors = [u for u in H.neighbors(v) if not eliminated[u]]
        for u, w in itertools.combinations(neighbors, 2):
            if not H.has_edge(u, w):
                H.add_edge(u, w)
        eliminated[v] = True

    return H


def build_jtree(rows, cols, d, elim_order=None):
    """Build a JunctionTree from a sparsity pattern.

    Triangulates the graph under the given elimination ordering, finds maximal
    cliques, and builds the junction tree via maximum spanning tree.

    Parameters
    ----------
    rows, cols : (m,) int arrays — observed index pairs
    d          : int — matrix dimension
    elim_order : (d,) int array — elimination ordering; default is CHOLMOD AMD

    Returns
    -------
    JunctionTree
    """
    if elim_order is None:
        elim_order = get_elim_order_cholmod(rows, cols, d)

    H       = triangulate(rows, cols, d, elim_order)
    cliques = [np.array(sorted(c), dtype=int) for c in nx.find_cliques(H)]
    K       = len(cliques)

    # Chordal extension edges: diagonal + upper-tri of H
    H.add_edges_from((i, i) for i in range(d))
    pairs = sorted((min(u,v), max(u,v)) for u, v in H.edges())
    rows, cols = zip(*pairs)
    edges = (np.array(rows, dtype=int), np.array(cols, dtype=int))

    if K == 1:
        return JunctionTree(
            cliques = cliques,
            parents = [None],
            seps    = [np.array([], dtype=int)],
            edges   = edges,
        )

    # Junction tree via maximum spanning tree of clique-intersection graph.
    # All K nodes are added explicitly so isolated cliques (variables with no
    # off-diagonal observations) are still included in the spanning tree.
    CG = nx.Graph()
    CG.add_nodes_from(range(K))
    for i in range(K):
        for j in range(i + 1, K):
            sep = np.intersect1d(cliques[i], cliques[j])
            if len(sep) > 0:
                CG.add_edge(i, j, weight=len(sep))
    # Connect any component not containing node 0 so the spanning tree covers all K cliques.
    # This handles variables with no off-diagonal observations (singleton cliques).
    for component in nx.connected_components(CG):
        if 0 not in component:
            CG.add_edge(0, min(component), weight=0)

    # Get BFS tree and reorder cliques in tree order
    T    = nx.bfs_tree(nx.maximum_spanning_tree(CG), 0)
    topo = list(nx.topological_sort(T))
    rank = {v: i for i, v in enumerate(topo)}

    cliques = [cliques[v] for v in topo]
    T       = nx.relabel_nodes(T, rank)   # now T's nodes are 0..K-1 in topo order

    parents = [None] * K
    seps    = [np.array([], dtype=int)] * K
    for j in range(1, K):
        parents[j] = next(T.predecessors(j))  # guaranteed < j by topo order
        seps[j]    = np.intersect1d(cliques[j], cliques[parents[j]])

    return JunctionTree(
        cliques = cliques,
        parents = parents,
        seps    = seps,
        edges   = edges,
    )


# ── Entropy ───────────────────────────────────────────────────────────────────

def unpack_y(y_flat, y_slices, clique_sizes):
    """Unpack flat y vector into a list of clique matrices."""
    return [unvec(y_flat[sl], s) for sl, s in zip(y_slices, clique_sizes)]


def log_det(jtree, y_mats):
    """Log-determinant of the max-entropy completion via the junction tree formula:

        log det X = Σ_j log det X_{β_j} - Σ_j log det X_{S_j}

    The j=0 separator is empty so its term vanishes.

    Parameters
    ----------
    jtree  : JunctionTree
    y_mats : list of (s_k, s_k) symmetric matrices, one per clique
    """
    ent = 0.0
    for j, Y_k in enumerate(y_mats):
        ent += np.sum(np.log(np.maximum(np.linalg.eigvalsh(Y_k), 1e-10)))
        sep = jtree.seps[j]
        if len(sep) > 0:
            parent = jtree.parents[j]
            local  = np.searchsorted(jtree.cliques[parent], sep)
            S      = y_mats[parent][np.ix_(local, local)]
            ent   -= np.sum(np.log(np.maximum(np.linalg.eigvalsh(S), 1e-10)))
    return float(ent)


# ── Solver geometry ───────────────────────────────────────────────────────────

def build_geometry(jtree, rows, cols, tau):
    """Precompute DR-solver index structures from a JunctionTree.

    Parameters
    ----------
    jtree      : JunctionTree
    rows, cols : (m,) int arrays — observed index pairs
    tau        : (m,) float array — noise std-devs

    Returns
    -------
    dict with fields:
        n             int — number of entries in x (len of chordal edge list)
        obs_inds      (m,) int — positions in x of observed entries
        clique_inds   list of (n_k,) int — positions in x of upper-tri entries of clique k
        n_covers      (n,) int — number of cliques covering each entry of x
        ny            int — total length of flat y (Σ_k s_k*(s_k+1)/2)
        y_slices      list of slice — slice into flat y for each clique
        y_gather_inds (ny,) int — x[y_gather_inds] gathers all clique entries into flat y
        w             (m,) float — normalized objective weights 1/τ²
    """
    edge_rows, edge_cols = jtree.edges
    cliques              = jtree.cliques
    n                    = len(edge_rows)

    obs_rows = np.minimum(rows, cols)
    obs_cols = np.maximum(rows, cols)

    edge2ind = {(r, c): i for i, (r, c) in enumerate(zip(edge_rows, edge_cols))}

    for i in range(edge_rows.max() + 1):
        if (i, i) not in edge2ind:
            raise ValueError(f"Missing diagonal entry ({i}, {i}) in chordal pattern")

    obs_inds = np.array([edge2ind[e] for e in zip(obs_rows, obs_cols)], dtype=int)

    clique_inds = []
    for beta in cliques:
        s = len(beta)
        r, c = np.triu_indices(s)
        clique_edges = [(beta[r[i]], beta[c[i]]) for i in range(len(r))]
        clique_inds.append(np.array([edge2ind[e] for e in clique_edges], dtype=int))

    n_covers = np.zeros(n, dtype=int)
    for cidx in clique_inds:
        n_covers[cidx] += 1

    y_slices = []
    offset   = 0
    for s in jtree.clique_sizes:
        nvec = s * (s + 1) // 2
        y_slices.append(slice(offset, offset + nvec))
        offset += nvec
    ny = offset
    y_gather_inds = np.concatenate(clique_inds)

    return dict(
        n             = n,
        obs_inds      = obs_inds,
        clique_inds   = clique_inds,
        n_covers      = n_covers,
        ny            = ny,
        y_slices      = y_slices,
        y_gather_inds = y_gather_inds,
        w             = _weights(tau),
    )


# ── Reconstruction ────────────────────────────────────────────────────────────

def maxent_completion(jtree, clique_mats):
    """Reconstruct the full d×d covariance from clique marginals via junction tree.

    Root-to-leaves pass using the conditional distribution of each clique given
    its separator.  When clique_mats are consistent and PD this recovers the
    max-entropy completion; otherwise returns a valid PSD matrix built from the
    conditional distributions implied by each clique.

    Parameters
    ----------
    jtree      : JunctionTree
    clique_mats: list of (s_k, s_k) symmetric PSD arrays, one per clique,
                 in the same order as jtree.cliques

    Returns
    -------
    Sigma : (d, d) ndarray — always PSD; PD when all clique_mats are PD
    """
    cliques = jtree.cliques
    parents = jtree.parents
    seps    = jtree.seps
    K       = len(cliques)
    d       = max(c.max() for c in cliques) + 1

    Sigma = np.zeros((d, d))
    b0    = cliques[0]
    Sigma[np.ix_(b0, b0)] = clique_mats[0]

    for ell in range(1, K):
        beta_l = cliques[ell]
        S      = seps[ell]
        X      = clique_mats[ell]

        if len(S) == 0:
            # Disconnected component: independent of everything seen so far
            Sigma[np.ix_(beta_l, beta_l)] = X
            continue

        # Local indices of S and R = beta_l \ S within beta_l
        s_loc  = np.searchsorted(beta_l, S)
        r_mask = np.ones(len(beta_l), dtype=bool)
        r_mask[s_loc] = False
        r_loc  = np.where(r_mask)[0]
        R      = beta_l[r_loc]                       # global indices of residual

        # Conditional distribution of x_R | x_S derived from X
        X_SS = X[np.ix_(s_loc, s_loc)]
        X_RS = X[np.ix_(r_loc, s_loc)]
        X_RR = X[np.ix_(r_loc, r_loc)]
        A    = X_RS @ np.linalg.pinv(X_SS)           # |R| × |S| regression coefs
        C    = X_RR - A @ X_SS @ A.T                 # |R| × |R| conditional cov, PSD

        # Cross-blocks: Sigma[beta_j, R] = Sigma[beta_j, S] @ A.T  for all j < ell
        for j in range(ell):
            bj       = cliques[j]
            Sigma_jS = Sigma[np.ix_(bj, S)]
            Sigma_jR = Sigma_jS @ A.T
            Sigma[np.ix_(bj, R)] = Sigma_jR
            Sigma[np.ix_(R, bj)] = Sigma_jR.T

        # Diagonal block: S part already in matrix from parent; fill R rows/cols
        Sigma_SS = Sigma[np.ix_(S, S)]
        Sigma_RS = A @ Sigma_SS
        Sigma_RR = C + A @ Sigma_SS @ A.T

        Sigma[np.ix_(R, S)] = Sigma_RS
        Sigma[np.ix_(S, R)] = Sigma_RS.T
        Sigma[np.ix_(R, R)] = Sigma_RR

    return Sigma


def maxent_completion_dense(jtree, clique_mats):
    """Dense reference implementation of maxent_completion.

    Assembles the precision matrix K via the junction tree formula
    K = Σ_k K_k - Σ_k K_{S_k}, then inverts.  Requires consistent PD clique
    marginals.  Useful for testing against maxent_completion.

    Parameters
    ----------
    jtree      : JunctionTree
    clique_mats: list of (s_k, s_k) symmetric PD arrays, one per clique

    Returns
    -------
    Sigma : (d, d) ndarray
    """
    cliques = jtree.cliques
    parents = jtree.parents
    seps    = jtree.seps
    d       = max(c.max() for c in cliques) + 1

    K = np.zeros((d, d))
    for ell, X in enumerate(clique_mats):
        b = cliques[ell]
        K[np.ix_(b, b)] += np.linalg.inv(X)
        S = seps[ell]
        if len(S) > 0:
            pa      = parents[ell]
            s_in_pa = np.searchsorted(cliques[pa], S)
            X_SS    = clique_mats[pa][np.ix_(s_in_pa, s_in_pa)]
            K[np.ix_(S, S)] -= np.linalg.inv(X_SS)

    return np.linalg.inv(K)


# ── Solver ────────────────────────────────────────────────────────────────────

def dr_splitting(A, inds, tau,
                 gamma=1.0, rho=1.0,
                 max_iters=5000, eps_abs=1e-5,
                 diag_every=25,
                 elim_order=None):
    """Douglas-Rachford splitting for the sparse PSD-nearness problem.

    Parameters
    ----------
    A          : (d, d) ndarray — noisy observations (symmetric; only E entries used)
    inds       : (m, 2) int array — observed index pairs
    tau        : (m,) float array — noise std-devs
    gamma      : float — proximal step size (> 0)
    rho        : float — D-R relaxation parameter (0 < rho < 2)
    max_iters  : int — maximum iterations
    eps_abs    : float — convergence tolerance on normalized primal/dual residuals
    diag_every : int — compute and record diagnostics every this many iterations
    elim_order : (d,) int array — elimination ordering; default is CHOLMOD AMD

    Returns
    -------
    Sigma_hat : (d, d) ndarray — estimated covariance (max-entropy completion)
    hist      : list of per-iteration dicts with keys:
                    time, obj, entropy, r_primal, r_dual
    """
    d              = A.shape[0]
    obs_rows, obs_cols = inds[:, 0], inds[:, 1]
    jtree = build_jtree(obs_rows, obs_cols, d, elim_order)
    geom  = build_geometry(jtree, obs_rows, obs_cols, tau)

    cliques        = jtree.cliques
    clique_sizes   = jtree.clique_sizes
    n_cliques      = len(cliques)

    n             = geom['n']
    obs_inds      = geom['obs_inds']
    clique_inds   = geom['clique_inds']
    n_covers      = geom['n_covers']
    y_slices      = geom['y_slices']
    y_gather_inds = geom['y_gather_inds']
    w             = geom['w']

    # vec scales off-diagonal entries by √2 (so ‖vec(Y)‖² = ‖Y‖_F²).  Halving w
    # for off-diagonal entries compensates, so np.sum(w * r_vec²) == outer objective.
    a = vec(A, obs_rows, obs_cols)
    w = w.copy()
    w[obs_rows != obs_cols] /= 2.0

    # ── Initialization ────────────────────────────────────────────────────────
    x = np.zeros(n)
    x[obs_inds] = a
    y = x[y_gather_inds].copy()

    u = x.copy()
    v = y.copy()

    # Precompute index sets for prox_psd
    _thresh         = 1e-8
    size_one_inds   = np.array([y_slices[k].start for k in range(n_cliques)
                                if clique_sizes[k] == 1], dtype=int)
    large_clique_ks = [k for k in range(n_cliques) if clique_sizes[k] > 1]

    hist = []
    t0   = time.perf_counter()

    def prox_psd(y):
        """Project all clique submatrices of flat y onto the PSD cone."""
        y_new = y.copy()
        y_new[size_one_inds] = np.maximum(y[size_one_inds], _thresh)
        for k in large_clique_ks:
            y_new[y_slices[k]] = vec(project_psd(unvec(y[y_slices[k]], clique_sizes[k])))
        return y_new

    def project_V(x, y):
        """Project (x, y) onto V = {(x,y) : y_k = P_{β_k} x}.

        Mutates x in-place. Returns projected (x, y) where y is a new array.
        Pass x.copy() if the original is needed afterward.
        """
        np.add.at(x, y_gather_inds, y)
        x /= (1.0 + n_covers)
        return x, x[y_gather_inds]

    def norm_xy(x, y):
        return np.hypot(np.linalg.norm(x), np.linalg.norm(y))

    for it in range(max_iters):
        # ── Step 1a: prox of weighted quadratic  (x update) ──────────────────
        x_new = u.copy()
        x_new[obs_inds] = (u[obs_inds] + gamma * w * a) / (1.0 + gamma * w)

        # ── Step 1b: prox of PSD indicator (y update) ────────────────────────
        y_new = prox_psd(v)

        # ── Step 2a: reflect (x, y) across their prox mappings ───────────────
        p = 2.0 * x_new - u
        q = 2.0 * y_new - v

        # ── Step 2b: project (p, q) onto V ───────────────────────────────────
        p, q = project_V(p, q)

        # ── s = γ⁻¹(z̃ - x̃): needs old u, v before dual update ──────────────
        do_diag = (it + 1) % diag_every == 0
        if do_diag:
            s_x = (u - x_new) / gamma
            s_y = (v - y_new) / gamma

        # ── Step 3: dual update ───────────────────────────────────────────────
        u += rho * (p - x_new)
        v += rho * (q - y_new)

        x = x_new
        y = y_new

        # ── Diagnostics and convergence check ────────────────────────────────
        if do_diag:
            # Primal: r_p = Π_V(x̃) - x̃,  scale by max{‖x̃‖, 1}
            x_proj, y_proj = project_V(x_new.copy(), y_new)
            r_primal = norm_xy(x_proj - x_new, y_proj - y_new)
            scale_p  = max(norm_xy(x_new, y_new), 1.0)

            # Dual: r_d = -Π_V(s̃),  scale by max{‖s̃‖, 1}
            s_x_proj, s_y_proj = project_V(s_x.copy(), s_y)
            r_dual  = norm_xy(s_x_proj, s_y_proj)
            scale_d = max(norm_xy(s_x, s_y), 1.0)

            obj     = float(np.sum(w * (p[obs_inds] - a) ** 2))
            entropy = log_det(jtree, unpack_y(q, y_slices, clique_sizes))
            hist.append({
                'time':     time.perf_counter() - t0,
                'obj':      obj,
                'entropy':  entropy,
                'r_primal': r_primal / scale_p,
                'r_dual':   r_dual   / scale_d,
            })

            if (r_primal / scale_p < eps_abs) and (r_dual / scale_d < eps_abs):
                break

    clique_mats = unpack_y(y, y_slices, clique_sizes)
    Sigma_hat   = maxent_completion(jtree, clique_mats)
    return Sigma_hat, hist
