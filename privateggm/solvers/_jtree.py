"""_jtree — Junction tree construction for chordal graphs.

Shared infrastructure used by dr_splitting and ipm_chol_sparse.
"""

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class JunctionTree:
    """Junction tree of a chordal graph.

    Cliques are stored in BFS traversal order from the root (cliques[0]).
    This guarantees parents[j] < j for all j > 0, so a root-to-leaves pass
    is forward iteration and a leaves-to-root pass is reverse iteration.

    Fields
    ------
    cliques : list of sorted int arrays - maximal cliques, traversal order
    parents : list - parents[0] = None; parents[j] = int index for j > 0
    seps    : list of int arrays - seps[j] = global node indices of separator
              between cliques[j] and its parent; seps[0] = empty array
    us      : list of int arrays - us[j] = global node indices of new nodes
              introduced in clique j (us[j] = cliques[j] \\ seps[j])
    edges   : (rows, cols) - upper-tri entries of the chordal extension
    """
    cliques : list
    parents : list
    seps    : list
    us      : list
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


def obs_routing(jtree, obs_rows, obs_cols):
    """Map each observed (i,j) to a clique containing both i and j.

    Parameters
    ----------
    jtree              : JunctionTree
    obs_rows, obs_cols : (m,) int arrays — observed index pairs

    Returns
    -------
    clique_idx : (m,) int — clique index for each observation
    local_rows : (m,) int — local row index within that clique
    local_cols : (m,) int — local col index within that clique
    """
    cliques   = jtree.cliques
    d         = max(c.max() for c in cliques) + 1
    node2cliq = [[] for _ in range(d)]
    for k, c in enumerate(cliques):
        for node in c:
            node2cliq[node].append(k)

    m          = len(obs_rows)
    clique_idx = np.empty(m, dtype=int)
    local_rows = np.empty(m, dtype=int)
    local_cols = np.empty(m, dtype=int)

    for idx in range(m):
        r, c = int(obs_rows[idx]), int(obs_cols[idx])
        for k in node2cliq[r]:
            clq = cliques[k]
            lr  = np.searchsorted(clq, r)
            lc  = np.searchsorted(clq, c)
            if lr < len(clq) and clq[lr] == r and lc < len(clq) and clq[lc] == c:
                clique_idx[idx] = k
                local_rows[idx] = lr
                local_cols[idx] = lc
                break

    return clique_idx, local_rows, local_cols


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
    rows_out, cols_out = zip(*pairs)
    edges = (np.array(rows_out, dtype=int), np.array(cols_out, dtype=int))

    if K == 1:
        return JunctionTree(
            cliques = cliques,
            parents = [None],
            seps    = [np.array([], dtype=int)],
            us      = [cliques[0].copy()],
            edges   = edges,
        )

    # Junction tree via maximum spanning tree of clique-intersection graph.
    CG = nx.Graph()
    CG.add_nodes_from(range(K))
    for i in range(K):
        for j in range(i + 1, K):
            sep = np.intersect1d(cliques[i], cliques[j])
            if len(sep) > 0:
                CG.add_edge(i, j, weight=len(sep))
    for component in nx.connected_components(CG):
        if 0 not in component:
            CG.add_edge(0, min(component), weight=0)

    T    = nx.bfs_tree(nx.maximum_spanning_tree(CG), 0)
    topo = list(nx.topological_sort(T))
    rank = {v: i for i, v in enumerate(topo)}

    cliques = [cliques[v] for v in topo]
    T       = nx.relabel_nodes(T, rank)

    parents = [None] * K
    seps    = [np.array([], dtype=int)] * K
    us      = [None] * K
    us[0]   = cliques[0].copy()
    for j in range(1, K):
        parents[j] = next(T.predecessors(j))
        seps[j]    = np.intersect1d(cliques[j], cliques[parents[j]])
        us[j]      = np.setdiff1d(cliques[j], seps[j])

    return JunctionTree(
        cliques = cliques,
        parents = parents,
        seps    = seps,
        us      = us,
        edges   = edges,
    )
