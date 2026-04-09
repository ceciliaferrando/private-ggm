"""privateggm.solve — top-level dispatch for covariance estimation.

Main entry point for solving the private GGM problem.  Handles connected-
component decomposition before dispatching to a specific solver.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .solvers._shared import normalize_coo
from .solvers.ipm_chol import ipm_chol
from .solvers.ipm_chol_sparse import ipm_chol_sparse
from .solvers.pgd_maxent import pgd, pgd_maxent
from .solvers.dr_splitting import dr_splitting

_SOLVER_REGISTRY = {
    'ipm_chol': ipm_chol,
    'ipm_chol_sparse': ipm_chol_sparse,
    'pgd': pgd,
    'pgd_maxent': pgd_maxent,
    'dr_splitting': dr_splitting,
}

try:
    from .solvers.clarabel import clarabel_solver
    _SOLVER_REGISTRY['clarabel'] = clarabel_solver
except ImportError:
    pass


def _get_solver_fn(solver):
    if callable(solver):
        return solver
    if solver not in _SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver {solver!r}. Available: {sorted(_SOLVER_REGISTRY)}"
        )
    return _SOLVER_REGISTRY[solver]


def _solve_block(rows, cols, vals, tau, d, solver_fn, **kwargs):
    """Call solver directly with sparse inputs."""
    return solver_fn(rows, cols, vals, tau, d, **kwargs)


def solve(rows, cols, vals, tau, d,
          solver='ipm_chol',
          decompose='blocks',
          min_diag=1e-10,
          verbose=False,
          **kwargs):
    """Solve the private GGM covariance estimation problem.

    Parameters
    ----------
    rows, cols : array-like, shape (m,)
        Indices of observed entries.  Any triangular format is accepted
        (upper, lower, or mixed); symmetric pairs and duplicates are combined
        via inverse-variance weighting and normalized to upper-triangle
        (rows <= cols) before solving.
    vals : array-like, shape (m,)
        Noisy observed values at (rows[k], cols[k]).
    tau : array-like, shape (m,)
        Noise scales, same length as vals.
    d : int
        Matrix dimension.
    solver : str or callable
        Solver to use.  String names: 'ipm_chol' (default), 'pgd',
        'pgd_maxent', 'ipm_chol_sparse', 'dr_splitting'.  Or pass any callable
        with signature fn(rows, cols, vals, tau, d, **kwargs) -> (Sigma, hist).
    decompose : None | 'diag' | 'blocks'
        Sparsity decomposition strategy:
        - None    — solve the full d×d problem directly.
        - 'diag'  — handle nodes with no off-diagonal observations as
                    independent scalars (max(val, min_diag)); solve the
                    remaining nodes as one block.
        - 'blocks'— split into all connected components; solve each
                    independently and assemble the block-diagonal result.
    min_diag : float
        Minimum variance assigned to isolated diagonal entries (default 1e-10).
    verbose : bool
        If True, print a summary of the decomposition (block sizes, singletons).
    **kwargs
        Forwarded verbatim to the solver.

    Returns
    -------
    Sigma : ndarray, shape (d, d)
        Estimated covariance matrix.
    hist : list of dicts or None
        Convergence history from the largest non-singleton component, or None
        if all components were handled as scalars.
    """
    rows, cols, vals, tau = normalize_coo(rows, cols, vals, tau)
    solver_fn = _get_solver_fn(solver)

    if decompose is None:
        if verbose:
            print(f'decompose=None: solving full {d}×{d} problem')
        return _solve_block(rows, cols, vals, tau, d, solver_fn, **kwargs)

    # ------------------------------------------------------------------
    # Build connected components from off-diagonal entries only
    # ------------------------------------------------------------------
    off_diag = rows != cols
    r_od, c_od = rows[off_diag], cols[off_diag]

    if len(r_od) == 0:
        # Pure diagonal — every node is a singleton
        if verbose:
            print(f'decompose={decompose!r}: all {d} nodes are singletons (pure diagonal)')
        Sigma = np.zeros((d, d))
        diag_mask = rows == cols
        idx = rows[diag_mask]
        Sigma[idx, idx] = np.maximum(vals[diag_mask], min_diag)
        return Sigma, None

    # Symmetric sparse graph (upper-triangle off-diag entries → symmetric)
    graph = csr_matrix(
        (np.ones(2 * len(r_od)),
         (np.concatenate([r_od, c_od]), np.concatenate([c_od, r_od]))),
        shape=(d, d),
    )
    _, labels = connected_components(graph, directed=False)
    comp_sizes = np.bincount(labels)
    is_singleton = comp_sizes[labels] == 1   # length-d boolean: is node i a singleton?

    Sigma = np.zeros((d, d))
    hist = None

    # ------------------------------------------------------------------
    # Singletons: vectorized diagonal fill
    # ------------------------------------------------------------------
    diag_mask = rows == cols
    sing_diag = diag_mask & is_singleton[rows]
    if sing_diag.any():
        idx = rows[sing_diag]
        Sigma[idx, idx] = np.maximum(vals[sing_diag], min_diag)

    # ------------------------------------------------------------------
    # Non-singleton blocks
    # ------------------------------------------------------------------
    non_sing_nodes = np.where(~is_singleton)[0]
    if len(non_sing_nodes) == 0:
        return Sigma, None

    if decompose == 'diag':
        blocks = [non_sing_nodes]
    else:  # 'blocks'
        unique_comps = np.unique(labels[non_sing_nodes])
        blocks = [np.where(labels == c)[0] for c in unique_comps]

    if verbose:
        n_sing = int(is_singleton.sum())
        block_sizes = sorted([len(b) for b in blocks], reverse=True)
        if decompose == 'diag':
            print(f'decompose={decompose!r}: {n_sing} singletons, '
                  f'1 block of size {block_sizes[0]}')
        else:
            sizes_str = ', '.join(str(s) for s in block_sizes)
            print(f'decompose={decompose!r}: {n_sing} singletons, '
                  f'{len(blocks)} blocks [sizes: {sizes_str}]')

    # Solve each block; track hist from the largest one
    largest_size = 0
    g2l = np.empty(d, dtype=int)   # reused scratch: global→local index map

    for nodes in blocks:
        sub_d = len(nodes)
        g2l[nodes] = np.arange(sub_d)

        # Keep entries whose row index is in this block.  For upper-triangle
        # inputs: if rows[k] ∈ nodes then cols[k] is either the same node
        # (diagonal) or a larger-index node in the same component.
        in_block = np.isin(rows, nodes)
        sub_rows = g2l[rows[in_block]]
        sub_cols = g2l[cols[in_block]]
        sub_vals = vals[in_block]
        sub_tau = tau[in_block]

        sub_Sigma, sub_hist = _solve_block(
            sub_rows, sub_cols, sub_vals, sub_tau, sub_d, solver_fn, **kwargs
        )

        # Place result into global Sigma
        ix = np.ix_(nodes, nodes)
        Sigma[ix] = sub_Sigma

        if sub_d > largest_size:
            largest_size = sub_d
            hist = sub_hist

    return Sigma, hist
