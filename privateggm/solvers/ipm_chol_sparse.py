"""ipm_chol_sparse — IPM with sparse Cholesky parameterization via junction tree.

Optimization variable: (A_k, L_{U_k,U_k}) for each clique k, where

    A_k          : (|U_k|, |S_k|) — regression coefficients (unconstrained)
    L_{U_k,U_k} : (|U_k|, |U_k|) lower-tri — conditional Cholesky (diag > 0)

These define the covariance via the junction tree recurrence (root → leaves):

    X_{S_k,S_k}  = (already computed from parent)
    X_{U_k,U_k}  = L_{U_k,U_k} L_{U_k,U_k}^T + A_k X_{S_k,S_k} A_k^T
    X_{U_k,S_k}  = A_k X_{S_k,S_k}

Log-det identity (follows from Schur complement + junction tree formula):

    log|X| = 2 Σ_k Σ_{i ∈ U_k} log L_{U_k,U_k}[i,i]

Backward pass (leaves → root): only propagates X̄_{S_k,S_k} to parent — no
Cholesky-factor propagation needed, avoiding the recursive ancestor walk.

Reference: notes/Efficient Cholesky of Covariance.md
"""

import time

import numpy as np
from scipy.optimize import minimize

from ._shared import _weights, normalize_coo
from ._jtree import build_jtree, obs_routing


# ── Precomputation helpers ────────────────────────────────────────────────────

def _build_var_info(jtree):
    """Precompute variable layout, bounds, and tril indices for each clique.

    For each clique k, variables are packed as:
        [A_k.ravel()         — |U_k| * |S_k| entries (unconstrained)
         tril(L_{U_k,U_k})  — |U_k|*(|U_k|+1)//2 entries (diagonals > 0)]

    Returns
    -------
    slices         : list of slice — slice into z for clique k's block
    diag_inds      : (ndiag,) int — positions in z of diagonal L_{U_k,U_k}[i,i]
    n_vars         : int — total length of z
    bounds         : list of (lo, hi) — one per variable
    tril_inds_list : list of (tril_r, tril_c) per clique
    n_A_list       : list of int — number of A entries per clique
    """
    us   = jtree.us
    seps = jtree.seps
    K    = len(jtree.cliques)

    slices         = []
    diag_inds      = []
    bounds         = []
    tril_inds_list = []
    n_A_list       = []
    offset         = 0

    for k in range(K):
        nU             = len(us[k])
        nS             = len(seps[k])
        n_A            = nU * nS
        tril_r, tril_c = np.tril_indices(nU)
        n_k            = n_A + len(tril_r)

        slices.append(slice(offset, offset + n_k))
        tril_inds_list.append((tril_r, tril_c))
        n_A_list.append(n_A)
        bounds.extend([(None, None)] * n_A)

        for i, (r, c) in enumerate(zip(tril_r, tril_c)):
            if r == c:
                diag_inds.append(offset + n_A + i)
                bounds.append((1e-8, None))
            else:
                bounds.append((None, None))

        offset += n_k

    return slices, np.array(diag_inds, dtype=int), offset, bounds, tril_inds_list, n_A_list


def _build_clique_perms(jtree):
    """Precompute ix_ index arrays for the forward/backward passes.

    X[k] is stored in sorted cliques[k] order.  The covariance recurrence
    assembles blocks in [S_k | U_k] order, so we need permutations to convert.

    Returns
    -------
    _SS     : list of open meshgrid (s_in_k, s_in_k) — for X[k][S,S]
    _UU     : list of open meshgrid (u_in_k, u_in_k) — for X[k][U,U]
    _US     : list of open meshgrid (u_in_k, s_in_k) — for X[k][U,S]
    _SU     : list of open meshgrid (s_in_k, u_in_k) — for X[k][S,U]
    _SS_par : list of open meshgrid (s_in_pa, s_in_pa) — for X[pa][S,S]
              (None for root)
    """
    cliques = jtree.cliques
    seps    = jtree.seps
    us      = jtree.us
    parents = jtree.parents

    SS     = []
    UU     = []
    US     = []
    SU     = []
    SS_par = []

    for k in range(len(cliques)):
        s_in_k = np.searchsorted(cliques[k], seps[k])
        u_in_k = np.searchsorted(cliques[k], us[k])
        SS.append(np.ix_(s_in_k, s_in_k))
        UU.append(np.ix_(u_in_k, u_in_k))
        US.append(np.ix_(u_in_k, s_in_k))
        SU.append(np.ix_(s_in_k, u_in_k))

        pa = parents[k]
        if pa is None:
            SS_par.append(None)
        else:
            spa = np.searchsorted(cliques[pa], seps[k])
            SS_par.append(np.ix_(spa, spa))

    return SS, UU, US, SU, SS_par


# ── Model class ───────────────────────────────────────────────────────────────

class SparseCholeskyModel:
    """Junction-tree parameterization of Cholesky factor of covariance matrix with sparse inverse.

    Precomputes all index arrays at construction time.  The forward/backward
    passes and the barrier objective only receive the optimization variables
    that change each iteration.

    Parameters
    ----------
    jtree     : JunctionTree
    Sigma_obs : (d, d) ndarray — noisy observations
    inds      : (m, 2) int array — observed index pairs (upper-tri including diagonal)
    tau       : (m,) float array — noise std-devs
    """

    def __init__(self, jtree, vals, inds, tau):
        obs_rows = inds[:, 0]
        obs_cols = inds[:, 1]

        self.jtree  = jtree
        self.w      = _weights(tau)
        self._a_obs = vals
        self.K      = len(jtree.cliques)

        # Variable layout
        (self._slices, self.diag_inds, self.n_vars,
         self.bounds, self._tril_inds_list, self._n_A_list) = _build_var_info(jtree)

        # Per-clique sizes (avoids repeated len(jtree.us[k]) in hot paths)
        self._nU_list = [len(jtree.us[k])   for k in range(self.K)]
        self._nS_list = [len(jtree.seps[k]) for k in range(self.K)]

        # Clique index arrays for forward/backward
        self._SS, self._UU, self._US, self._SU, self._SS_par = \
            _build_clique_perms(jtree)

        # Observation routing
        clique_idx, local_rows, local_cols = obs_routing(jtree, obs_rows, obs_cols)
        self._obs_lr        = local_rows
        self._obs_lc        = local_cols
        self._obs_by_clique = [np.where(clique_idx == k)[0] for k in range(self.K)]

        # Preallocated gradient buffers (reused every phi_and_grad call)
        csizes          = jtree.clique_sizes
        self._X_bar_buf = [np.zeros((csizes[k], csizes[k])) for k in range(self.K)]
        self._A_bar_buf = [np.zeros((nU, nS))
                           for nU, nS in zip(self._nU_list, self._nS_list)]
        self._L_bar_buf = [np.zeros((nU, nU)) for nU in self._nU_list]

    # ── Vec / unvec ───────────────────────────────────────────────────────────

    def vec(self, A, L):
        """Pack (A_k, L_{U_k,U_k}) lists into flat vector z."""
        z = np.empty(self.n_vars)
        for k, sl in enumerate(self._slices):
            n_A            = self._n_A_list[k]
            tril_r, tril_c = self._tril_inds_list[k]
            chunk          = z[sl]
            if n_A > 0:
                chunk[:n_A] = A[k].ravel()
            chunk[n_A:] = L[k][tril_r, tril_c]
        return z

    def unvec(self, z):
        """Unpack flat vector z into (A_k, L_{U_k,U_k}) lists."""
        A = []
        L = []
        for k, sl in enumerate(self._slices):
            nU             = self._nU_list[k]
            nS             = self._nS_list[k]
            n_A            = self._n_A_list[k]
            tril_r, tril_c = self._tril_inds_list[k]
            chunk          = z[sl]

            A_k                  = chunk[:n_A].reshape(nU, nS) if nS > 0 else np.zeros((nU, 0))
            L_k                  = np.zeros((nU, nU))
            L_k[tril_r, tril_c] = chunk[n_A:]

            A.append(A_k)
            L.append(L_k)

        return A, L

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, A, L):
        """Root-to-leaves covariance recurrence.

        Returns X[k] in sorted cliques[k] order, as required by
        maxent_completion and the separator index reads in the backward pass.
        """
        parents = self.jtree.parents
        K       = self.K

        X = [None] * K

        for k in range(K):
            nS  = self._nS_list[k]
            L_k = L[k]

            if nS == 0:
                X[k] = L_k @ L_k.T
            else:
                nU   = self._nU_list[k]
                A_k  = A[k]
                X_SS = X[parents[k]][self._SS_par[k]]
                X_UU = L_k @ L_k.T + A_k @ X_SS @ A_k.T
                X_US = A_k @ X_SS

                X_CC = np.empty((nU + nS, nU + nS))   # all four blocks filled below
                X_CC[self._SS[k]] = X_SS
                X_CC[self._UU[k]] = X_UU
                X_CC[self._US[k]] = X_US
                X_CC[self._SU[k]] = X_US.T
                X[k] = X_CC

        return X

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self, A, L, X, X_bar):
        """Leaves-to-root gradient backprop.

        X_bar is modified in-place during the pass (separator gradients
        are accumulated into parent blocks).
        """
        parents = self.jtree.parents

        for buf in self._A_bar_buf: buf.fill(0.0)
        for buf in self._L_bar_buf: buf.fill(0.0)

        for k in reversed(range(self.K)):
            nS  = self._nS_list[k]
            A_k = A[k]
            L_k = L[k]

            # X[k][U,S] and X[k][S,U] are separate nodes in the forward graph,
            # so XbUS and XbSU must be read independently (not symmetrized).
            XbUU = X_bar[k][self._UU[k]]
            XbUS = X_bar[k][self._US[k]]
            XbSU = X_bar[k][self._SU[k]]

            # L̄: d/dL [ tr((XbUU + XbUU^T)(L L^T)) / 2 ] = (XbUU + XbUU^T) L
            self._L_bar_buf[k] += (XbUU + XbUU.T) @ L_k

            if nS > 0:
                pa   = parents[k]
                X_SS = X[pa][self._SS_par[k]]

                self._A_bar_buf[k] += (XbUU + XbUU.T) @ A_k @ X_SS \
                                      + XbUS @ X_SS + XbSU.T @ X_SS

                # Propagate X̄_{S_k,S_k} to parent
                XbSS = X_bar[k][self._SS[k]]
                X_bar[pa][self._SS_par[k]] += \
                    A_k.T @ XbUU @ A_k + A_k.T @ XbUS + (XbSU @ A_k).T + XbSS

        return self._A_bar_buf, self._L_bar_buf

    # ── Gather / objective / gradient ─────────────────────────────────────────

    def gather_obs(self, X):
        """Gather observed entries from clique marginals into a flat (m,) array."""
        X_obs = np.empty(len(self._a_obs))
        for k, idx in enumerate(self._obs_by_clique):
            if len(idx):
                X_obs[idx] = X[k][self._obs_lr[idx], self._obs_lc[idx]]
        return X_obs

    def _eval(self, z):
        """Forward-only evaluation: returns (fval, logdet, X) at z."""
        A, L     = self.unvec(z)
        X        = self.forward(A, L)
        r_vec    = self.gather_obs(X) - self._a_obs
        fval     = float(np.dot(self.w, r_vec ** 2))
        logdet   = 2.0 * float(np.sum(np.log(z[self.diag_inds])))
        return fval, logdet, X

    def phi_and_grad(self, z, mu):
        """Barrier objective φ = f(X_E) − μ·log|X| and its gradient w.r.t. z."""
        A, L   = self.unvec(z)
        X      = self.forward(A, L)

        r_vec  = self.gather_obs(X) - self._a_obs
        fval   = float(np.dot(self.w, r_vec ** 2))
        logdet = 2.0 * float(np.sum(np.log(z[self.diag_inds])))

        # Seed X̄ from ∂f/∂X_{ij} = 2 w_{ij} r_{ij}
        seed = 2.0 * self.w * r_vec
        for k in range(self.K):
            self._X_bar_buf[k].fill(0.0)
            idx = self._obs_by_clique[k]
            if len(idx):
                np.add.at(self._X_bar_buf[k],
                          (self._obs_lr[idx], self._obs_lc[idx]), seed[idx])

        A_bar, L_bar = self.backward(A, L, X, self._X_bar_buf)
        grad_z = self.vec(A_bar, L_bar)

        # Log-det gradient: −2μ / L_{U_k,U_k}[i,i]
        grad_z[self.diag_inds] -= 2.0 * mu / z[self.diag_inds]

        return float(fval - mu * logdet), grad_z


# ── Recovery ─────────────────────────────────────────────────────────────────

def _recover_covariance(jtree, A, L):
    """Assemble the full dxd covariance from junction-tree Cholesky parameters.

    Expanded forward pass using A_k and L_k directly — no pseudoinverse needed.
    Cross-clique blocks are filled via Sigma[prev, U_k] = Sigma[prev, S_k] @ A_k.T,
    which follows from S_k separating U_k from all earlier cliques in the tree.

    Parameters
    ----------
    jtree : JunctionTree
    A     : list of (|U_k|, |S_k|) arrays — regression coefficients
    L     : list of (|U_k|, |U_k|) lower-tri arrays — conditional Cholesky factors

    Returns
    -------
    X : (d, d) ndarray
    """
    cliques = jtree.cliques
    seps    = jtree.seps
    us      = jtree.us
    K       = len(cliques)
    d       = max(c.max() for c in cliques) + 1

    X = np.zeros((d, d))
    b0    = cliques[0]
    X[np.ix_(b0, b0)] = L[0] @ L[0].T

    for k in range(1, K):
        U_k = us[k]
        S_k = seps[k]
        A_k = A[k]
        L_k = L[k]

        # Cross-blocks for all previously seen variables
        for j in range(k):
            bj       = cliques[j]
            X_jU = X[np.ix_(bj, S_k)] @ A_k.T
            X[np.ix_(bj, U_k)] = X_jU
            X[np.ix_(U_k, bj)] = X_jU.T

        # Diagonal block: separator part already filled; add unique rows/cols
        X_SS = X[np.ix_(S_k, S_k)]
        X_SU = X_SS @ A_k.T
        X[np.ix_(S_k, U_k)] = X_SU
        X[np.ix_(U_k, S_k)] = X_SU.T
        X[np.ix_(U_k, U_k)] = L_k @ L_k.T + A_k @ X_SS @ A_k.T

    return X


# ── Solver ────────────────────────────────────────────────────────────────────

def ipm_chol_sparse(rows, cols, vals, tau, d,
                    mu_init=1.0, mu_factor=0.5, eps=1e-8, eps_coarse=1e-2,
                    max_outer=40, max_inner=200,
                    elim_order=None):
    """IPM with sparse Cholesky parameterization via junction tree.

    Exploits the chordal sparsity structure: only O(|E'|) variables are stored
    and the forward/backward passes are O(Σ_k |C_k|³) per iteration.

    Parameters
    ----------
    rows, cols : (m,) int arrays — observed index pairs (any triangular format)
    vals       : (m,) float array — noisy observed values
    tau        : (m,) float array — noise std-devs
    d          : int — matrix dimension
    mu_init    : initial barrier weight
    mu_factor  : geometric decrease factor for μ (< 1)
    eps        : termination: stop outer loop when μ ≤ eps
    eps_coarse : coarse inner L-BFGS tolerance (early outer levels)
    max_outer  : max outer (μ-reduction) iterations
    max_inner  : max inner L-BFGS steps per outer iteration
    elim_order : (d,) int array — elimination ordering; default is CHOLMOD AMD

    Returns
    -------
    X_hat : (d, d) ndarray — estimated covariance
    hist  : list of per-outer-iteration dicts with keys:
                time, obj, entropy, mu, inner_iters
            Note: KKT diagnostic fields (min_eig_G, comp_slack) present in
            ipm_chol are omitted here because the sparse model does not
            assemble the full d×d gradient matrix G.
    """
    rows, cols, vals, tau = normalize_coo(rows, cols, vals, tau)
    obs_rows = rows
    obs_cols = cols
    inds     = np.stack([rows, cols], axis=1)

    jtree = build_jtree(obs_rows, obs_cols, d, elim_order)
    model = SparseCholeskyModel(jtree, vals, inds, tau)

    K = model.K

    # Initialise at X = αI  →  A_k = 0, L_{U_k,U_k} = √α I
    alpha      = max(float(np.mean(np.abs(vals[rows == cols]))), 1.0)
    sqrt_alpha = np.sqrt(alpha)
    A = [np.zeros((model._nU_list[k], model._nS_list[k])) for k in range(K)]
    L = [sqrt_alpha * np.eye(model._nU_list[k])            for k in range(K)]

    z  = model.vec(A, L)
    mu = float(mu_init)

    hist = []
    t0   = time.perf_counter()

    for _outer in range(max_outer):
        eps_inner = min(mu, eps_coarse)
        result    = minimize(model.phi_and_grad, z, method='L-BFGS-B', jac=True,
                             args=(mu,), bounds=model.bounds,
                             options={'maxiter': max_inner,
                                      'ftol':    eps_inner,
                                      'gtol':    eps_inner})
        z = result.x

        fval, logdet, _ = model._eval(z)
        hist.append({
            'time':        time.perf_counter() - t0,
            'obj':         fval,
            'entropy':     logdet,
            'mu':          mu,
            'inner_iters': result.nit,
        })

        if mu <= eps:
            break
        mu *= mu_factor

    A, L = model.unvec(z)
    return _recover_covariance(jtree, A, L), hist
