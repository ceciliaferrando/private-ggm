"""Experimental solvers — not production-ready, kept for research.

ipm_chol_kspace  IPM with K = LL^T Cholesky parameterization (precision space).
                 Dense-L baseline for a future sparse Takahashi implementation.
                 Converges at low noise; ill-conditioned near the PSD boundary
                 (med/high noise).  See notes/K-space IPM with Sparse Cholesky.md.

The forward/backward passes are factored out for easy swapping:

    sparse_inverse_fwd(z, geom)       z → u = (LL^T)^{-1}_E  (+ cache)
    sparse_inverse_bwd(g, cache, geom) g=df/du → dz=df/dz    (VJP)
"""

import time

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from ._shared import project_psd, _weights


# ── Index geometry ────────────────────────────────────────────────────────────

def _sparse_inverse_geom(d, inds):
    """Precompute index geometry for the dense K-space Cholesky solver."""
    rows_obs, cols_obs = inds[:, 0], inds[:, 1]
    tril_r, tril_c = np.tril_indices(d)
    is_diag = (tril_r == tril_c)
    obs_is_diag = (rows_obs == cols_obs)
    return dict(d=d, rows_obs=rows_obs, cols_obs=cols_obs,
                tril_r=tril_r, tril_c=tril_c, is_diag=is_diag,
                obs_is_diag=obs_is_diag)


# ── Forward pass ──────────────────────────────────────────────────────────────

def sparse_inverse_fwd(z, geom):
    """Forward pass:  z (lower-tri of L)  →  u = ((LL^T)^{-1})_E.

    Returns (u, logdet_K, cache) where cache holds intermediates for backward.
    """
    d = geom['d']
    tril_r, tril_c = geom['tril_r'], geom['tril_c']
    rows_obs, cols_obs = geom['rows_obs'], geom['cols_obs']

    L = np.zeros((d, d))
    L[tril_r, tril_c] = z
    diag_L = np.diag(L)
    logdet_K = 2.0 * np.sum(np.log(diag_L))

    # X = (LL^T)^{-1}  via two triangular solves
    Y = la.solve_triangular(L, np.eye(d), lower=True, check_finite=False)
    X = la.solve_triangular(L, Y, lower=True, trans='T', check_finite=False)

    u = X[rows_obs, cols_obs]
    cache = (X, Y, diag_L)
    return u, logdet_K, cache


# ── Backward pass (VJP) ───────────────────────────────────────────────────────

def sparse_inverse_bwd(g, cache, geom):
    """Backward pass:  g = df/du  →  dz = df/dz  (VJP through the sparse inverse).

    Chain rule:  u = extract(X),  X = K^{-1},  K = LL^T.

      1. Symmetric lift:  G = adjoint of extracting u from symmetric X
         G_ij = G_ji = g_k/2  (off-diagonal obs k at (i,j))
         G_ii = g_k            (diagonal obs k at (i,i))

      2. Through X = K^{-1}:  dK = -X G X

      3. Through K = LL^T:    dL = 2 dK L
         Using XL = L^{-T} = Y^T:  dL = -2 X G Y^T

      4. Extract lower triangle of dL.
    """
    d = geom['d']
    tril_r, tril_c = geom['tril_r'], geom['tril_c']
    rows_obs, cols_obs = geom['rows_obs'], geom['cols_obs']
    obs_is_diag = geom['obs_is_diag']
    X, Y, _ = cache

    # Symmetric lift
    G = np.zeros((d, d))
    G[rows_obs, cols_obs] = 0.5 * g
    G[cols_obs, rows_obs] = 0.5 * g
    G[rows_obs[obs_is_diag], cols_obs[obs_is_diag]] = g[obs_is_diag]

    # dL = -2 X G Y^T  (see derivation above)
    dz = (-2.0 * X @ G @ Y.T)[tril_r, tril_c]
    return dz


# ── Solver ────────────────────────────────────────────────────────────────────

def ipm_chol_kspace(A, inds, tau,
                    mu_init=1.0, mu_factor=0.5, eps=1e-8, eps_coarse=1e-2,
                    max_outer=40, max_inner=200):
    """
    IPM with Cholesky parameterization K = LL^T  (L lower triangular, L[i,i] > 0).

    Minimizes  φ_μ(L) = f((LL^T)^{-1}_E) + μ log|LL^T|  over L, decreasing μ → 0.
    Equivalent to ipm_chol but parameterizing the precision K = X^{-1} instead of X.

    The forward/backward passes (sparse_inverse_fwd, sparse_inverse_bwd) are
    factored out so the same IPM loop can be reused with a sparse Takahashi
    implementation.

    Status: converges at low noise; fails at med/high noise due to K-space
    ill-conditioning near the PSD boundary.
    """
    d = A.shape[0]
    geom = _sparse_inverse_geom(d, inds)
    rows_obs, cols_obs = geom['rows_obs'], geom['cols_obs']
    is_diag = geom['is_diag']

    w = _weights(tau)
    a_obs = A[rows_obs, cols_obs]

    bounds = [(1e-8, None) if is_diag[k] else (None, None) for k in range(len(is_diag))]

    # Initialize: K₀ = α⁻¹ I  →  L₀ = α^{-1/2} I  (so X₀ = K₀⁻¹ = α I)
    alpha = max(float(np.mean(np.abs(np.diag(A)))), 1.0)
    z  = np.where(is_diag, 1.0 / np.sqrt(alpha), 0.0)
    mu = float(mu_init)

    hist = []
    t0   = time.perf_counter()

    for _outer in range(max_outer):
        eps_inner = min(mu, eps_coarse)

        def phi_and_grad(z, _mu=mu):
            # Forward: z → u (observed entries of X = (LL^T)^{-1})
            u, logdet_K, cache = sparse_inverse_fwd(z, geom)
            _, _, diag_L = cache

            # Objective and gradient w.r.t. u
            r = u - a_obs
            fval = float(np.sum(w * r ** 2))
            g = 2.0 * w * r

            # Backward: g → dz (VJP through sparse inverse)
            dz = sparse_inverse_bwd(g, cache, geom)

            # Barrier: μ log|K| = μ · 2 Σ log L_ii
            dz[is_diag] += 2.0 * _mu / diag_L

            return float(fval + _mu * logdet_K), dz

        result = minimize(phi_and_grad, z, method='L-BFGS-B', jac=True,
                          bounds=bounds,
                          options={'maxiter': max_inner, 'ftol': 0, 'gtol': eps_inner})
        z = result.x

        # Record diagnostics
        u, logdet_K, cache = sparse_inverse_fwd(z, geom)
        fval = float(np.sum(w * (u - a_obs) ** 2))

        hist.append({
            'time':        time.perf_counter() - t0,
            'obj':         fval,
            'entropy':     -logdet_K,       # log|X| = -log|K|
            'mu':          mu,
            'inner_iters': result.nit,
        })

        if mu <= eps:
            break
        mu *= mu_factor

    X = cache[0]
    return project_psd(X), hist
