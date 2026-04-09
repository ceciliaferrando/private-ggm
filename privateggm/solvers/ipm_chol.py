"""ipm_chol — IPM with Cholesky parameterization X = LL^T (+ min_eig·I)."""

import time

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from ._shared import project_psd, _weights, _weighted_obj, normalize_coo


def ipm_chol(rows, cols, vals, tau, d,
             mu_init=1.0, mu_factor=0.5, eps=1e-8, eps_coarse=1e-2,
             max_outer=40, max_inner=200,
             min_eig=0.0, logdet_thresh=-np.inf):
    """
    IPM with Cholesky parameterization X = LL^T  (L lower triangular, L[i,i] > 0).

    Minimizes  φ_μ(X) = f(X_E) − μ log|X|  over L, decreasing μ → 0.  The
    central path converges to the max-entropy solution because −log|X| is the
    negative Gaussian entropy.

    Two cases depending on min_eig:

    min_eig = 0  (default):
        X = LL^T.  Key identity: log|LL^T| = 2 Σᵢ log L[i,i], so the entropy
        and its gradient are diagonal-only — no matrix inverse, O(d²) gradient.
        X ≻ 0 is guaranteed by the bound L[i,i] ≥ ε.

    min_eig > 0:
        X = LL^T + min_eig·I, enforcing X ⪰ min_eig·I.  The entropy is
        log|X| = log|LL^T + min_eig·I|, which requires a Cholesky of X per
        step (O(d³)) and a triangular solve for the gradient (d/dL log|X| =
        2 X⁻¹L).  The central path still converges to the max-entropy
        minimizer of f subject to X ⪰ min_eig·I.

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
    min_eig        : enforce X ⪰ min_eig·I (0 = standard PSD cone)
    logdet_thresh  : stop as soon as log det(X) drops below this value.
                     Default -inf (disabled).

    Hist fields  (one entry per outer iteration)
    -----------
    Uniform  : time, obj, entropy
    IPM      : mu, inner_iters
    KKT      : min_eig_G, comp_slack
    """
    rows, cols, vals, tau = normalize_coo(rows, cols, vals, tau)
    diag_d = np.arange(d)

    w   = _weights(tau)
    off = rows != cols

    tril_r, tril_c = np.tril_indices(d)
    is_diag = (tril_r == tril_c)
    bounds  = [(1e-8, None) if is_diag[k] else (None, None) for k in range(len(tril_r))]

    def vec_to_L(z):
        L = np.zeros((d, d))
        L[tril_r, tril_c] = z
        return L

    # Initialize: X₀ = α I  →  L₀ = √α I
    alpha = max(float(np.mean(np.abs(vals[~off]))), 1.0)
    z  = np.where(is_diag, np.sqrt(alpha), 0.0)
    mu = float(mu_init)

    hist = []
    t0   = time.perf_counter()

    for _outer in range(max_outer):
        eps_inner = min(mu, eps_coarse)

        def phi_and_grad(z, _mu=mu):
            L      = vec_to_L(z)
            diag_L = L[diag_d, diag_d]
            X      = L @ L.T + min_eig * np.eye(d)
            r      = X[rows, cols] - vals
            fval   = float(np.sum(w * r ** 2))
            G = np.zeros((d, d))
            G[rows[off],  cols[off]] = w[off] * r[off]
            G[cols[off],  rows[off]] = w[off] * r[off]
            G[rows[~off], cols[~off]] = w[~off] * r[~off]
            G[diag_d, diag_d] *= 2
            GL     = G @ L
            grad_z = 2.0 * GL[tril_r, tril_c]

            if min_eig == 0.0:
                # Cheap: log|LL^T| = 2 Σ log L_ii; gradient is diagonal-only.
                logdet = 2.0 * np.sum(np.log(diag_L))
                grad_z[is_diag] -= 2.0 * _mu / diag_L
            else:
                # Full: log|X| = log|LL^T + min_eig·I|; gradient is -2μ X⁻¹L.
                # Use Cholesky of X for both logdet and the solve.
                L_X    = np.linalg.cholesky(X)
                logdet = 2.0 * np.sum(np.log(np.diag(L_X)))
                X_inv_L = la.cho_solve((L_X, True), L)
                grad_z -= 2.0 * _mu * X_inv_L[tril_r, tril_c]

            return float(fval - _mu * logdet), grad_z

        result = minimize(phi_and_grad, z, method='L-BFGS-B', jac=True,
                          bounds=bounds,
                          options={'maxiter': max_inner, 'ftol': eps_inner, 'gtol': eps_inner})
        z     = result.x
        L_opt = vec_to_L(z)
        X     = L_opt @ L_opt.T + min_eig * np.eye(d)

        r = X[rows, cols] - vals
        fval = float(np.sum(w * r ** 2))
        G = np.zeros((d, d))
        G[rows[off],  cols[off]] = w[off] * r[off]
        G[cols[off],  rows[off]] = w[off] * r[off]
        G[rows[~off], cols[~off]] = w[~off] * r[~off]
        G[diag_d, diag_d] *= 2
        if min_eig == 0.0:
            logdet = 2.0 * float(np.sum(np.log(z[is_diag])))
        else:
            logdet = float(np.linalg.slogdet(X)[1])

        hist.append({
            'time':        time.perf_counter() - t0,
            'obj':         fval,
            'entropy':     logdet,
            'mu':          mu,
            'inner_iters': result.nit,
            'min_eig_G':   float(np.linalg.eigvalsh(G).min()),
            'comp_slack':  float(abs(np.sum(G * X))),
        })

        if logdet < logdet_thresh:
            break
        if mu <= eps:
            break
        mu *= mu_factor

    return X, hist
