"""clarabel_solver — off-the-shelf conic IPM via Clarabel (requires clarabel >= 0.11.1)."""

import time

import clarabel
import numpy as np
from scipy import sparse

from ._shared import _weights, _weighted_obj


def clarabel_solver(A, inds, tau,
                    verbose=False, max_iter=200,
                    tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8):
    """
    Off-the-shelf conic IPM via Clarabel (requires clarabel >= 0.11.1).

    Formulates the problem as a QP with PSD cone constraint:
        min  (1/2) x^T P x + q^T x   s.t.  -D x + s = 0,  s in PSDTriangleCone(d)
    where x is the upper triangle of X in column-major order and D is diagonal
    with entries 1 (diagonal of X) or sqrt(2) (off-diagonal of X).

    Clarabel's cost_primal = f(X_E) - C where C = sum(w * a^2) is the constant
    dropped when expanding the squared-error objective. The callback adds C back
    to recover our objective at each iteration.

    Parameters
    ----------
    verbose     : pass True to enable Clarabel's own iteration output
    max_iter    : maximum IPM iterations
    tol_gap_abs : absolute duality gap tolerance
    tol_gap_rel : relative duality gap tolerance
    tol_feas    : primal/dual feasibility tolerance

    Hist fields  (one entry per IPM iteration)
    -----------
    Uniform  : time, obj, entropy  (entropy = nan except on the last entry)
    Clarabel : mu, res_primal, res_dual, status
    """
    d    = A.shape[0]
    nvec = d * (d + 1) // 2

    # Upper-triangle column-major ordering: column c, rows 0..c
    r_ut = np.array([r for c in range(d) for r in range(c + 1)])
    c_ut = np.array([c for c in range(d) for r in range(c + 1)])
    is_diag_ut = (r_ut == c_ut)

    # Map inds (which may be upper- or lower-triangle) to upper-triangle
    # column-major index.  For upper-triangle entry (r,c) with r<=c:
    #   k = c*(c+1)//2 + r  =  big*(big+1)//2 + small
    i_obs, j_obs = inds[:, 0], inds[:, 1]
    big   = np.maximum(i_obs, j_obs)
    small = np.minimum(i_obs, j_obs)
    k_obs = big * (big + 1) // 2 + small

    w     = _weights(tau)
    a_obs = A[i_obs, j_obs]

    # Constant dropped from Clarabel objective: C = sum(w * a^2)
    # Recover our objective via:  obj = cost_primal + C_const
    C_const = float(np.sum(w * a_obs ** 2))

    # P: diagonal sparse, P[k,k] = 2w for observed entries
    P_diag        = np.zeros(nvec)
    P_diag[k_obs] = 2.0 * w
    P = sparse.diags(P_diag, format='csc')

    # q: q[k] = -2*w*a for observed entries
    q        = np.zeros(nvec)
    q[k_obs] = -2.0 * w * a_obs

    # Constraint matrix: A_clar = -D  where D[k,k] = 1 (diag) or sqrt(2) (off-diag)
    # Encodes  -D x + s = 0  =>  s = D x = vec(X)  in PSDTriangleCone
    scale  = np.where(is_diag_ut, 1.0, np.sqrt(2.0))
    A_clar = sparse.diags(-scale, format='csc')
    b      = np.zeros(nvec)
    cones  = [clarabel.PSDTriangleConeT(d)]

    settings             = clarabel.DefaultSettings()
    settings.verbose     = verbose
    settings.max_iter    = max_iter
    settings.tol_gap_abs = tol_gap_abs
    settings.tol_gap_rel = tol_gap_rel
    settings.tol_feas    = tol_feas

    hist = []
    t0   = time.perf_counter()

    def _callback(info):
        hist.append({
            'time':       time.perf_counter() - t0,
            'obj':        info.cost_primal + C_const,
            'entropy':    float('nan'),
            'mu':         info.mu,
            'res_primal': info.res_primal,
            'res_dual':   info.res_dual,
            'status':     str(info.status),
        })
        return False  # never request early termination

    solver = clarabel.DefaultSolver(P, q, A_clar, b, cones, settings)
    solver.set_termination_callback(_callback)
    sol = solver.solve()

    # Reconstruct Sigma_hat from solution vector x (upper triangle, column-major)
    x         = np.array(sol.x)
    Sigma_hat = np.zeros((d, d))
    Sigma_hat[r_ut, c_ut] = x
    Sigma_hat[c_ut, r_ut] = x  # symmetrize

    # Patch final entry with entropy and solved status; earlier entries stay nan
    _, logdet = np.linalg.slogdet(Sigma_hat)
    if hist:
        hist[-1]['entropy'] = float(logdet)
        hist[-1]['status']  = str(sol.status)
    else:
        hist.append({
            'time':    time.perf_counter() - t0,
            'obj':     _weighted_obj(Sigma_hat, A, inds, w),
            'entropy': float(logdet),
            'status':  str(sol.status),
        })

    return Sigma_hat, hist
