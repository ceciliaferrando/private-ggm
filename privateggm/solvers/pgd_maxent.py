"""pgd and pgd_maxent solvers.

pgd       — Stage A projected gradient descent (PSD baseline, no max-entropy structure)
pgd_maxent — Two-stage: Stage A PGD then Stage B max-entropy completion
"""

import time

import numpy as np
from scipy.optimize import minimize

from ._shared import project_psd, _weights, _weighted_obj, normalize_coo


def pgd(rows, cols, vals, tau, d,
        eta=0.1, max_iters=2000, eps=1e-8, track_kkt=False,
        fill_unobserved=0.0):
    """
    Stage A: Projected Gradient Descent onto the PSD cone.

    Minimizes f(X_E) subject to X ⪰ 0.  Does NOT enforce max-entropy structure;
    use pgd_maxent for that.

    Parameters
    ----------
    rows, cols       : (m,) int arrays — observed index pairs (any triangular format)
    vals             : (m,) float array — noisy observed values
    tau              : (m,) float array — noise std-devs
    d                : int — matrix dimension
    eta              : initial step size
    max_iters        : maximum PGD iterations
    eps              : convergence tolerance on step size ‖X_new − X‖_F
    track_kkt        : if True, record min_eig_X, min_eig_G, comp_slack each iter
                       (adds one eigendecomposition per step — disable for speed)
    fill_unobserved  : value used to fill unobserved entries of X at initialization
                       (default 0.0; try small positive values to bias toward denser
                       initial guess before PSD projection)

    Hist fields
    -----------
    Uniform : time, obj, entropy
    PGD     : delta (‖X_new − X‖_F), eta (step size)
    KKT     : min_eig_X, min_eig_G, comp_slack  (only if track_kkt=True)
    """
    rows, cols, vals, tau = normalize_coo(rows, cols, vals, tau)
    w = _weights(tau)

    # Build local dense A for initialization and gradient computation.
    # Unobserved entries only affect initialization (gradient is masked to E).
    A = np.full((d, d), fill_unobserved)
    A[rows, cols] = vals
    A[cols, rows] = vals

    W = np.zeros((d, d))
    W[rows, cols] = w
    W[cols, rows] = w
    mask = W > 0

    armijo = 0.5
    beta   = 0.5
    gamma  = 1.05

    X    = project_psd(A.copy())
    hist = []
    t0   = time.perf_counter()

    for _ in range(max_iters):
        G   = np.where(mask, W * (X - A), 0.0)
        G[np.arange(d), np.arange(d)] *= 2
        obj = _weighted_obj(X, rows, cols, vals, w)

        # Armijo backtracking line search for PGD
        for _ in range(50):
            X_new, evals_new = project_psd(X - eta * G, return_evals=True)
            predicted_decrease = -np.sum(G * (X_new - X))
            if _weighted_obj(X_new, rows, cols, vals, w) <= obj - armijo * predicted_decrease:
                break
            eta *= beta
            if eta < 1e-14:
                break

        delta = float(np.linalg.norm(X_new - X, 'fro'))
        entry = {
            'time':    time.perf_counter() - t0,
            'obj':     obj,
            'entropy': float(np.sum(np.log(evals_new))),
            'delta':   delta,
            'eta':     eta,
        }
        if track_kkt:
            entry['min_eig_X']  = float(np.linalg.eigvalsh(X).min())
            entry['min_eig_G']  = float(np.linalg.eigvalsh(G).min())
            entry['comp_slack'] = float(abs(np.sum(G * X)))

        hist.append(entry)
        X    = X_new
        eta *= gamma

        if delta < eps:
            break

    return X, hist


def pgd_maxent(rows, cols, vals, tau, d,
               eta=0.1, pgd_iters=2000, pgd_eps=1e-8,
               lbfgs_iters=1000, eps=1e-8,
               track_kkt=False, fill_unobserved=0.0):
    """
    Two-stage pipeline: Stage A PGD then Stage B max-entropy completion.

    Stage A finds X̃ ⪰ 0 fitting the observations; Stage B solves the
    dual of max-entropy completion given X̃_S as constraints.

    Parameters
    ----------
    rows, cols  : (m,) int arrays — observed index pairs (any triangular format)
    vals        : (m,) float array — noisy observed values
    tau         : (m,) float array — noise std-devs
    d           : int — matrix dimension
    eta              : Stage A initial step size
    pgd_iters        : Stage A max PGD iterations
    pgd_eps          : Stage A convergence tolerance on step size
    lbfgs_iters      : Stage B max L-BFGS iterations
    eps              : Stage B L-BFGS gradient tolerance
    track_kkt        : if True, add KKT diagnostics to Stage A hist entries
    fill_unobserved  : passed to pgd — value for unobserved entries at init

    Returns
    -------
    Sigma_hat : (d, d) estimated covariance (max-entropy completion of X̃_S)
    hist      : combined per-iteration list; Stage A then Stage B entries

    Hist fields
    -----------
    Uniform : time, obj, entropy
    Stage A : delta, eta  (+ min_eig_X, min_eig_G, comp_slack if track_kkt)
    Stage B : dual_loss, constraint_viol, grad_norm  (dual-domain diagnostics)
    """
    rows, cols, vals, tau = normalize_coo(rows, cols, vals, tau)
    w    = _weights(tau)

    t0    = time.perf_counter()
    X_psd, hist = pgd(rows, cols, vals, tau, d, eta=eta, max_iters=pgd_iters, eps=pgd_eps,
                      track_kkt=track_kkt, fill_unobserved=fill_unobserved)
    for entry in hist:
        entry['stage'] = 'A'

    # ── Stage B: max-entropy completion ───────────────────────────────────────
    diag_mask = rows == cols
    mult      = np.where(diag_mask, 1.0, 2.0)

    def _build_K(z):
        K = np.zeros((d, d))
        K[rows, cols] = z
        K[cols, rows] = z
        return K

    # Mutable cell so loss_and_grad can share Sigma with the callback.
    _last = [None, None]   # [logdet_K, Sigma]

    def loss_and_grad(z):
        K = _build_K(z)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            shift = -np.linalg.eigvalsh(K).min() + 1e-4
            K_reg = K + shift * np.eye(d)
            Sigma_reg = np.linalg.inv(K_reg)
            grad = mult * (X_psd[rows, cols] - Sigma_reg[rows, cols])
            _last[0] = None
            _last[1] = None
            return 1e15, grad
        logdet_K = 2.0 * np.sum(np.log(np.diag(L)))
        Sigma = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(d)))
        _last[0] = logdet_K
        _last[1] = Sigma
        loss  = -logdet_K + np.sum(K * X_psd)
        grad  = mult * (X_psd[rows, cols] - Sigma[rows, cols])
        return float(loss), grad

    def callback(z):
        logdet_K, Sigma = _last
        if logdet_K is None:
            return
        grad = mult * (X_psd[rows, cols] - Sigma[rows, cols])
        hist.append({
            'time':             time.perf_counter() - t0,
            'obj':              _weighted_obj(Sigma, rows, cols, vals, w),
            'entropy':          float(-logdet_K),
            'dual_loss':        float(-logdet_K + np.sum(_build_K(z) * X_psd)),
            'constraint_viol':  float(np.sqrt(np.mean((Sigma[rows, cols] - X_psd[rows, cols]) ** 2))),
            'grad_norm':        float(np.linalg.norm(grad)),
            'stage':            'B',
        })

    z0 = np.where(diag_mask, 1.0 / np.maximum(X_psd[rows, cols], 1e-6), 0.0)
    result = minimize(
        loss_and_grad, z0,
        method='L-BFGS-B', jac=True,
        callback=callback,
        options={'maxiter': lbfgs_iters, 'ftol': 0, 'gtol': eps},
    )

    K_opt = _build_K(result.x)
    try:
        Sigma_hat = np.linalg.inv(K_opt)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            "Stage B precision matrix K is singular at convergence. "
            "This typically means Stage B diverged near the PSD boundary."
        )
    return Sigma_hat, hist
