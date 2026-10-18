import sys
import numpy as np
from math import exp, sqrt
from scipy.special import erf
import scipy.optimize
from types import SimpleNamespace
from privateggm.privacy import cov_jk_sensitivity_linf, cov_jk_sensitivity_linf_diag, cov_jk_sensitivity_l2, SSP_gradient, gaussian_mechanism

# ── PrivateCovariance repo wrappers ───────────────────────────────────────────
# Clone https://github.com/hkustDB/PrivateCovariance and set this path, or
# export PRIVATE_COV_REPO=/path/to/PrivateCovariance before running.
import os as _os
_PRIVATE_COV_REPO = _os.environ.get(
    'PRIVATE_COV_REPO',
    _os.path.expanduser('~/PrivateCovariance'),
)


def _ensure_private_cov_on_path():
    """Add PrivateCovariance repo to sys.path if not already importable."""
    try:
        import adaptive.algos  # noqa: F401 — already on path
        return
    except ModuleNotFoundError:
        pass
    if not _os.path.isdir(_PRIVATE_COV_REPO):
        raise RuntimeError(
            f"PrivateCovariance repo not found at '{_PRIVATE_COV_REPO}'. "
            "Clone https://github.com/hkustDB/PrivateCovariance and set "
            "the PRIVATE_COV_REPO environment variable (or edit "
            "_PRIVATE_COV_REPO in baselines.py)."
        )
    if _PRIVATE_COV_REPO not in sys.path:
        sys.path.insert(0, _PRIVATE_COV_REPO)


def _to_tensor(X_np):
    """Convert a float64 numpy array to a float32 torch tensor."""
    import torch
    return torch.from_numpy(X_np.astype('float32'))


def adaptive_cov(X, rho, r=1.0, beta=0.1):
    """Private covariance via AdaptiveCov (zCDP).

    Wrapper around `adaptive.algos.AdaptiveCov` from the PrivateCovariance repo
    (Wang et al., 2021). Uses the adaptive Sparse Vector Technique to choose
    between a Gaussian and a SeparateCov estimator after adaptively clipping rows.

    Args:
        X:    numpy array of shape (n, d), raw (centred) data matrix.
        rho:  zCDP privacy budget.
        r:    L2 row-norm upper bound (default 1.0; clip X rows before calling).
        beta: failure probability (default 0.1).

    Returns:
        Sigma_dp: numpy array of shape (d, d), private covariance estimate.
    """
    _ensure_private_cov_on_path()
    from adaptive.algos import AdaptiveCov

    n, d = X.shape
    args = SimpleNamespace(total_budget=rho, n=n, d=d, beta=beta)
    X_t = _to_tensor(X)
    result = AdaptiveCov(X_t, args, r=r)
    return result.numpy().astype('float64')


def gauss_cov(X, rho, r=1.0, b_fleig=True):
    """Private covariance via GaussCov (zCDP) — simple Gaussian noise baseline.

    Wrapper around `adaptive.algos.GaussCov` from the PrivateCovariance repo.

    Args:
        X:       numpy array of shape (n, d).
        rho:     zCDP privacy budget.
        r:       L2 row-norm upper bound (default 1.0).
        b_fleig: project eigenvalues to [0, r^2] (default True).

    Returns:
        Sigma_dp: numpy array of shape (d, d).
    """
    _ensure_private_cov_on_path()
    from adaptive.algos import GaussCov

    n, d = X.shape
    X_t = _to_tensor(X)
    result = GaussCov(X_t, n, d, rho, r=r, b_fleig=b_fleig)
    return result.numpy().astype('float64')


def em_cov(X, rho, delta=1e-5, beta=0.1, b_budget=False, b_fleig=True):
    """Private covariance via EMCov (approximate DP).

    Wrapper around `exponential.algos.EMCov` from the PrivateCovariance repo.
    Uses an EM algorithm with the Bingham distribution to iteratively estimate
    eigenvectors under (epsilon, delta)-DP.

    The zCDP budget rho is converted to epsilon via the standard formula:
        epsilon = rho + 2 * sqrt(rho * log(1/delta))

    Args:
        X:        numpy array of shape (n, d), raw (centred) data matrix.
                  Rows must already satisfy the norm bound assumed by the method.
        rho:      zCDP budget (converted to approx-DP internally).
        delta:    delta for (epsilon, delta)-DP (default 1e-5).
        beta:     failure probability used by the budget-allocation heuristic
                  when b_budget=True (default 0.1).
        b_budget: if True, use the eigenvalue-weighted budget allocation (Eq. 12
                  of the paper); if False, uniform allocation (default False).
        b_fleig:  project eigenvalues to [0, n] (default True).

    Returns:
        Sigma_dp: numpy array of shape (d, d), private covariance estimate.
    """
    _ensure_private_cov_on_path()
    from exponential.algos import EMCov

    n, d = X.shape
    args = SimpleNamespace(total_budget=rho, delta=delta, n=n, d=d, beta=beta)
    X_t = _to_tensor(X)
    result = EMCov(X_t, args, b_budget=b_budget, b_fleig=b_fleig)
    return result.numpy().astype('float64')


def coinpress_cov(X, rho, r=1.0, t=4):
    """Private covariance via CoinPress/cov_est (zCDP).

    Wrapper around `coinpress.algos.cov_est` from the PrivateCovariance repo.
    Uses an iterative Gaussian-noise + PSD-projection scheme.

    Budget split across t iterations follows the schedule from the paper:
        rho[i] = rho / (4*(t-1))   for i = 0, ..., t-2
        rho[t-1] = 3/4 * rho       (last iteration gets 3/4 of the budget)

    Args:
        X:   numpy array of shape (n, d), raw (centred) data matrix.
        rho: total zCDP privacy budget.
        r:   L2 row-norm upper bound; used to set the eigenvalue upper bound
             u = r^2 (default 1.0).
        t:   number of CoinPress iterations (default 4).

    Returns:
        Sigma_dp: numpy array of shape (d, d), private covariance estimate.
    """
    _ensure_private_cov_on_path()
    from coinpress.algos import cov_est

    n, d = X.shape
    u = r * r  # eigenvalue upper bound

    if t == 1:
        rho_schedule = [rho]
    else:
        rho_per_early = rho / (4.0 * (t - 1))
        rho_schedule = [rho_per_early] * (t - 1) + [0.75 * rho]

    args = SimpleNamespace(n=n, d=d, u=u, t=t, rho=rho_schedule)
    X_t = _to_tensor(X)
    result = cov_est(X_t, args)
    return result.numpy().astype('float64')


def zcdp_to_approxdp(rho, delta):
    """Convert rho-zCDP guarantee to (epsilon, delta)-DP via standard formula.

    Given rho-zCDP, the mechanism satisfies (epsilon, delta)-DP where:
        epsilon = rho + 2 * sqrt(rho * ln(1/delta))

    Args:
        rho:   zCDP parameter
        delta: target delta for approximate DP (e.g. 1e-5)

    Returns:
        epsilon: the corresponding approximate-DP epsilon
    """
    return rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta))


def AdaSSP_linear_regression(Sigma, n, rho, bound_X, bound_y, bound_XTX,
                              failure_prob=0.05, XTy_budget_split=0.5,
                              delta=1e-5):
    """Differentially private linear regression via AdaSSP (Algorithm 2 of
    Wang et al., 2018; https://arxiv.org/pdf/1803.02596.pdf).

    Uses the original (epsilon, delta)-DP noise calibration.  The zCDP budget
    rho is converted to (epsilon, delta)-DP via zcdp_to_approxdp before being
    split across the three mechanism steps.

    Args:
        Sigma:            numpy array of shape (d+1, d+1), joint empirical covariance
                          with target as the last row/column (normalised by n-1)
        n:                number of training samples
        rho:              zCDP privacy budget (converted internally to epsilon-delta)
        bound_X:          L2 row-norm bound on X
        bound_y:          bound on |y|
        bound_XTX:        spectral-norm sensitivity bound on X^T X
        failure_prob:     failure probability for the adaptive ridge term (rho in
                          original paper, default 0.05)
        XTy_budget_split: fraction of the 2/3 epsilon budget allocated to X^T y
                          (default 0.5, matching AdaSSP paper)
        delta:            delta parameter for (epsilon, delta)-DP conversion (default 1e-5)

    Returns:
        theta_dp:    numpy array of shape (d,), privatised regression coefficients
        XTX_dp_reg:  privatised regularised X^T X (d x d)
        XTy_dp:      privatised X^T y (d,)
        lambda_dp:   adaptive ridge magnitude applied to X^T X
    """
    Sigma_XX = Sigma[:-1, :-1]
    Sigma_Xy = Sigma[:-1, -1]
    d = Sigma_XX.shape[0]

    # Recover unnormalised sufficient statistics (covariance uses n-1 denominator)
    XTX = (n - 1) * Sigma_XX
    XTy = (n - 1) * Sigma_Xy

    # ── Convert rho-zCDP → (epsilon, delta)-DP ───────────────────────────────
    epsilon = zcdp_to_approxdp(rho, delta)

    # ── Budget splits (matching original AdaSSP) ─────────────────────────────
    epsilon_eigen = epsilon / 3.0
    delta_eigen   = delta   / 3.0

    epsilon_XTX   = epsilon * (2.0 / 3.0) * (1.0 - XTy_budget_split)
    delta_XTX     = delta   * (2.0 / 3.0) * (1.0 - XTy_budget_split)

    epsilon_XTy   = epsilon * (2.0 / 3.0) * XTy_budget_split
    delta_XTy     = delta   * (2.0 / 3.0) * XTy_budget_split

    # ── (epsilon, delta)-DP noise sensitivities: sqrt(log(2/delta)) / epsilon ─
    sensitivity_eigen = np.sqrt(np.log(2.0 / delta_eigen)) / epsilon_eigen
    sensitivity_XTX   = np.sqrt(np.log(2.0 / delta_XTX))  / epsilon_XTX
    sensitivity_XTy   = np.sqrt(np.log(2.0 / delta_XTy))  / epsilon_XTy

    # ── Adaptive ridge: privatise minimum eigenvalue ──────────────────────────
    eigen_min    = max(0.0, float(np.min(np.linalg.eigvalsh(XTX))))
    z_eigen      = np.random.normal(0.0, 1.0)
    eigen_min_dp = max(0.0,
                       eigen_min
                       + sensitivity_eigen * bound_XTX * z_eigen
                       - bound_XTX * np.log(6.0 / delta) / epsilon_eigen)

    # Adaptive ridge magnitude (original formula, failure_prob = rho in paper)
    lambda_dp = max(0.0,
                    np.sqrt(d * np.log(2.0 / delta_eigen)
                              * np.log(2.0 * d**2 / failure_prob))
                    * bound_XTX / epsilon_eigen
                    - eigen_min_dp)

    # ── Privatise X^T X ───────────────────────────────────────────────────────
    tri  = np.triu(np.random.normal(0.0, 1.0, (d, d)))
    Zsym = tri + tri.T - np.diag(np.diag(tri))
    XTX_dp = XTX + sensitivity_XTX * bound_XTX * Zsym

    # ── Privatise X^T y ───────────────────────────────────────────────────────
    z_XTy  = np.random.normal(0.0, 1.0, size=(d,))
    XTy_dp = XTy + sensitivity_XTy * bound_X * bound_y * z_XTy

    # ── Solve regularised system ──────────────────────────────────────────────
    XTX_dp_reg = XTX_dp + lambda_dp * np.eye(d)
    theta_dp   = np.linalg.solve(XTX_dp_reg, XTy_dp)

    return theta_dp, XTX_dp_reg, XTy_dp, lambda_dp


def private_min_eigenvalue_ridge_zcdp(Sigma_XX, rho_eigen, sigma_sensitivity,
                                       failure_prob=0.05, delta=1e-5, rng=None):
    """Privately estimate the minimum eigenvalue of Sigma_XX and compute an
    adaptive ridge parameter, mirroring the AdaSSP eigenvalue step exactly.

    The rho-zCDP budget is converted to (epsilon, delta)-DP via zcdp_to_approxdp,
    and then the original AdaSSP noise calibration and threshold formula are used
    unchanged on the Sigma (not XTX) scale.

    Args:
        Sigma_XX:          (d, d) empirical feature covariance
        rho_eigen:         zCDP budget for this step (typically rho_total / 3)
        sigma_sensitivity: spectral-norm sensitivity of Sigma_XX under one-sample
                           change (= bound_XTX / (n - 1))
        failure_prob:      failure probability for PSD guarantee (default 0.05)
        delta:             delta for zCDP -> approxDP conversion (default 1e-5)
        rng:               numpy random Generator

    Returns:
        eigen_min_dp:  privately estimated minimum eigenvalue (Sigma scale, >= 0)
        lambda_sigma:  adaptive ridge to add to Sigma_XX (Sigma scale, >= 0)
        noise_std:     noise standard deviation used
    """
    if rng is None:
        rng = np.random.default_rng()

    d = Sigma_XX.shape[0]
    eigen_min = float(np.min(np.linalg.eigvalsh(Sigma_XX)))

    # Convert rho-zCDP -> (epsilon, delta)-DP, then use AdaSSP noise calibration
    epsilon = zcdp_to_approxdp(rho_eigen, delta)
    noise_std = np.sqrt(np.log(2.0 / delta)) / epsilon * sigma_sensitivity

    eigen_min_dp = max(0.0, eigen_min + rng.normal(0.0, noise_std)
                       - sigma_sensitivity * np.log(6.0 / delta) / epsilon)

    # AdaSSP threshold formula (unchanged, on Sigma scale)
    threshold = (np.sqrt(d * np.log(2.0 / delta) * np.log(2.0 * d**2 / failure_prob))
                 * sigma_sensitivity / epsilon)
    lambda_sigma = max(0.0, threshold - eigen_min_dp)

    return eigen_min_dp, lambda_sigma, noise_std


def diagonal_XTy_regression(Sigma, n, rho, T, rng, xty_frac=0.5):
    """Differentially private regression using only diagonal covariance entries and X^T y.

    Accepts the full joint covariance matrix (features + target as last column/row),
    extracts the diagonal of Sigma_XX and the cross-covariance Sigma_Xy, and
    privatises both using single-entry L∞ sensitivity under zCDP.

    Args:
        Sigma:     numpy array of shape (d+1, d+1), joint empirical covariance
                   with target as the last row/column (normalised by n-1)
        n:         number of training samples
        rho:       total zCDP privacy budget
        T:         clipping bound (L∞ per coordinate, used for sensitivity)
        rng:       numpy random Generator
        xty_frac:  fraction of rho allocated to X^T y (default 0.5)

    Returns:
        beta:          numpy array of shape (d,), privatised regression coefficients
        diag_XX_dp:    privatised diagonal of Sigma_XX
        Sigma_Xy_dp:   privatised Sigma_Xy
        noise_scale:   noise std used for X^T y privatisation
    """
    diag_XX  = np.diag(Sigma[:-1, :-1])   # (d,)
    Sigma_Xy = Sigma[:-1, -1]             # (d,)

    rho_Xy      = rho * xty_frac
    rho_diag    = rho * (1.0 - xty_frac)
    sensitivity = cov_jk_sensitivity_linf(n, T)

    diag_XX_dp, _      = SSP_gradient(diag_XX,  sensitivity, rho_diag, rng=rng)
    Sigma_Xy_dp, noise_scale = SSP_gradient(Sigma_Xy, sensitivity, rho_Xy,  rng=rng)

    diag_safe = np.maximum(diag_XX_dp, 1e-10)
    beta = Sigma_Xy_dp / diag_safe

    return beta, diag_XX_dp, Sigma_Xy_dp, noise_scale


# ── IHM: Analytic Gaussian Mechanism + RDP calibration helpers ────────────────
# Adapted from Balle & Wang ICML'18 (analytic Gaussian) and
# Lev et al. '25 HessianMix repo (RDP-to-DP binary search).

def _calibrate_analytic_gaussian(epsilon, delta, sensitivity, tol=1e-12):
    """Tight (ε,δ)-DP Gaussian noise std via the analytic mechanism."""
    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def caseA(eps, s):
        return Phi(sqrt(eps * s)) - exp(eps) * Phi(-sqrt(eps * (s + 2.0)))

    def caseB(eps, s):
        return Phi(-sqrt(eps * s)) - exp(eps) * Phi(-sqrt(eps * (s + 2.0)))

    def doubling(pred_stop, s_inf, s_sup):
        while not pred_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def bsearch(pred_stop, pred_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not pred_stop(s_mid):
            if pred_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)
    if delta == delta_thr:
        alpha = 1.0
    elif delta > delta_thr:
        pred_stop_DT = lambda s: caseA(epsilon, s) >= delta
        f = lambda s: caseA(epsilon, s)
        pred_left = lambda s: f(s) > delta
        f_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)
    else:
        pred_stop_DT = lambda s: caseB(epsilon, s) <= delta
        f = lambda s: caseB(epsilon, s)
        pred_left = lambda s: f(s) < delta
        f_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

    if delta != delta_thr:
        pred_stop_BS = lambda s: abs(f(s) - delta) <= tol
        s_inf, s_sup = doubling(pred_stop_DT, 0.0, 1.0)
        s_final = bsearch(pred_stop_BS, pred_left, s_inf, s_sup)
        alpha = f_alpha(s_final)

    return alpha * sensitivity / sqrt(2.0 * epsilon)


def _obj_full(alpha, k, gamma, delta, inflation):
    if alpha <= 1 or alpha >= gamma / (1 + inflation):
        return np.inf
    t1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + inflation) / gamma)
    t2 = -(k / (2 * (alpha - 1))) * np.log(1 - (alpha * (1 + inflation)) / gamma)
    t3 = (np.log(1.0 / delta) + (alpha - 1) * np.log(1 - 1 / alpha) - np.log(alpha)) / (alpha - 1)
    t4 = sqrt(2.0 * np.log(1.25 / delta)) / (gamma / sqrt(k))
    return t1 + t2 + t3 + t4


def _obj_full_comp(alpha, k, gamma, delta, inflation, T):
    if alpha <= 1 or alpha >= gamma / (1 + inflation):
        return np.inf
    t1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + inflation) / gamma)
    t2 = -(k / (2 * (alpha - 1))) * np.log(1 - (alpha * (1 + inflation)) / gamma)
    t3 = (np.log(1.0 / delta) + (alpha - 1) * np.log(1 - 1 / alpha) - np.log(alpha)) / (alpha - 1)
    t4 = sqrt(2.0 * np.log(1.25 / delta)) / (gamma / sqrt(k))
    return T * (t1 + t2) + t3 + t4


def _solve_gamma_renyi(init_gamma, k, target_delta, target_epsilon, inflation):
    """Binary search for gamma (noise² per sketch row) for a single sketch step."""
    left, right = init_gamma / 5e5, 5e5 * init_gamma
    best = right
    internal_delta = target_delta / 3.0
    while right - left > 1e-6:
        mid = (left + right) / 2
        res = scipy.optimize.minimize_scalar(
            _obj_full, bounds=(1 + 1e-5, mid - 1e-5),
            args=(k, mid, internal_delta, inflation), method='bounded')
        if res.success and res.fun < target_epsilon:
            best = mid
            right = mid
        else:
            left = mid
    return best


def _solve_gamma_renyi_comp(init_gamma, k, target_delta, target_epsilon, inflation, T):
    """Binary search for gamma for T composed sketch steps."""
    left, right = init_gamma / 5e5, 5e5 * init_gamma
    best = right
    internal_delta = target_delta / 3.0
    while right - left > 1e-6:
        mid = (left + right) / 2
        res = scipy.optimize.minimize_scalar(
            _obj_full_comp, bounds=(1 + 1e-5, mid - 1e-5),
            args=(k, mid, internal_delta, inflation, T), method='bounded')
        if res.success and res.fun < target_epsilon:
            best = mid
            right = mid
        else:
            left = mid
    return best


def ihm_regression(X_train, y_train, rho, T, n_iters=3, percentage_k=6,
                   delta=None, lambda_min=0.0, rng=None):
    """Iterative Hessian Mixing (IHM) for private linear regression (zCDP).

    Lev et al. 2025, "Near-Optimal Private Linear Regression via Iterative
    Hessian Mixing" (https://github.com/omrilev1/HessianMix).

    Args:
        X_train:      (n, d_feat) feature matrix, centered and [-1,1]-scaled.
        y_train:      (n,) target, centered and [-1,1]-scaled.
        rho:          zCDP privacy budget.
        T:            L∞ per-coordinate clipping bound (1.0 after rescaling).
        n_iters:      number of IHM iterations (default 3).
        percentage_k: sketch dimension multiplier: k = percentage_k * max(d, log(...)).
        delta:        DP delta (default 1/n²).
        lambda_min:   lower bound on min eigenvalue of X^T X (0 = conservative).
        rng:          numpy random Generator.

    Returns:
        theta: (d_feat,) regression coefficients.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = X_train.shape
    if delta is None:
        delta = 1.0 / (n ** 2)

    # Normalize X to unit max-L2 row norm and y to unit max-|y|.
    # This fixes two bugs from the original implementation:
    #   Bug 1 (Hessian): _solve_gamma_renyi_comp calibrates noise assuming max
    #     L2 row norm = 1. With L∞-scaled data, max L2 row norm = sqrt(d)*T,
    #     making the Hessian undercalibrated by a factor of d*T².
    #   Bug 2 (XTy): sensitivity of X^T@residual is max||x_i||_2 * C_max, not
    #     C_max alone. Old C_max = sqrt(d+1)*T missed the sqrt(d)*T factor from
    #     ||x_i||_2, undercalibrating noise by sqrt(d)*T.
    # After normalization both issues reduce to C_max = 1 (unit sensitivity).
    #
    # PRIVACY NOTE: x_l2_max and y_abs_max must be DATA-INDEPENDENT bounds to
    # preserve the DP guarantee. After MinMaxScaler, every feature is in [-T, T],
    # so the worst-case L2 row norm is sqrt(d)*T and max|y| is T — both are
    # domain properties, not data-dependent statistics.
    # Using the actual max from the training set (x_l2_max = sqrt(max row L2²))
    # is a privacy violation: it leaks information about the data for free.
    x_l2_max = float(np.sqrt(d)) * T   # domain bound: worst-case L2 row norm in [-T,T]^d
    y_abs_max = T                       # domain bound: max|y| after MinMaxScaler to [-T,T]

    # OLD (privacy bug): used data-dependent statistics
    #   x_l2_max = float(np.sqrt(np.max(np.sum(X_train**2, axis=1))))  # private!
    #   y_abs_max = float(np.max(np.abs(y_train)))                      # private!

    X_norm = X_train / x_l2_max
    y_norm = y_train / y_abs_max

    # OLD (wrong): C_max = float(np.sqrt(d + 1)) * T
    # Sensitivity of X^T@residual = max||x_i||_2 * C_max = sqrt(d)*T * sqrt(d+1)*T,
    # but we were passing only C_max — missing the sqrt(d)*T factor from ||x_i||_2.
    # After L2 normalization max||x_norm_i||_2 = 1 and max|y_norm| = 1, so C_max = 1.
    C_max = 1.0

    # zCDP -> (epsilon, delta)-DP (full budget for reference; split below)
    epsilon = zcdp_to_approxdp(rho, delta)
    eps_half = epsilon / 2.0

    # Failure probability
    varrho = delta / 10.0

    # Sketch dimension
    k = max(int(percentage_k * d),
            int(percentage_k * np.log(max(4.0 * n_iters / varrho, 2))))

    # Threshold for eigenvalue noise adjustment
    tau_mix = sqrt(2.0 * np.log(max(4.0 / delta, 4.0 / varrho)))

    # Gaussian mechanism baseline (used as init for binary search)
    init_gamma = 2.0 * np.log(1.25 / delta) / (epsilon ** 2)

    # Noise for Hessian sketching: half budget, 3/4*delta for Hessian.
    # OLD (wrong): called with raw X_train which has max L2 row norm sqrt(d)*T;
    # _solve_gamma_renyi_comp assumes unit row norm so noise was off by d*T².
    # Fixed by passing X_norm (unit max L2 row norm) to the loop below.
    gamma_matrix = _solve_gamma_renyi_comp(
        init_gamma, k,
        target_delta=3.0 * delta / 4.0,
        target_epsilon=eps_half,
        inflation=0.0,
        T=n_iters,
    )
    sigma_Hessian = gamma_matrix / sqrt(k)

    # Noise for X^T y: half budget, 1/4*delta for XTy; scaled by sqrt(n_iters) for composition.
    # C_max = 1.0 after normalization, so sensitivity = max||x_norm||_2 * C_max = 1 * 1 = 1.
    sigma_XTY = _calibrate_analytic_gaussian(eps_half, delta / 4.0, C_max)
    sigma_XTY *= sqrt(n_iters)

    # Private eigenvalue adjustment (conservative lambda_min=0 gives gamma_tilde=0)
    gamma_tilde = max(0.0, lambda_min - sigma_Hessian * (tau_mix - rng.standard_normal()))
    gamma_final = sqrt(max(0.0, gamma_matrix - gamma_tilde))

    # IHM iterations on normalized data
    theta = np.zeros(d)
    for _ in range(n_iters):
        S = rng.standard_normal((k, n))
        N = rng.standard_normal((k, d))
        X_sketch = S @ X_norm + gamma_final * N         # OLD: S @ X_train
        H_hat = X_sketch.T @ X_sketch / k
        residual = np.clip(y_norm - X_norm @ theta, -C_max, C_max)  # OLD: y_train, X_train
        XY_pr = X_norm.T @ residual + sigma_XTY * rng.standard_normal(d)  # OLD: X_train.T
        theta += np.linalg.solve(H_hat + 1e-10 * np.eye(d), XY_pr)

    # Unscale: theta_original = theta_norm * y_abs_max / x_l2_max = theta_norm * T / (sqrt(d)*T)
    return theta * (y_abs_max / x_l2_max)


def coinpress_regression(X_train, y_train, rho, T, r=None, t=1, xty_frac=0.5, rng=None):
    """CoinPress-based regression: CoinPress(t) for Sigma_XX, SSP for Sigma_Xy.

    Uses CoinPress covariance estimation on the feature block, then privately
    estimates the feature-target cross-covariance with SSP, and solves the
    regression system via lstsq.

    Args:
        X_train:   (n, d_feat+1) centered matrix with target as last column,
                   OR (n, d_feat) if y_train is provided separately.
        y_train:   (n,) target values (centered).
        rho:       total zCDP privacy budget.
        T:         L∞ per-coordinate clipping bound (1.0 after rescaling).
        r:         L2 row-norm bound for CoinPress (default sqrt(d_feat)*T).
        t:         number of CoinPress iterations (default 1).
        xty_frac:  fraction of rho for Sigma_Xy estimation (default 0.5).
        rng:       numpy random Generator.

    Returns:
        beta: (d_feat,) regression coefficients.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d_feat = X_train.shape
    if r is None:
        r = float(np.sqrt(d_feat)) * T

    rho_cov = rho * (1.0 - xty_frac)
    rho_Xy  = rho * xty_frac

    # Feature covariance via CoinPress
    Sigma_XX_dp = coinpress_cov(X_train, rho_cov, r=r, t=t)

    # Cross-covariance via SSP
    sensitivity  = cov_jk_sensitivity_linf(n, T)
    Sigma_Xy_true = X_train.T @ y_train / (n - 1)
    Sigma_Xy_dp, _ = SSP_gradient(Sigma_Xy_true, sensitivity, rho_Xy, rng=rng)

    beta, _, _, _ = np.linalg.lstsq(Sigma_XX_dp, Sigma_Xy_dp, rcond=None)
    return beta


def random_selection_baseline(rng, X, Sigma_true, d, n, rho, alpha,
                               n_offdiag, sensitivity_type, norm_bound):
    """
    Random selection baseline for covariance estimation.

    Measures all d diagonal entries (same budget as PACE-GGM round 0), then
    randomly selects n_offdiag off-diagonal entries from the lower triangle and
    measures them with the remaining budget. Reconstructs with IPM (ipm_chol).

    Budget:
        diagonal:    rho * alpha / d  per entry  (matches PACE-GGM)
        off-diagonal: rho * (1 - alpha) / n_offdiag  per entry
    """
    from privateggm.solve import solve as ggm_solve
    from privateggm.utils import mahalanobis_cov_error, frobenius_error

    X_centered = X - X.mean(axis=0)
    empirical_cov = X_centered.T @ X_centered / n

    if sensitivity_type == 'linf':
        sensitivity_diag    = cov_jk_sensitivity_linf_diag(n, norm_bound)
        sensitivity_offdiag = cov_jk_sensitivity_linf(n, norm_bound)
    else:
        sensitivity_diag    = cov_jk_sensitivity_l2(n, norm_bound)
        sensitivity_offdiag = cov_jk_sensitivity_l2(n, norm_bound)

    rho_per_diag    = rho * alpha / d
    rho_per_offdiag = rho * (1 - alpha) / n_offdiag if n_offdiag > 0 else 0.0

    Sigma_tilde  = np.zeros((d, d))
    noise_scales = np.zeros((d, d))

    # Measure diagonal
    for i in range(d):
        noisy_val, _, sigma = gaussian_mechanism(rng, empirical_cov[i, i], sensitivity_diag, rho_per_diag)
        Sigma_tilde[i, i]  = noisy_val
        noise_scales[i, i] = sigma

    # Randomly select and measure off-diagonal entries
    if n_offdiag > 0:
        lower_tri  = [(i, j) for i in range(d) for j in range(i)]
        k          = min(n_offdiag, len(lower_tri))
        chosen_idx = rng.choice(len(lower_tri), size=k, replace=False)
        for idx in chosen_idx:
            i, j = lower_tri[idx]
            noisy_val, _, sigma = gaussian_mechanism(rng, empirical_cov[i, j], sensitivity_offdiag, rho_per_offdiag)
            Sigma_tilde[i, j]  = noisy_val
            Sigma_tilde[j, i]  = noisy_val
            noise_scales[i, j] = sigma
            noise_scales[j, i] = sigma

    # Solve with ipm
    measured = noise_scales > 0
    rows, cols = np.where(measured)
    vals = Sigma_tilde[rows, cols]
    tau  = noise_scales[rows, cols]

    try:
        Sigma_est, _ = ggm_solve(
            rows, cols, vals, tau, d, solver='ipm_chol',
            mu_init=1.0, mu_factor=0.5, eps=1e-8,
            max_outer=40, max_inner=200, decompose='blocks'
        )
    except Exception as e:
        print(f"  WARNING: random_selection_baseline IPM failed ({e}), using empirical cov")
        Sigma_est = empirical_cov

    return {
        'mahalanobis_error': float(mahalanobis_cov_error(Sigma_true, Sigma_est)),
        'frobenius_error':   float(frobenius_error(Sigma_true, Sigma_est)),
    }
