import sys
import numpy as np
from types import SimpleNamespace
from privateggm.privacy import cov_jk_sensitivity_linf, SSP_gradient, gaussian_mechanism

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
