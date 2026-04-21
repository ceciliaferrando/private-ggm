import numpy as np
from privateggm.privacy import cov_jk_sensitivity_linf, SSP_gradient, gaussian_mechanism


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
