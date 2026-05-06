import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from privateggm.privacy import SSP_covariance, cov_sensitivity_linf, cov_jk_sensitivity_linf, SSP_gradient, privatize_mean


def prepare_regression_split(df, seed, test_size=0.2, mu_dp=None):
    """
    Split df into train/test, fit MinMaxScaler on train only, center data.

    The target must be the LAST column of df.

    If mu_dp is provided (a privately released mean vector), centering is done
    with mu_dp instead of the empirical mean.  The test set is always centered
    with the same mu used for training so that predictions are consistent.

    Returns:
        X_train_c  : (n_train, d) centered scaled train features+target
        reg_data   : dict with test arrays, scaler, and centering info
        Sigma_true : empirical covariance of centered train data (using mu_dp if given)
        T          : linf clipping bound (1.0 after [-1,1] rescaling)
        d          : total dimension including target
        n_train    : number of training samples
        mu_used    : (d,) mean vector actually used for centering
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_total = len(df)
    n_train = int((1 - test_size) * n_total)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test  = df.iloc[n_train:].reset_index(drop=True)

    # Fit scaler on train only, apply to test
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(df_train)
    test_scaled  = scaler.transform(df_test)

    # Center: use private mean if provided, else empirical mean
    mu_emp = train_scaled.mean(axis=0)
    mu_used = mu_dp if mu_dp is not None else mu_emp

    X_train_c = train_scaled - mu_used
    X_test_c  = test_scaled  - mu_used

    # Sigma_true from centered train — use second moment (no internal mean subtraction)
    # so it is consistent with the centering above regardless of mu_used
    Sigma_true = X_train_c.T @ X_train_c / (n_train - 1)

    # After [-1,1] rescaling, linf bound is still 1
    T = 1.0
    d = df.shape[1]

    reg_data = {
        'X_test_feat':  X_test_c[:, :-1],   # features only, centered
        'y_test':       X_test_c[:, -1],    # target, centered
        'train_scaled': train_scaled,        # uncentered scaled train (d,) — for per-trial re-centering
        'test_scaled':  test_scaled,         # uncentered scaled test  (d,)
        'mu_used':      mu_used,
        'mu_emp':       mu_emp,
        'scaler':       scaler,
        'columns':      list(df.columns),
    }

    return X_train_c, reg_data, Sigma_true, T, d, n_train, mu_used


def extract_regression_coefficients(Sigma, d):
    """
    Extract OLS regression coefficients from joint covariance of [X, y].

    Sigma is (d x d) where the last row/col corresponds to the target y.
    Uses least-squares solve to handle near-singular Sigma_XX gracefully.

    Returns:
        beta : (d-1,) coefficient vector
    """
    Sigma_XX = Sigma[:-1, :-1]   # (d-1) x (d-1)
    Sigma_Xy = Sigma[:-1, -1]    # (d-1,)
    beta, _, _, _ = np.linalg.lstsq(Sigma_XX, Sigma_Xy, rcond=None)
    return beta


def SSP_regression_XTy(X_train_c, rho, target_frac, T, rng):
    """
    SSP baseline with dedicated budget for X^Ty (cross-covariance with target).

    Splits the privacy budget into:
      - rho_XX = (1 - target_frac) * rho  ->  Gaussian noise on Sigma_XX (features only)
      - rho_Xy = target_frac * rho         ->  Gaussian noise on Sigma_Xy (feature-target cross-cov)

    Sensitivity of each Sigma_Xy entry is the same as cov_jk_sensitivity_linf:
      Delta = 2 * T^2 / n  (L-inf, add/remove one observation).

    Returns:
        Sigma_XX_dp : (d-1, d-1) noisy feature covariance
        Sigma_Xy_dp : (d-1,) noisy cross-covariance vector
        beta        : (d-1,) regression coefficients from lstsq
        noise_scale_XX : float
        noise_scale_Xy : float
    """
    n, d = X_train_c.shape
    X_feat = X_train_c[:, :-1]   # (n, d-1)
    y_vec  = X_train_c[:, -1]    # (n,)

    # Empirical statistics (n-1 denominator, no mean subtraction — X_train_c must be pre-centered).
    # Using the second moment directly ensures consistency with the centering used by the caller
    # (empirical mean or private mean). np.cov would re-center with the sample mean, which
    # undoes private-mean centering and creates a mismatch with Sigma_Xy.
    Sigma_XX_true = X_feat.T @ X_feat / (n - 1)            # (d-1, d-1)
    Sigma_Xy_true = X_feat.T @ y_vec / (n - 1)             # (d-1,)

    # Budget split
    rho_XX = (1.0 - target_frac) * rho
    rho_Xy = target_frac * rho

    # Sigma_XX: full SSP with linf sensitivity 
    sensitivity_XX = cov_sensitivity_linf(n, d - 1, T)
    Sigma_XX_dp, noise_scale_XX = SSP_covariance(
        Sigma_XX_true, sensitivity_XX, rho_XX, rng=rng
    )

    # Sigma_Xy: entry-wise Gaussian noise, same linf sensitivity as cov_jk 
    sensitivity_Xy = cov_jk_sensitivity_linf(n, T)          # 2 * T^2 / n
    Sigma_Xy_dp, noise_scale_Xy = SSP_gradient(
        Sigma_Xy_true, sensitivity_Xy, rho_Xy, rng=rng
    )

    # Regression coefficients via lstsq
    beta, _, _, _ = np.linalg.lstsq(Sigma_XX_dp, Sigma_Xy_dp, rcond=None)

    print(f"  SSP XTy — noise_scale_XX={noise_scale_XX:.4e}, noise_scale_Xy={noise_scale_Xy:.4e}")

    return Sigma_XX_dp, Sigma_Xy_dp, beta, noise_scale_XX, noise_scale_Xy


def regression_mse(X_test_feat, y_test, beta, mu_y=0.0):
    """
    Compute MSE on centered/scaled test features using coefficient vector beta.

    If mu_y is provided (the privately released mean of y on the training set),
    it is added back to the prediction:
        y_pred = mu_y + beta^T (x - mu_x)
    where x is already centered by mu_x (done in prepare_regression_split),
    so the call reduces to:
        y_pred = mu_y + X_test_feat @ beta

    Without private mean (mu_y=0.0) this is equivalent to the old behaviour.

    Args:
        X_test_feat : (n_test, d-1) centered scaled test features
        y_test      : (n_test,) centered scaled test targets
        beta        : (d-1,) regression coefficients
        mu_y        : float, private mean of target (default 0.0 = no correction)

    Returns:
        mse : float
    """
    y_pred = mu_y + X_test_feat @ beta
    return float(np.mean((y_test - y_pred) ** 2))
