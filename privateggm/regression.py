import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_regression_split(df, seed, test_size=0.2):
    """
    Split df into train/test, fit MinMaxScaler on train only, center using train mean.

    The target must be the LAST column of df.

    Returns:
        X_train : (n_train, d) centered scaled train features+target — passed to PACE-GGM/SSP
        reg_data : dict with test arrays and scaler info for evaluation
        Sigma_true : empirical covariance of centered train data
        T : linf clipping bound (1.0 after [-1,1] rescaling)
        d : total dimension including target
        n_train : number of training samples
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

    # Center using train mean (same assumption already made in model.py)
    mu_train = train_scaled.mean(axis=0)
    X_train_c = train_scaled - mu_train
    X_test_c  = test_scaled  - mu_train

    # Sigma_true from centered train
    Sigma_true = np.cov(X_train_c.T)

    # After [-1,1] rescaling and centering, linf bound is still 1
    T = 1.0
    d = df.shape[1]

    reg_data = {
        'X_test_feat': X_test_c[:, :-1],   # features only
        'y_test':      X_test_c[:, -1],    # target (centered scaled)
        'mu_train':    mu_train,
        'scaler':      scaler,
        'columns':     list(df.columns),
    }

    return X_train_c, reg_data, Sigma_true, T, d, n_train


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


def regression_mse(X_test_feat, y_test, beta):
    """
    Compute MSE on centered/scaled test features using coefficient vector beta.

    Args:
        X_test_feat : (n_test, d-1) centered scaled test features
        y_test      : (n_test,) centered scaled test targets
        beta        : (d-1,) regression coefficients

    Returns:
        mse : float
    """
    y_pred = X_test_feat @ beta
    return float(np.mean((y_test - y_pred) ** 2))
