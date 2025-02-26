import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_well_conditioned_gaussian(n_samples=100, d=4, cond_max=10.0):
    """
    Generates synthetic multivariate Gaussian data with a well-conditioned SPD covariance matrix.

    Args:
    - n_samples: Number of data points.
    - d: Dimensionality.
    - cond_max: Maximum allowed condition number.
    - filename: If provided, saves data to CSV.

    Returns:
    - X: (n_samples, d) matrix of synthetic data.
    - mu: Mean vector (d,).
    - Sigma: Well-conditioned covariance matrix (d, d).
    """
    # Generate a random mean vector
    mu = np.random.randn(d)

    # Generate a random SPD covariance matrix using Q Lambda Q^T
    Q, _ = np.linalg.qr(np.random.randn(d, d))  # Random orthogonal matrix
    eigvals = np.exp(np.random.randn(d))  # Ensure positive eigenvalues
    eigvals = np.clip(eigvals, 1.0 / cond_max, cond_max)  # Restrict condition number
    Sigma = Q @ np.diag(eigvals) @ Q.T  # Construct SPD covariance matrix

    # Generate and save multivariate Gaussian data
    X = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(d)])
    df.to_csv(f"synthetic_data_{d}.csv", index=False)

    return X, mu, Sigma


def preprocess(data):
    """
    attribute rescaling between -1 and 1 needed for sensitivity analysis
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    return scaled_df


def load_and_process_data(data_path):
    """
    Load and preprocess the data from a CSV file, and compute relevant quantities.

    Parameters:
        data_path (str): Path to the CSV dataset.

    Returns:
        data (pd.DataFrame): Preprocessed data.
        n (int): Number of rows (samples).
        d (int): Number of columns (attributes).
        Sigma_true (np.array): Covariance matrix of the data.
        T (float): Maximum ℓ₂ norm over the rows of the data.
    """
    import pandas as pd
    import numpy as np

    # Read and preprocess the data
    data = preprocess(pd.read_csv(data_path))
    n, d = data.shape

    # Compute the covariance matrix
    Sigma_true = data.cov().to_numpy()

    # Compute the maximum ℓ2 norm among the rows
    T = np.linalg.norm(data.to_numpy(), axis=1).max()

    return data, n, d, Sigma_true, T