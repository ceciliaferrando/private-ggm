import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import MinMaxScaler

def generate_well_conditioned_gaussian(outdir, n_samples, d, rescale, cond_max=10.0):
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
    mu = np.zeros((d,))

    # Generate a random SPD covariance matrix using Q Lambda Q^T
    Q, _ = np.linalg.qr(np.random.randn(d, d))  # Random orthogonal matrix
    eigvals = np.exp(np.random.randn(d))  # Ensure positive eigenvalues
    eigvals = np.clip(eigvals, 1.0 / cond_max, cond_max)  # Restrict condition number
    Sigma = Q @ np.diag(eigvals) @ Q.T  # Construct SPD covariance matrix

    # Generate and save multivariate Gaussian data
    X = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(d)])

    if rescale:
        df = rescale_data(df)
        L2_bound = 2 * np.sqrt(d)
        print("L2 bound at origin", L2_bound)
        Linf_bound = 1

    print("saving this data:", df)

    data_path = os.path.join(outdir, f"synthdata{d}rescale{rescale}.csv")
    df.to_csv(data_path, index=False)

    if not rescale:
        # Re-simulate the data to get data bound
        surrogate_X = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)

        # get L2 bound on row entries of surrogate_X
        L2_bound = np.linalg.norm(surrogate_X, axis=1).max()

        # get L_infinity bound on row entries of surrogate_X
        Linf_bound = np.max(np.abs(surrogate_X))

    # Save L2 and L_infinity values to a file (e.g., in a JSON file)
    bounds_path = os.path.join(outdir, f"L_bounds_rescale{rescale}.json")
    if os.path.exists(bounds_path):
        with open(bounds_path, "r") as f:
            bounds = json.load(f)
    else:
        bounds = {}

    bounds[str(d)] = {"l2": float(L2_bound), "linf": float(Linf_bound)}

    print("bounds", L2_bound, Linf_bound)

    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=4)

    print(f"Data saved to {data_path}")
    print(f"L2 and Linf values for {d}-dimensional data saved to {bounds_path}")

    return df


def rescale_data(data):
    """
    attribute rescaling between -1 and 1 needed for sensitivity analysis
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    return scaled_df


def load_and_process_data(data_path, bounds_path, clip_type="linf"):
    """
    Load and preprocess the data from a CSV file, and compute relevant quantities.

    Parameters:
        data_path (str): Path to the CSV dataset.
        bounds_path (str): Path to the JSON file containing the bound values.
        clip_type (str): Which norm bound to use for clipping, either "l2" or "linf".

    Returns:
        data (pd.DataFrame): Preprocessed and optionally clipped data.
        n (int): Number of rows (samples).
        d (int): Number of columns (attributes).
        Sigma_true (np.array): Covariance matrix of the data.
        T (float): The clipping bound used (either ℓ₂ bound or ℓ∞ bound).
    """
    # Load the data from CSV.
    data = pd.read_csv(data_path)
    n, d = data.shape

    # Compute the covariance matrix.
    Sigma_true = data.cov().to_numpy()

    # Load bound values from the JSON file.
    with open(bounds_path, "r") as f:
        bounds = json.load(f)
    # We assume the key for the d-dimensional bound is stored as str(d).
    if str(d) not in bounds:
        raise ValueError(f"No bound values found for d = {d} in {bounds_path}.")

    # Select the desired bound.
    if clip_type.lower() == "l2":
        T = bounds[str(d)]["l2"]
    elif clip_type.lower() == "linf":
        T = bounds[str(d)]["linf"]
    else:
        raise ValueError("clip_type must be either 'l2' or 'linf'")

    # Convert the DataFrame to a numpy array.
    X = data.to_numpy()

    # Clip the data according to the chosen norm.
    if clip_type.lower() == "l2":
        # For each row, compute its l2 norm and scale it if needed
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Scale rows where norm > T.
        X_clipped = np.where(norms > T, X * (T / norms), X)
    elif clip_type.lower() == "linf":
        # clip each element to lie in [-T, T].
        X_clipped = np.clip(X, -T, T)

    # Replace the data in the DataFrame with the clipped values.
    data = pd.DataFrame(X_clipped, columns=data.columns)

    print(f"Data loaded from {data_path}")
    print(f"Dimension: {d}, Number of samples: {n}")
    print(f"Using {clip_type.upper()} clipping with bound T = {T}")
    print("Covariance matrix computed and data clipped.")

    return data, n, d, Sigma_true, T
