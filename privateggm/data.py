import numpy as np
import pandas as pd
import os
import json
import time
from sklearn.preprocessing import MinMaxScaler

def _is_numeric_string(s):
    """Check if a string represents a number (int or float)"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def load_data(data_dir, data_name="housing", target_column=None, drop_columns=None):
    """
    Load a dataset from a CSV file, optionally excluding a target variable and other columns.
    Automatically handles preprocessing: keeps only numerical columns, removes rows with NaNs.
    
    Args:
        data_dir (str): Directory containing the CSV file
        data_name (str): Name of the dataset (without .csv extension). Defaults to 'housing'.
        target_column (str, optional): Name of the target variable column to exclude. 
                                     For regression datasets, this column will be dropped
                                     as we don't want it in the covariance estimation.
        drop_columns (list, optional): List of additional column names to drop 
                                     (e.g., categorical variables, IDs, etc.)
                                     If None, all non-numerical columns are automatically dropped,
                                     along with common identifier columns (Order, PID, ID, etc.).
        
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame, cleaned and ready for use.
                     Only numerical columns, no missing values, specified columns dropped.
    """
    data_path = os.path.join(data_dir, f"{data_name}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_name} dataset not found at {data_path}")
    
    # Auto-detect if CSV has a header by checking if first row is numeric
    header = 'infer'  # Default behavior
    first_line = None
    for encoding in ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']:
        try:
            with open(data_path, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
            break
        except Exception:
            continue
    
    if first_line is not None:
        # Try to parse first line as floats - if all numeric, likely data not header
        try:
            values = first_line.split(',')
            all_numeric = all(
                _is_numeric_string(v.strip()) for v in values if v.strip()
            )
            if all_numeric:
                print(f"Detected headerless CSV for {data_name}, loading with header=None")
                header = None
        except:
            pass
    
    # Try different encodings
    for encoding in ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv(data_path, encoding=encoding, header=header)
            if encoding != 'utf-8':
                print(f"Successfully loaded {data_name} with encoding: {encoding}")
            break
        except Exception:
            continue
    else:
        raise UnicodeDecodeError(f"Could not decode {data_path} with any common encoding")
    
    print(f"Loaded {data_name}: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # If drop_columns is None, automatically drop all non-numerical columns
    if drop_columns is None:
        # Keep only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numerical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numerical_cols:
            print(f"Auto-dropping {len(non_numerical_cols)} non-numerical columns: {non_numerical_cols}")
            df = df[numerical_cols]
        
        # Also drop common identifier columns
        id_patterns = ['order', 'pid', 'id', 'index', 'year', 'date']
        id_cols_to_drop = []
        for col in df.columns:
            # Convert to string in case column name is an integer (headerless CSV)
            col_str = str(col).lower().replace('_', '').replace(' ', '')
            if col_str in id_patterns or col_str.endswith('id'):
                id_cols_to_drop.append(col)
        
        if id_cols_to_drop:
            print(f"Auto-dropping {len(id_cols_to_drop)} identifier-like columns: {id_cols_to_drop}")
            df = df.drop(columns=id_cols_to_drop)
    else:
        # User specified columns to drop
        columns_to_drop = []
        for col in drop_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in dataset, skipping")
            else:
                columns_to_drop.append(col)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
    
    # Drop target column if specified
    if target_column is not None:
        if target_column not in df.columns:
            print(f"WARNING: Target column '{target_column}' not found in dataset! Already removed by auto-drop?")
            print(f"  Available columns: {list(df.columns)}")
        else:
            print(f"Dropping target column: '{target_column}'")
            before_drop = df.shape
            df = df.drop(columns=[target_column])
            after_drop = df.shape
            print(f"  Shape before: {before_drop}, after: {after_drop}")
    else:
        print(f"No target column specified")
    
    # Drop rows with any missing values
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    if rows_before > rows_after:
        print(f"Dropped {rows_before - rows_after} rows with missing values ({rows_before} → {rows_after})")
    
    # Remove constant columns (zero variance) which cause singular matrix issues
    constant_cols = [col for col in df.columns if df[col].std() == 0]
    if constant_cols:
        print(f"Auto-dropping {len(constant_cols)} constant columns (zero variance): {constant_cols}")
        df = df.drop(columns=constant_cols)
    
    print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df

def compute_and_save_bounds(data_df, outdir, rescale=True):
    """
    Compute L2 and Linf bounds for a dataset and save them to a JSON file.
    
    PRIVACY WARNING: This function computes bounds from the actual data,
    which may leak information. In a production DP setting, bounds should be
    provided by the data analyst based on domain knowledge, not computed
    from the data itself.
    
    Args:
        data_df (pd.DataFrame): Input dataset
        outdir (str): Directory to save the bounds file
        rescale (bool): Whether to rescale the data to [-1,1] before computing bounds
        
    Returns:
        str: Path to the saved bounds file
    """

    if rescale:
        data_df = rescale_data(data_df)
        L2_bound = np.sqrt(data_df.shape[1])  # Same as synthetic case
        Linf_bound = 1  # Due to [-1,1] rescaling
        print(f"   Using analytical bounds after rescaling: L2={L2_bound:.3f}, Linf={Linf_bound}")
    else:
        X = data_df.to_numpy()
        # Get L2 bound on row entries
        L2_bound = np.linalg.norm(X, axis=1).max()
        # Get L_infinity bound on row entries
        Linf_bound = np.max(np.abs(X))
        print(f"   Computed empirical bounds from data: L2={L2_bound:.3f}, Linf={Linf_bound:.3f}")
        print("⚠️  WARNING: Computing bounds from data itself - not DP-compliant!")
        print("   In practice, bounds should be provided by analyst based on domain knowledge.")
    
    
    bounds_path = os.path.join(outdir, f"L_bounds_rescale{rescale}.json")
    d = data_df.shape[1]
    
    # Load existing bounds if file exists and is non-empty
    bounds = {}
    if os.path.exists(bounds_path) and os.path.getsize(bounds_path) > 0:
        try:
            with open(bounds_path, "r") as f:
                bounds = json.load(f)
        except (json.JSONDecodeError, ValueError):
            # File corrupted or empty, start fresh
            bounds = {}
    
    bounds[str(d)] = {"l2": float(L2_bound), "linf": float(Linf_bound)}
    
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=4)
        f.flush()
        os.fsync(f.fileno())  # Ensure write is synced to disk on NFS
    
    # Small delay for NFS to fully propagate changes
    time.sleep(0.1)
        
    return bounds_path

def generate_well_conditioned_gaussian(rng, outdir, n_samples, d, rescale, cond_max=10.0):
    """
    Generates synthetic multivariate Gaussian data with a well-conditioned SPD covariance matrix.
    
    DP-COMPLIANT: Bounds are computed analytically from known data distribution,
    not from the actual data values, preserving differential privacy.

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

    random_matrix = rng.random((d, d))
    # Perform QR decomposition to get a random orthogonal matrix
    Q, _ = np.linalg.qr(random_matrix)
    eigvals = np.exp(rng.random(d))  # Ensure positive eigenvalues
    eigvals = np.clip(eigvals, 1.0 / cond_max, cond_max)  # Restrict condition number
    Sigma = Q @ np.diag(eigvals) @ Q.T  # Construct SPD covariance matrix

    # Generate and save multivariate Gaussian data
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(d)])

    if rescale:
        df = rescale_data(df)
        L2_bound = np.sqrt(d)
        Linf_bound = 1
        print(f"Analytical bounds: L2={L2_bound:.3f}, Linf={Linf_bound}")

    #print("saving this data:", df)

    data_path = os.path.join(outdir, f"synthdata{d}rescale{rescale}.csv")
    df.to_csv(data_path, index=False)

    if not rescale:
        # Re-simulate the data to get data bound
        surrogate_X = rng.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)

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

    #print("bounds", L2_bound, Linf_bound)

    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=4)

    #print(f"Data saved to {data_path}")
    #print(f"L2 and Linf values for {d}-dimensional data saved to {bounds_path}")

    return df


def rescale_data(data):
    """
    attribute rescaling between -1 and 1 needed for sensitivity analysis
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    return scaled_df


def load_and_process_data(data_path, bounds_path, clip_type="linf", rescale=None):
    """
    Load and preprocess the data from a CSV file, and compute relevant quantities.

    Parameters:
        data_path (str): Path to the CSV dataset.
        bounds_path (str): Path to the JSON file containing the bound values.
        clip_type (str): Which norm bound to use for clipping, either "l2" or "linf".
        rescale (bool): Whether the data is rescaled. Must match the bounds file used.
                       If None, will infer from bounds_path filename.

    Returns:
        data (pd.DataFrame): Preprocessed and optionally clipped data.
        n (int): Number of rows (samples).
        d (int): Number of columns (attributes).
        Sigma_true (np.array): Covariance matrix of the data.
        T (float): The clipping bound used (either ℓ₂ bound or ℓ∞ bound).
    """
    # Infer rescale from bounds_path if not provided
    if rescale is None:
        if "rescaleTrue" in bounds_path:
            rescale = True
        elif "rescaleFalse" in bounds_path:
            rescale = False
        else:
            raise ValueError(f"Cannot infer rescale setting from bounds_path: {bounds_path}. "
                           "Please specify rescale=True or rescale=False explicitly.")
    
    # Validate bounds file matches rescale setting
    expected_bounds_file = f"L_bounds_rescale{rescale}.json"
    if expected_bounds_file not in bounds_path:
        raise ValueError(f"Bounds file mismatch! rescale={rescale} but bounds_path={bounds_path}. "
                       f"Expected bounds file to contain '{expected_bounds_file}'")
    
    # Load the data from CSV.
    data = pd.read_csv(data_path)
    n, d = data.shape

    # Load bound values from the JSON file with retry logic for NFS
    bounds = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(bounds_path, "r") as f:
                bounds = json.load(f)
            break  # Success
        except (json.JSONDecodeError, FileNotFoundError) as e:
            if attempt < max_retries - 1:
                print(f"   Warning: Failed to load bounds on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(0.2)  # Wait and retry (NFS sync issue)
            else:
                raise ValueError(f"Failed to load bounds from {bounds_path} after {max_retries} attempts: {e}")
    
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
    
    # Sanity check: if rescaled, data should be approximately in [-1, 1]
    if rescale:
        data_min, data_max = X.min(), X.max()
        if data_min < -1.1 or data_max > 1.1:
            raise ValueError(f"Data range [{data_min:.3f}, {data_max:.3f}] is outside expected "
                           f"rescaled range [-1, 1]. Did you forget to rescale the data?")
        if clip_type.lower() == "linf" and T != 1.0:
            import warnings
            warnings.warn(f"Using rescale=True but Linf bound is {T}, expected 1.0. "
                        "This may indicate a bounds file mismatch.")

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
    
    # Compute the covariance matrix from the CLIPPED data.
    Sigma_true = data.cov().to_numpy()

    # print(f"Data loaded from {data_path}")
    # print(f"Dimension: {d}, Number of samples: {n}")
    # print(f"Using {clip_type.upper()} clipping with bound T = {T}")
    # print("Covariance matrix computed and data clipped.")

    return data, n, d, Sigma_true, T
