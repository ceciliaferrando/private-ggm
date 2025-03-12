import numpy as np
import pandas as pd
import os
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import sqrtm

import matplotlib.pyplot as plt

DEBUG_MODE = True

def get_K(z, inds, d):
    """
    z [jnp.array]: vector supporting precision matrix estimate K(z)
    inds [np.array]: list of the lower triangular nonzero element indices in Sigma_tilde
    d: dimension of Sigma_tilde
    """

    rows = inds[:, 0]
    cols = inds[:, 1]

    K = jnp.zeros((d, d))

    # Write lower triangle
    K = K.at[rows, cols].add(z)

    # Write upper triangle (diagonal elements are counted twice, see initialize_z)
    K = K.at[cols, rows].add(z)
    # jax.debug.print("K: {}", K)

    return K


def get_z(K_z, inds):
    rows = inds[:, 0]
    cols = inds[:, 1]

    # extract vectorized entries from K_z
    z = K_z[rows, cols]

    # correct diagonal elements (they were added twice in get_K)
    diag_mask = rows == cols
    z = z.at[diag_mask].set(z[diag_mask] / 2)

    return z


def initialize_z(Sigma_tilde, inds):
    """
        Sigma_tilde [np.array]: matrix of observed (privatized) covariance entries
        inds [np.array]: vector of indices of nonzero covariance entries
    """

    d = Sigma_tilde.shape[0]

    z_init = []

    for (i, j) in inds:
        if i == j:
            z_init.append(max(0.01, 1. / Sigma_tilde[i, j]) / 2)  # diagonal is halved to match K construction in get_K
        else:
            z_init.append(0)

    return jnp.array(z_init)


def project_to_PSD(Sigma, thresh=1e-4):
    eigvals, eigvecs = jnp.linalg.eigh(Sigma)
    eigvals = jnp.maximum(eigvals, thresh)  # ensure a minimum positive threshold
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
    # w, V = np.linalg.eigh(Sigma)
    # w[w < thresh] = thresh
    # return (V * w).dot(V)


def get_observed_Sigma_entries(Sigma):
    """
    extract the indices of the observed entries in Sigma
    """

    d = Sigma.shape[0]

    diag_inds = [(i, i) for i in range(d)]  # assumed it's always observed
    tril_inds = get_strict_lower_triangle_nonzero(Sigma)
    triu_inds = [(j, i) for (i, j) in tril_inds]

    # all_observed_inds = diag_inds + tril_inds + triu_inds
    all_observed_inds = diag_inds + tril_inds

    return np.array(all_observed_inds)


def compute_d(inds):
    mask = (inds[:, 0] == inds[:, 1])
    d = np.sum(mask)
    return d


def reconstruct_sigma_tilde(inds, sigma_tilde_vec, d):
    Sigma_tilde = np.zeros((d, d))

    rows = inds[:, 0]
    cols = inds[:, 1]

    # Populate the matrix at the positions indicated by inds
    Sigma_tilde[rows, cols] = sigma_tilde_vec
    Sigma_tilde[cols, rows] = sigma_tilde_vec  # Fill in the symmetric counterparts

    return Sigma_tilde


def is_positive_definite(M):
    return jnp.all(jnp.linalg.eigvalsh(M) > 0)


def is_positive_semidefinite(M):
    return jnp.all(jnp.linalg.eigvalsh(M) >= 0)


def get_strict_lower_triangle_nonzero(M):
    # lower triangle indices excluding the diagonal
    lower_triangle_indices = np.tril_indices(M.shape[0], k=-1)

    # nonzero elements in the lower triangle
    nonzero_mask = M[lower_triangle_indices] != 0
    filtered_indices = (lower_triangle_indices[0][nonzero_mask], lower_triangle_indices[1][nonzero_mask])

    # zip indices together into tuples
    indices_tuples = list(zip(filtered_indices[0].tolist(), filtered_indices[1].tolist()))

    return indices_tuples


@jax.jit
def logdet_psd(A):
    """
    computes log determinand and determines if A is PSD
    """
    # compute the Cholesky decomposition
    L = jnp.linalg.cholesky(A)

    # Check if any entry of the Cholesky factor is NaN
    is_nan = jnp.any(jnp.isnan(L))

    def nan_case(_):
        return -jnp.inf

    def valid_case(L):
        diag_L = jnp.diag(L)
        log_det = 2 * jnp.sum(jnp.log(diag_L))
        return log_det

    # Use jax.cond to choose between the cases
    return jax.lax.cond(is_nan, nan_case, valid_case, L)


def generate_chain_structure(Sigma_true):
    """
    Generate a chain structure where only the diagonal and adjacent subdiagonal entries are non-zero.

    Parameters:
    - Sigma_true: np.array, true covariance matrix.

    Returns:
    - Sigma_tilde: np.array, covariance matrix with chain structure.
    """
    d = Sigma_true.shape[0]
    Sigma_tilde = np.zeros_like(Sigma_true)
    np.fill_diagonal(Sigma_tilde, np.diagonal(Sigma_true))

    # Set adjacent subdiagonal entries
    for i in range(1, d):
        Sigma_tilde[i, i - 1] = Sigma_true[i, i - 1]
        Sigma_tilde[i - 1, i] = Sigma_true[i - 1, i]

    return Sigma_tilde


import numpy as np

def generate_random_offdiag_structure(Sigma_true, percentage=0.5):
    """
    Generate a structure where the cov matrix has its diagonal
    entries preserved and a percentage of the lower triangle
    off-diagonal entries (randomly selected) retained, with the remaining
    off-diagonal entries set to zero. The lower triangle is then mirrored
    to the upper triangle to ensure symmetry

    Parameters:
    - Sigma_true: np.array, true covariance matrix.
    - percentage: float, percentage of lower triangle off-diagonal entries to retain (default: 0.5).

    Returns:
    - Sigma_tilde: np.array, covariance matrix with random off-diagonals.
    """
    d = Sigma_true.shape[0]
    Sigma_tilde = np.zeros_like(Sigma_true)

    # Keep the diagonal entries as is
    np.fill_diagonal(Sigma_tilde, np.diag(Sigma_true))

    # Create a mask for the lower triangle (excluding the diagonal)
    lower_triangle_mask = np.tril(np.ones((d, d), dtype=bool), k=-1)

    # Flatten the lower triangle mask and get the indices of True values
    lower_triangle_indices = np.where(lower_triangle_mask.flatten())[0]

    # Calculate the number of lower triangle off-diagonal elements to retain
    num_lower_triangle = len(lower_triangle_indices)
    num_to_retain = int(percentage * num_lower_triangle)

    # randomly select indices to retain
    retain_indices = np.random.choice(lower_triangle_indices, size=num_to_retain, replace=False)

    # Create a flattened version of Sigma_tilde and assign the retained values
    Sigma_tilde_flat = Sigma_tilde.flatten()
    Sigma_tilde_flat[retain_indices] = Sigma_true.flatten()[retain_indices]

    # Reshape back to the original matrix shape
    Sigma_tilde = Sigma_tilde_flat.reshape(d, d)

    # mirror the lower triangle to the upper triangle to ensure symmetry
    Sigma_tilde = np.triu(Sigma_tilde.T) + np.tril(Sigma_tilde, k=-1)  # don't count diag twice

    return Sigma_tilde


def generate_custom_structure(Sigma_true, structure_dict):
    """
    Generate a custom structure based on the provided structure dictionary.

    Parameters:
    - Sigma_true: np.array, true covariance matrix.
    - structure_dict: dict, specifies which elements to keep non-zero.

    Returns:
    - Sigma_tilde: np.array, covariance matrix with custom structure.
    """
    Sigma_tilde = np.zeros_like(Sigma_true)
    np.fill_diagonal(Sigma_tilde, np.diagonal(Sigma_true))

    if not structure_dict:
        return Sigma_tilde  # return just the diagonal

    # Set custom entries
    for (i, j) in structure_dict.get('off_diag_positions', []):
        Sigma_tilde[i, j] = Sigma_true[i, j]
        Sigma_tilde[j, i] = Sigma_true[j, i]

    return Sigma_tilde


def compute_mses(inds, Sigma_true, Sigma_opt):
    rows, cols = inds[:, 0], inds[:, 1]

    # mse at positions in inds (observed positions)
    Sigma_true_inds = Sigma_true[rows, cols]
    Sigma_opt_inds = Sigma_opt[rows, cols]
    mse_inds = np.mean((Sigma_true_inds - Sigma_opt_inds) ** 2)

    # mse at positions not in inds
    mask = np.ones(Sigma_true.shape, dtype=bool)
    mask[rows, cols] = False  # Set observed positions to False

    Sigma_true_not_inds = Sigma_true[mask]
    Sigma_opt_not_inds = Sigma_opt[mask]
    mse_not_inds = np.mean((Sigma_true_not_inds - Sigma_opt_not_inds) ** 2)

    # overall mse across all positions
    mse_all = np.mean((Sigma_true - Sigma_opt) ** 2)

    # Return a dictionary with all the computed MSEs
    return {
        'mse_inds': mse_inds,
        'mse_not_inds': mse_not_inds,
        'mse_all': mse_all
    }


def plot_mses_over_iterations(results_dict):
    """
    Plot the different MSEs over iterations from the results of run_iterative_mode.

    Parameters:
    results_dict (dict): A dictionary containing MSEs and other results from run_iterative_mode.
    """
    mses_vec = results_dict['mses']

    mse_inds_list = []
    mse_not_inds_list = []
    mse_all_list = []

    # extract mse values for each iteration
    for mses in mses_vec:
        mse_inds_list.append(mses['mse_inds'])
        mse_not_inds_list.append(mses['mse_not_inds'])
        mse_all_list.append(mses['mse_all'])

    x_ticks = [str(i) for i in range(len(mse_inds_list))]

    plt.figure(figsize=(10, 6))
    plt.plot(x_ticks, mse_inds_list, label='MSE at observed positions (inds)', marker='o')
    plt.plot(x_ticks, mse_not_inds_list, label='MSE at unobserved positions (not inds)', marker='o')
    plt.plot(x_ticks, mse_all_list, label='Overall MSE (all positions)', marker='o')
    plt.title('MSEs over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    plt.show()


# Toggle this to enable/disable debug logs

def debug_print(message):
    """Centralized debug print function."""
    if DEBUG_MODE:
        print(message)


def check_matrix_stability(K_z, iteration):
    """Checks and prints stability metrics of K_z."""
    eigvals = np.linalg.eigvalsh(K_z)
    min_eigval = np.min(eigvals)
    max_eigval = np.max(eigvals)
    cond_number = max_eigval / min_eigval if min_eigval > 1e-10 else np.inf

    debug_print(f"Iteration {iteration}: Min Eigval: {min_eigval:.4e}, "
                f"Max Eigval: {max_eigval:.4e}, Condition Number: {cond_number:.2e}")

    return min_eigval, cond_number

def mahalanobis_cov_error(Sigma, Sigma_hat):
    """
    Computes the Mahalanobis distance-based error for covariance matrices.

    Args:
    - Sigma: True covariance matrix (d, d).
    - Sigma_hat: Estimated covariance matrix (d, d).

    Returns:
    - Frobenius norm of the Mahalanobis-adjusted error.
    """
    Sigma_inv_sqrt = np.linalg.inv(sqrtm(Sigma))
    error_matrix = Sigma_inv_sqrt @ Sigma_hat @ Sigma_inv_sqrt - np.eye(Sigma.shape[0])
    fro_norm = np.linalg.norm(error_matrix, ord="fro")

    return fro_norm


# def save_results(MaxEnt_results, SSP_results, dataset_name):
#     """
#     Saves Mahalanobis errors for MaxEnt and SSP to a JSON file.
#     Also saves a plot comparing the errors.
#
#     Args:
#     - MaxEnt_results: List of Mahalanobis errors for MaxEnt.
#     - SSP_results: List of Mahalanobis errors for SSP.
#     - dataset_name: Name of the dataset (used for filenames).
#     """
#
#     # Ensure results directory exists
#     results_dir = "results"
#     os.makedirs(results_dir, exist_ok=True)
#
#     # === Save Error Vectors as JSON ===
#     output_filename = os.path.join(results_dir, f"{dataset_name}_mahalanobis_errors.json")
#     results_dict = {
#         "MaxEnt_Mahalanobis": np.array(MaxEnt_results).tolist(),
#         "SSP_Mahalanobis": np.array(SSP_results).tolist()
#     }
#     with open(output_filename, "w") as f:
#         json.dump(results_dict, f, indent=4)
#     print(f"Mahalanobis errors saved to {output_filename}")
#
#     # === Plot Mahalanobis Errors ===
#     mean_maxent = np.mean(MaxEnt_results)
#     mean_ssp = np.mean(SSP_results)
#     trials = range(1, len(MaxEnt_results) + 1)
#
#     plt.figure(figsize=(6, 5))
#     plt.scatter(trials, MaxEnt_results, marker='o', label='MaxEnt Mahalanobis Error', color='blue')
#     plt.scatter(trials, SSP_results, marker='o', label='SSP Mahalanobis Error', color='orange')
#
#     plt.axhline(mean_maxent, color='blue', linestyle='--', label=f'Mean MaxEnt: {mean_maxent:.2e}')
#     plt.axhline(mean_ssp, color='orange', linestyle='--', label=f'Mean SSP: {mean_ssp:.2e}')
#
#     plt.xlabel('Trial')
#     plt.ylabel('Mahalanobis Error')
#     plt.title(f'MaxEnt vs SSP Mahalanobis Error ({dataset_name})')
#     plt.yscale('log')
#     plt.legend()
#     plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#
#     # === Save Plot ===
#     plot_filename = os.path.join(results_dir, f"{dataset_name}_mahalanobis_plot.png")
#     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#     print(f"Plot saved to {plot_filename}")