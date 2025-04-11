import os

from privateggm.data import *
from privateggm.utils import *
from privateggm.model import *
from privateggm.privacy import *

def run_driver(rng, data, Sigma_true, T, structure='custom', structure_dict=None, max_iters=50,
               iterative=False, max_off_diag_percentage=0.5, apply_dp=False, rho=None,
               clip_type="l2", rescale=True, alpha=None, optimizer="lbfgs"):
    """
    General driver function for processing different datasets with flexible off-diagonal configurations.

    Parameters:
    - rng: Random number generator
    - data: Input dataset
    - Sigma_true: True covariance matrix
    - T: Sensitivity parameter
    - structure: Structure type ('chain', 'custom', 'rndpercent')
    - structure_dict: Custom structure dictionary
    - max_iters: Maximum optimization iterations
    - iterative: Whether to use iterative mode
    - max_off_diag_percentage: Percentage of off-diagonal elements to keep
    - apply_dp: Whether to apply differential privacy
    - rho: Differential privacy budget
    - clip_type: Type of clipping for sensitivity ('l2' or 'linf')
    - rescale: Whether to rescale the data
    - alpha: Hyperparameter for privacy
    - optimizer: Optimization method ('lbfgs' or 'mirror_descent')

    Returns:
    - results_dict: dict, contains mse, gradient values, objective values, and other metadata.
    """

    n, d = data.shape

    # Define different structures
    if structure == 'chain':
        Sigma_tilde = generate_chain_structure(Sigma_true)
    elif structure == 'custom':
        Sigma_tilde = generate_custom_structure(Sigma_true, structure_dict)
    elif structure == 'rndpercent':
        Sigma_tilde = generate_random_offdiag_structure(Sigma_true, percentage=max_off_diag_percentage)
    else:
        raise ValueError("Invalid structure. Choose 'chain', 'custom', 'rndpercent', or provide a valid structure dictionary.")

    # Non-iterative and iterative modes
    if iterative:
        results_dict = run_iterative_mode(rng, n, Sigma_true, Sigma_tilde, max_iters, max_off_diag_percentage,
                                          apply_dp, rho, alpha)
    else:
        results_dict = run_non_iterative_mode(rng, n, T, Sigma_true, Sigma_tilde, max_iters, apply_dp,
                                              rho, clip_type, optimizer)

    return results_dict


def run_non_iterative_mode(rng, n, T, Sigma_true, Sigma_tilde, max_iters, apply_dp, rho, clip_type, optimizer):
    """
    Run the driver in non-iterative mode.

    Parameters:
    - optimizer: Which optimization method to use ('lbfgs' or 'mirror_descent')
    """

    Prec_true = np.linalg.inv(Sigma_true)
    inds = get_observed_Sigma_entries(Sigma_tilde)
    d = compute_d(inds)
    rows, cols = inds[:, 0], inds[:, 1]
    Sigma_tilde_vec = Sigma_tilde[rows, cols]

    if apply_dp:
        if clip_type == 'linf':
            Delta_cov_jk = cov_jk_sensitivity_linf(n, T)
        elif clip_type == 'l2':
            Delta_cov_jk = cov_jk_sensitivity_l2(n, T)
        Sigma_tilde, \
        Sigma_tilde_vec


    # **Optimization Method Selection**
    if optimizer == "maxent":
        # L-BFGS Optimization
        z_init = initialize_z(Sigma_tilde, inds)
        z_opt, g_vals, obj_vals = optimize_lbfgs(Sigma_tilde_vec, z_init, inds, d, max_iters=max_iters)
        K_z_opt = get_K(z_opt, inds, d)
        Sigma_opt = np.linalg.inv(K_z_opt)

    elif optimizer == "mirror-descent":
        # Mirror Descent Optimization
        Sigma_init = np.eye(d)
        tau = np.ones_like(Sigma_true)  # Use noise matrix if DP is applied
        Sigma_opt, K_z_opt = mirror_descent(Sigma_init, Sigma_tilde, inds, tau, max_iters=max_iters)

    elif optimizer == "pgd-maxent":
        # need to first privatize the whole matrix

        Sigma_opt = pgd_maxent(Sigma_tilde, inds, max_iters=1000, eta=0.1, tol=1e-6, verbose=True)
        K_z_opt = np.linalg.inv(Sigma_opt)

    else:
        raise ValueError("Invalid optimizer choice. Use 'lbfgs' or 'mirror_descent'.")

    print("M ERROR PREC:", mahalanobis_cov_error(Prec_true, K_z_opt))
    print("M ERROR COV:", mahalanobis_cov_error(Sigma_true, Sigma_opt))

    results_dict = {'Sigma_true': Sigma_true, 'Sigma_tilde': Sigma_tilde, 'Sigma_opt': Sigma_opt}

    # Add SSP baseline
    if clip_type == 'l2':
        cov_sensitivity = cov_sensitivity_l2(n, T)
    elif clip_type == 'linf':
        cov_sensitivity = cov_sensitivity_linf(n, d, 1)  # Unit ball
    Sigma_SSP = SSP_covariance(Sigma_true, cov_sensitivity, rho)

    results_dict['Sigma_SSP'] = Sigma_SSP

    return results_dict