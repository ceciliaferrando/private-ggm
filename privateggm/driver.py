import os

from privateggm.data import *
from privateggm.utils import *
from privateggm.model import *
from privateggm.privacy import *

def run_driver(data, Sigma_true, T, structure='custom', structure_dict=None, max_iters=50,
               iterative=False, max_off_diag_percentage=0.5, apply_dp=False, rho=None,
               clip_type="l2", rescale=True, alpha=None):
    """
    General driver function for processing different datasets with flexible off-diagonal configurations.

    Parameters:
    - data_path: str, path to the dataset file.
    - structure: str, predefined structure ('chain', 'custom', 'rndpercent') or 'custom' for user-defined.
    - structure_dict: dict, specifies the non-zero elements for custom structures.
    - max_iters: int, maximum number of iterations for optimization.
    - iterative: bool, whether to run in iterative mode.
    - max_off_diag_percentage: float, maximum percentage of off-diagonal elements to observe in iterative mode.

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
        results_dict = run_iterative_mode(n, Sigma_true, Sigma_tilde, max_iters, max_off_diag_percentage, apply_dp, rho,
                                          alpha)
    else:
        results_dict = run_non_iterative_mode(n, T, Sigma_true, Sigma_tilde, max_iters, apply_dp, rho, clip_type)

    return results_dict


def run_non_iterative_mode(n, T, Sigma_true, Sigma_tilde, max_iters, apply_dp, rho, clip_type):
    """
    Run the driver in non-iterative mode.
    """
    Prec_tilde = np.linalg.inv(Sigma_tilde)
    inds = get_observed_Sigma_entries(Sigma_tilde)
    d = compute_d(inds)
    rows, cols = inds[:, 0], inds[:, 1]
    Sigma_tilde_vec = Sigma_tilde[rows, cols]

    if apply_dp:
        if clip_type == 'linf':
            Delta_cov_jk = cov_jk_sensitivity_linf(n, T)
        elif clip_type == 'l2':
            Delta_cov_jk = cov_jk_sensitivity_l2(n, T)
        Sigma_tilde, Sigma_tilde_vec = privatize_Sigma_tilde(Sigma_tilde_vec, inds, d, Delta_cov_jk, rho)

    # initialize z
    z_init = initialize_z(Sigma_tilde, inds)

    # optimize using LBFGS
    z_opt, g_vals, obj_vals = optimize_lbfgs(Sigma_tilde_vec, z_init, inds, d, max_iters=max_iters)
    K_z_opt = get_K(z_opt, inds, d)
    Sigma_opt = np.linalg.inv(K_z_opt)

    # compute mses
    mses = compute_mses(inds, Sigma_true, Sigma_opt)

    results_dict = {'mses': mses, 'g_vals': g_vals, 'obj_vals': obj_vals,
                    'Sigma_true': Sigma_true, 'Sigma_tilde': Sigma_tilde, 'Sigma_opt': Sigma_opt}

    # add SSP baseline
    if clip_type == 'l2':
        cov_sensitivity = cov_sensitivity_l2(n, T)
    elif clip_type == 'linf':
        cov_sensitivity = cov_sensitivity_linf(n, d, 1) # unit ball
    print("SSP sensitivity", cov_sensitivity)
    Sigma_SSP = SSP_covariance(Sigma_true, cov_sensitivity, rho)

    results_dict['Sigma_SSP'] = Sigma_SSP

    return results_dict