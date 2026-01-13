import os

from privateggm.data import *
from privateggm.utils import *
from privateggm.model import *
from privateggm.privacy import *


def run_driver(
    rng,
    X,
    Sigma_true,
    T,
    structure='custom',
    structure_dict=None,
    max_iters=50,
    iterative=False,
    max_off_diag_percentage=0.5,
    apply_dp=False,
    rho=None,
    clip_type="linf",
    rescale=True,
    alpha=None,
    optimizer="pgd-maxent"
):
    """
    General driver function for processing different datasets with flexible off-diagonal configurations.

    Parameters:
    -----------
    rng : RandomState
        Random number generator
    X : array-like
        Input dataset of shape (n, d)
    Sigma_true : array-like
        True covariance matrix
    T : float
        Sensitivity parameter
    structure : str, default='custom'
        Structure type ('chain', 'custom', 'rndpercent')
    structure_dict : dict, optional
        Custom structure dictionary
    max_iters : int, default=50
        Maximum optimization iterations
    iterative : bool, default=False
        Whether to use iterative mode
    max_off_diag_percentage : float, default=0.5
        Percentage of off-diagonal elements to keep
    apply_dp : bool, default=False
        Whether to apply differential privacy
    rho : float, optional
        Differential privacy budget
    clip_type : str, default='l2'
        Type of clipping for sensitivity ('l2' or 'linf')
    rescale : bool, default=True
        Whether to rescale the data
    alpha : float, optional
        Hyperparameter for privacy
    optimizer : str, default='lbfgs'
        Optimization method ('lbfgs', 'mirror_descent', or 'pgd-maxent')

    Returns:
    --------
    results_dict : dict
        Contains Sigma_true, Sigma_tilde, Sigma_opt, and Sigma_SSP (if DP applied)
    """
    n, d = X.shape

    # Generate structure based on specified type
    if structure == 'chain':
        Sigma_tilde = generate_chain_structure(Sigma_true)
    elif structure == 'custom':
        Sigma_tilde = generate_custom_structure(Sigma_true, structure_dict)
    elif structure == 'rndpercent':
        Sigma_tilde = generate_random_offdiag_structure(
            Sigma_true, 
            percentage=max_off_diag_percentage,
            rng=rng
        )
    else:
        raise ValueError(
            "Invalid structure. Choose 'chain', 'custom', 'rndpercent', "
            "or provide a valid structure dictionary."
        )

    # Run optimization based on mode
    if iterative:
        iters = int(((d * d) - d) * max_off_diag_percentage)
        print(f"Iterations: {iters}")
        
        Sigma_tilde, Sigma_opt = run_iterative_mode(
            rng, d, X, iters, 
            rho, alpha, clip_type, T
        )

    else:
        Sigma_opt, observed_inds, pgd_noise_scale = run_non_iterative_mode(
            rng, n, T, Sigma_true, Sigma_tilde, 
            max_iters, apply_dp, rho, clip_type, optimizer
        )

    # Construct results dictionary
    results_dict = {
        'Sigma_true': Sigma_true,
        'Sigma_tilde': Sigma_tilde,
        'Sigma_opt': Sigma_opt
    }
    
    # Add observed indices if available (from non-iterative mode)
    if 'observed_inds' in locals():
        results_dict['observed_inds'] = observed_inds
    
    # Add PGD noise scale if available
    if 'pgd_noise_scale' in locals() and pgd_noise_scale is not None:
        results_dict['pgd_noise_scale'] = pgd_noise_scale

    # Add SSP baseline if differential privacy is applied
    if apply_dp and rho is not None:
        if clip_type == 'l2':
            cov_sensitivity = cov_sensitivity_l2(n, T)
        elif clip_type == 'linf':
            cov_sensitivity = cov_sensitivity_linf(n, d, T)  # Unit ball
            print("DRIVER: cov_sensitivity linf: ", cov_sensitivity)
        else:
            raise ValueError(f"Unknown clip_type: {clip_type}")
            
        Sigma_SSP, ssp_noise_scale = SSP_covariance(Sigma_true, cov_sensitivity, rho, rng=rng)
        results_dict['Sigma_SSP'] = Sigma_SSP
        results_dict['ssp_noise_scale'] = ssp_noise_scale

    return results_dict


def run_non_iterative_mode(
    rng,
    n,
    bound,
    Sigma_true,
    Sigma_tilde,
    max_iters,
    apply_dp,
    rho,
    clip_type,
    optimizer
):
    """
    Run the driver in non-iterative mode.

    Parameters:
    -----------
    rng : RandomState
        Random number generator
    n : int
        Number of samples
    bound : float
        Sensitivity bound
    Sigma_true : array-like
        True covariance matrix
    Sigma_tilde : array-like
        Observed covariance structure
    max_iters : int
        Maximum optimization iterations
    apply_dp : bool
        Whether to apply differential privacy
    rho : float
        Privacy budget
    clip_type : str
        Type of clipping ('l2' or 'linf')
    optimizer : str
        Optimization method ('lbfgs', 'mirror_descent', or 'pgd-maxent')

    Returns:
    --------
    Sigma_opt : array-like
        Optimized covariance matrix
    """
    # Compute precision matrix and observed entries
    Prec_true = np.linalg.inv(Sigma_true)
    inds = get_observed_Sigma_entries(Sigma_tilde)
    d = compute_d(inds)
    rows, cols = inds[:, 0], inds[:, 1]
    Sigma_tilde_vec = Sigma_tilde[rows, cols]

    # Apply differential privacy if requested
    pgd_noise_scale = None  # Initialize
    if apply_dp:
        if clip_type == 'linf':
            Delta_cov_jk = cov_jk_sensitivity_linf(n, bound)
        elif clip_type == 'l2':
            Delta_cov_jk = cov_jk_sensitivity_l2(n, bound)
        else:
            raise ValueError(f"Unknown clip_type: {clip_type}")
            
        Sigma_tilde, Sigma_tilde_vec, pgd_noise_scale = privatize_Sigma_tilde(
            rng, Sigma_tilde_vec, inds, d, Delta_cov_jk, rho
        )

    # Select and run optimization method
    if optimizer == "maxent":
        # L-BFGS Optimization
        z_init = initialize_z(Sigma_tilde, inds)
        z_opt, g_vals, obj_vals = optimize_lbfgs(
            Sigma_tilde_vec, z_init, inds, d, max_iters=max_iters
        )
        K_z_opt = get_K(z_opt, inds, d)
        Sigma_opt = np.linalg.inv(K_z_opt)

    elif optimizer == "mirror-descent":
        # Mirror Descent Optimization
        Sigma_init = np.eye(d)
        tau = np.ones_like(Sigma_true)  # Use noise matrix if DP is applied
        Sigma_opt, K_z_opt = mirror_descent(
            Sigma_init, Sigma_tilde, inds, tau, max_iters=max_iters
        )

    elif optimizer == "pgd-maxent":
        # Projected Gradient Descent - MaxEnt
        # Note: Requires privatization of the entire matrix
        Sigma_opt = pgd_maxent(
            Sigma_tilde, inds, 
            max_iters=1000, eta=0.1, tol=1e-6, verbose=False
        )
        K_z_opt = np.linalg.inv(Sigma_opt)

    else:
        raise ValueError(
            f"Invalid optimizer '{optimizer}'. "
            "Use 'maxent', 'mirror-descent', or 'pgd-maxent'."
        )

    # Print error metrics
    print(f"Mahalanobis Error (Precision): {mahalanobis_cov_error(Prec_true, K_z_opt)}")
    print(f"Mahalanobis Error (Covariance): {mahalanobis_cov_error(Sigma_true, Sigma_opt)}")

    return Sigma_opt, inds, pgd_noise_scale


def run_iterative_mode(
    rng,
    d,
    X,
    n_measurements,
    rho,
    alpha,
    sensitivity_type,
    bound
):
    """
    Run the driver in iterative mode.

    Parameters:
    -----------
    rng : RandomState
        Random number generator
    d : int
        Dimension of the covariance matrix
    X : array-like
        Input dataset
    n_measurements : int
        Number of off-diagonal covariance entries to measure adaptively
    rho : float
        Privacy budget
    alpha : float
        Privacy hyperparameter (fraction of budget for diagonal measurements)
    sensitivity_type : str
        Type of sensitivity ('l2' or 'linf')
    bound : float
        Sensitivity bound

    Returns:
    --------
    Sigma_opt : array-like
        Optimized covariance matrix
    model : IterativePrivateGGM
        The fitted model (contains noise scale info)
    """
    # Initialize and fit the iterative private GGM model
    iterative_model = IterativePrivateGGM(
        rng, d, rho, alpha, n_measurements, sensitivity_type, bound
    )
    
    Sigma_opt = iterative_model.fit(X)
    
    return Sigma_opt, iterative_model