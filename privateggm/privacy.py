import numpy as np
from scipy.special import logsumexp

from privateggm.utils import *


def cov_sensitivity_linf(n, d, B):
    return np.sqrt(2) * d * B**2 / n


def cov_sensitivity_l2(n, R):
    return np.sqrt(2) / n * R ** 2


def cov_jk_sensitivity_linf(n, B):
    Delta_cov_jk = 2 * B**2 / n
    return Delta_cov_jk


def cov_jk_sensitivity_l2(n, R):
    return R ** 2 / n


def exponential_mechanism(rng, flat_scores, Delta_cov_jk, rho):
    # note about privacy conversion:
    # ρ_cost = ε² / 8   --->   ε = √(8ρ) = 2√(2ρ)
    # ρ_cost = (2√ρ)² / 8 = 4ρ / 8 = ρ/2

    # compute the log of the unnormalized probabilities
    # log_unnorm_probabilities = ((2 * np.sqrt(rho)) / (2 * Delta_cov_jk)) * np.array(
    #     flat_scores)
    log_unnorm_probabilities = ((2 * np.sqrt(2 * rho)) / (2 * Delta_cov_jk)) * np.array(
          flat_scores)

    # compute log-sum-exp to normalize probabilities
    log_prob_sum = logsumexp(log_unnorm_probabilities)

    # compute normalizd probabilities using log-probabilities
    off_diag_probabilities = np.exp(log_unnorm_probabilities - log_prob_sum)
    #print("off_diag_probabilities", off_diag_probabilities)

    # select
    worst_off_diag_index = rng.choice(len(flat_scores), p=off_diag_probabilities)

    return worst_off_diag_index


def SSP_covariance(cov_matrix, cov_sensitivity, rho, rng=None, ssp_seed=1):
    """
    Simultaneous Structured Perturbation for covariance matrices.
    
    See Overleaf for derivation
    """

    d, _ = cov_matrix.shape

    # Use the provided RNG for proper randomness across trials
    if rng is None:
        ssp_rng = np.random.default_rng(ssp_seed)
    else:
        # Use the provided RNG directly to ensure different noise per trial
        ssp_rng = rng

    # #noise_scale = (2 * np.log(1.25/delta) * cov_sensitivity**2) / epsilon**2
    noise_scale = cov_sensitivity / np.sqrt(2 * rho) # add/remove one
    print(f" %%%%% noise scale SSP: {noise_scale} %%%%%")

    # Generate symmetric noise matrix using the dedicated SSP RNG
    tri = np.triu(ssp_rng.normal(0, noise_scale, (d, d)))
    Zsym = tri + tri.T - np.diag(np.diag(tri))
    cov_dp = cov_matrix + Zsym
    
    # project to PSD, which is also done for PrivateGGM. Use the same threshold
    cov_dp = project_to_PSD(cov_dp, thresh=1e-3)

    return cov_dp, noise_scale


def gaussian_mechanism(rng, obj, sensitivity, rho):
    """
    Apply the Gaussian mechanism to a function output under zCDP.

    Parameters:
    - obj: np.array, the output of the function f.
    - sensitivity: float, the L2 sensitivity of the function f.
    - rho: float, the zCDP privacy budget.

    Returns:
    - noisy_obj: np.array, the output of the function f with Gaussian noise added.
    - noise: np.array, the actual noise sample that was added.
    """
    noise_scale = sensitivity / np.sqrt(2 * rho) # add/remove one
    noise = rng.normal(0, noise_scale, size=obj.shape)
    
    noisy_obj = obj + noise
    
    return noisy_obj, noise, noise_scale


def privatize_Sigma_tilde(rng, Sigma_tilde_vec, inds, d, Delta_cov_jk, rho):
    """
    Privatize the entries of a covariance matrix using the Gaussian mechanism.

    Parameters:
    - Sigma_tilde_vec: np.array, the vectorized version of the covariance matrix.
    - inds: np.array, the indices of the lower triangle elements.
    - d: int, the dimension of the covariance matrix.
    - Delta_cov_jk: float, the sensitivity of the covariance matrix entries.
    - rho: float, the total privacy budget (zCDP).

    Returns:
    - Sigma_tilde: np.array, the privatized covariance matrix.
    """
    # Distribute budget across all observed Sigma_tilde entries
    # inds includes diagonal + lower triangle
    diag_and_lowertri_inds_num = len(inds)  # Only need to privatize lower triangle elements
    rho_per_entry = rho / diag_and_lowertri_inds_num

    # Apply the Gaussian mechanism to privatize the entries
    #print("MAXENT SENSITIVITY", Delta_cov_jk)
    Sigma_tilde_vec_dp, noise_components, noise_scale = gaussian_mechanism(rng, Sigma_tilde_vec, Delta_cov_jk, rho_per_entry)
    print(f"noise scale MaxEnt (privatize_Sigma_tilde): {noise_scale}")

    # Reconstruct the full covariance matrix from the vectorized version
    Sigma_tilde_dp = reconstruct_sigma_tilde(inds, Sigma_tilde_vec_dp, d)

    eigvals = np.linalg.eigvalsh(Sigma_tilde_dp)
    min_eigval = np.min(eigvals)
    max_eigval = np.max(eigvals)
    cond_number = max_eigval / min_eigval if min_eigval > 1e-10 else np.inf

    is_non_pd = not jnp.all(eigvals > 0)
    is_ill_conditioned = cond_number > 1e10
    is_nearly_singular = min_eigval < 1e-8

    if is_non_pd or is_ill_conditioned or is_nearly_singular:
        print("0. PROBLEM DETECTED WITH Sigma:")
        print(f"0. is_non_pd? {is_non_pd}")
        print(f"0. is_ill_conditioned? {is_ill_conditioned}")
        print(f"0. is_nearly_singular? {is_nearly_singular}")
        #print(Sigma_tilde_dp)
        np.save("pathologic_sigma_tilde_dp.npy", Sigma_tilde_dp)


    # Ensure the matrix is symmetric
    assert np.allclose(Sigma_tilde_dp, Sigma_tilde_dp.T), "Sigma_tilde is not symmetric"

    return Sigma_tilde_dp, Sigma_tilde_vec_dp, noise_scale