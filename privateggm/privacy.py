import numpy as np


def cov_sensitivity_linf(n, d, B):
    return 6 * d * (n - 1) / n ** 2


def cov_sensitivity_l2(n, R):
    return np.sqrt(2) / n * R ** 2


def cov_jk_sensitivity_linf(n, B):
    Delta_cov_jk = 6 * (n - 1) * B ** 2 / n ** 2
    return Delta_cov_jk


def cov_jk_sensitivity_l2(n, R):
    return R ** 2 / n


def exponential_mechanism(off_diag_errors, rho, alpha, Delta_cov_jk, rounds):
    # compute the log of the unnormalized probabilities
    log_unnorm_probabilities = ((2 * np.sqrt(rho * (1 - alpha) / rounds)) / (2 * Delta_cov_jk)) * np.array(
        off_diag_errors)

    # compute log-sum-exp to normalize probabilities
    log_prob_sum = logsumexp(log_unnorm_probabilities)

    # compute normalizd probabilities using log-probabilities
    off_diag_probabilities = np.exp(log_unnorm_probabilities - log_prob_sum)
    print("off_diag_probabilities", off_diag_probabilities)

    # select
    worst_off_diag_index = np.random.choice(len(off_diag_errors), p=off_diag_probabilities)

    return worst_off_diag_index


def SSP_covariance(cov_matrix, cov_sensitivity, rho):
    """
    See Overleaf for derivation
    """

    d, _ = cov_matrix.shape

    # tri = np.triu(np.random.normal(0, 1, (d, d)))
    # Zsym = tri + tri.T - np.diag(np.diag(tri))

    # #noise_scale = (2 * np.log(1.25/delta) * cov_sensitivity**2) / epsilon**2
    noise_scale = cov_sensitivity / np.sqrt(2 * rho)
    print(f"SSP noise scale: {noise_scale}")

    # # leveraging scaling property of normal distributions
    # cov_dp = cov_matrix + noise_scale * Zsym
    tri = np.triu(np.random.normal(0, noise_scale, (d, d)))
    Zsym = tri + tri.T - np.diag(np.diag(tri))
    cov_dp = cov_matrix + Zsym

    return cov_dp


def gaussian_mechanism(obj, sensitivity, rho):
    """
    Apply the Gaussian mechanism to a function output under zCDP.

    Parameters:
    - obj: np.array, the output of the function f.
    - sensitivity: float, the L2 sensitivity of the function f.
    - rho: float, the zCDP privacy budget.

    Returns:
    - noisy_obj: np.array, the output of the function f with Gaussian noise added.
    """
    noise_scale = sensitivity / np.sqrt(2 * rho)
    print("noise scale MaxEnt: ", noise_scale)
    noise = np.random.normal(0, noise_scale, size=obj.shape)

    noisy_obj = obj + noise

    return noisy_obj
