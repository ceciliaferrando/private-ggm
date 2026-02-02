import numpy as np
from scipy.linalg import solve, cholesky, eigh
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
import optax
from functools import partial

from privateggm.utils import *
from privateggm.privacy import gaussian_mechanism, exponential_mechanism, cov_jk_sensitivity_l2, cov_jk_sensitivity_linf

# DEBUG_MODE = True

# DIRECT LBFGS OPTIMIZATION METHOD

def loss_fn(z, inds, Sigma_tilde_vec, d, weights=None):
    """
    z [jnp.array]: vector supporting precision matrix estimate K(z)
    inds [np.array]: list of the lower triangular nonzero element indices in Sigma_tilde
    Sigma_tilde_vec [np.array]: vector of Sigma_tilde indexed at inds
    d [int]: static argument, number of attributes
    weights [np.array]: optional weights for the loss terms
    """

    # Reconstruct K_z
    K_z = get_K(z, inds, d)

    # Get loss terms
    log_det_term = logdet_psd(K_z)
    trace_term = 2 * jnp.dot(z, Sigma_tilde_vec)

    # Total loss
    total_loss = -log_det_term + trace_term

    return total_loss

def optimize_lbfgs(Sigma_tilde_vec, z, inds, d, max_iters=100, verbose=False):
    """
    Sigma_tilde_vec [np.array]: matrix of observed (privatized) covariance entries
    z [jnp.array]: vector supporting K(z)
    inds [np.array]: vector of indices of nonzero covariance entries
    d [int]: number of attributes
    """

    rng_key = jrandom.PRNGKey(42)

    # Linesearch
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)

    # Set up optimizer
    opt = optax.chain(
        optax.lbfgs(),
        linesearch,
    )

    # Init state
    state = opt.init(z)

    # Loss function wrapper
    loss_fn_wrapper = partial(loss_fn, inds=inds, Sigma_tilde_vec=Sigma_tilde_vec, d=d)

    # Value and gradient function
    v_and_g = jax.value_and_grad(loss_fn_wrapper)

    @jax.jit
    def optimization_step(z, state, step_size):
        v, g = v_and_g(z)
        updates, state = opt.update(g, state, z, value=v, grad=g, value_fn=loss_fn_wrapper)
        z_new = z + step_size * updates
        grad_sum = jnp.sum(jnp.abs(g))
        return z_new, state, v, grad_sum

    # Optimization loop
    g_vals = []
    obj_vals = []
    prev_v = None
    tol = 1e-8
    eta = 1.0

    for i in range(max_iters):
        z_new, state, v, grad_sum = optimization_step(z, state, eta)

        # Backtracking if loss increases
        if prev_v is not None and v > prev_v:
            eta *= 0.5
            if verbose:
                print(f"Iter {i:5d} | Loss increased → reducing step size to {eta:.3e}")
            continue

        z = z_new
        eta = min(eta * 1.1, 1.0)

        obj_vals.append(v.item())
        g_vals.append(grad_sum.item())

        if verbose and i % 100 == 0:
            print(f"Iter {i:5d} | Obj: {v:.3e} | Grad Sum: {grad_sum:.3e} | η: {eta:.3e}")

        if grad_sum < tol:
            if verbose:
                print(f"Converged at iter {i:5d} | Grad Sum: {grad_sum:.3e}")
            break

        if prev_v is not None and abs(v - prev_v) < tol:
            if verbose:
                print(f"Converged at iter {i:5d} | ΔLoss: {abs(v - prev_v):.3e}")
            break

        prev_v = v

    # Final K(z) construction
    K_z = get_K(z, inds, d)

    eigvals = np.linalg.eigvalsh(K_z)
    min_eigval = np.min(eigvals)
    max_eigval = np.max(eigvals)
    cond_number = max_eigval / min_eigval if min_eigval > 1e-10 else np.inf

    is_non_pd = not jnp.all(eigvals > 0)
    is_ill_conditioned = cond_number > 1e10
    is_nearly_singular = min_eigval < 1e-8

    if is_non_pd or is_ill_conditioned or is_nearly_singular:
        print("-" * 60)
        print("jIssue detected with optimized K(z):")
        print(f"  • Non-PD:            {is_non_pd}")
        print(f"  • Ill-conditioned:   {is_ill_conditioned} (cond #: {cond_number:.2e})")
        print(f"  • Nearly singular:   {is_nearly_singular} (min eigval: {min_eigval:.2e})")
        print("-" * 60)

    return z, g_vals, obj_vals


# MIRROR DESCENT METHOD

def mirror_descent(Sigma_init, Sigma_tilde, inds, tau, max_iters=20000, eta_init=1e-5, tol=1e-8, verbose=False):
    """
    Args:
        Sigma_init: Initial covariance matrix (positive definite)
        Sigma_tilde: Empirical covariance matrix with noisy observations
        inds: List of observed indices as tuples (i,j) where i <= j
        tau: Matrix of noise scales (same shape as Sigma_tilde)
        max_iters: Maximum number of iterations
        eta_init: Initial learning rate
        tol: Convergence tolerance (Frobenius norm change in K)
        verbose: Print progress

    Returns:
        Sigma: Optimized covariance matrix
        K: Inverse of Sigma
    """
    # Initialize variables
    Sigma = Sigma_init.copy()
    K = np.linalg.inv(Sigma)
    eta = eta_init

    def objective(Sigma):
        loss = 0.0
        for (i, j) in inds:
            loss += (1 / tau[i, j] ** 2) * (Sigma_tilde[i, j] - Sigma[i, j]) ** 2
        return loss

    for iter in range(max_iters):

        # compute gradient (symmetric)
        grad = np.zeros_like(Sigma)
        for (i, j) in inds:
            grad_ij = (2 / tau[i, j] ** 2) * (Sigma[i, j] - Sigma_tilde[i, j])
            grad[i, j] = grad_ij
            if i != j:
                grad[j, i] = grad_ij

        # backtracking line search
        success = False
        current_eta = eta  # Start with previous iteration's eta
        K_candidate = None

        while not success and current_eta > 1e-12:
            K_candidate = K + current_eta * grad
            try:
                # Check positive definiteness
                cholesky(K_candidate, lower=True)
                success = True
            except np.linalg.LinAlgError:
                current_eta *= 0.9  # Reduce step size

        if not success:
            raise RuntimeError(f"Failed to find PD K matrix after {iter} iterations")

        # update
        K_prev = K.copy()
        K = K_candidate
        Sigma = np.linalg.inv(K)
        eta = current_eta

        # step size decay regardless of the conditions above
        # eta *= 0.995

        # convergence check
        delta = np.linalg.norm(K - K_prev, 'fro')
        if delta < tol:
            if verbose:
                print(f"Converged at iteration {iter}")
            break

        # progress prints
        if verbose and iter % 100 == 0:
            obj = objective(Sigma)
            grad_norm = np.linalg.norm(grad)
            print(f"Iter {iter:5d} | Obj: {obj:.3e} | ΔK: {delta:.3e} | η: {eta:.3e} | Grad: {grad_norm:.3e}")

    return Sigma, K


def projected_gradient_descent(S_observed, mask, weights=None, eta=0.1, max_iters=1000, tol=1e-8):
    d = S_observed.shape[0]

    # Ensure mask is 2D boolean array w/ matching shape
    assert mask.shape == S_observed.shape, "Mask must match matrix dimensions"

    # 0. Set up weights (default to uniform if not provided)
    if weights is None:
        weight_matrix = np.ones_like(S_observed)
    else:
        weight_matrix = weights.copy()
        
        # Cap the weight ratio to prevent extreme imbalance
        # This trades off statistical optimality for numerical stability
        if np.any(mask):
            min_weight = np.min(weight_matrix[mask])
            max_weight = np.max(weight_matrix[mask])
            weight_ratio = max_weight / min_weight if min_weight > 0 else 1.0
            
            # Cap ratio at 100 (conservative)
            max_allowed_ratio = 100.0
            if weight_ratio > max_allowed_ratio:
                # Clip weights to reduce ratio
                # Strategy: raise the floor (increase min weights) rather than lower ceiling
                target_min_weight = max_weight / max_allowed_ratio
                weight_matrix[mask] = np.maximum(weight_matrix[mask], target_min_weight)
                new_ratio = np.max(weight_matrix[mask]) / np.min(weight_matrix[mask])
                print(f"  [PGD] Capping weight ratio: {weight_ratio:.1f} -> {new_ratio:.1f}")
        
        # Scale weights to prevent overflow in gradient computation
        max_weight = np.max(weight_matrix[mask]) if np.any(mask) else 1.0
        weight_scale = max(1.0, max_weight / 1000.0)  # Normalize to ~1000 max weight
        weight_matrix = weight_matrix / weight_scale

    # Use fixed learning rate (no adaptive scaling needed with capped weights)
    eta_adjusted = eta 

    # 1. set unobserved entries to 0
    S = np.zeros_like(S_observed)
    S[mask] = S_observed[mask]  # Correct 2D indexing

    # 2. PSD projection
    Sigma = project_to_PSD(S, thresh=1e-3)

    prev_Sigma = Sigma.copy()
    for t in range(max_iters):
        # 3. compute gradient ONLY on observed entries
        gradient = np.zeros_like(Sigma)
        # OLD unweighted version gradient[mask] = Sigma[mask] - S[mask]
        gradient[mask] = weight_matrix[mask] * (Sigma[mask] - S[mask])
        # double the diagonal elements of the gradient
        gradient_diag = np.diag(np.diag(gradient))
        gradient = gradient + gradient_diag

        # DEBUG: Check for numerical issues
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            print(f"  [PGD DEBUG] Iteration {t}: gradient has NaN or Inf!")
            print(f"    weight_matrix[mask] range: [{np.min(weight_matrix[mask]):.2e}, {np.max(weight_matrix[mask]):.2e}]")
            print(f"    Sigma[mask] range: [{np.min(Sigma[mask]):.2e}, {np.max(Sigma[mask]):.2e}]")
            print(f"    S[mask] range: [{np.min(S[mask]):.2e}, {np.max(S[mask]):.2e}]")
            print(f"    gradient[mask] range: [{np.min(gradient[mask]):.2e}, {np.max(gradient[mask]):.2e}]")
            print(f"    gradient_diag range: [{np.min(np.diag(gradient_diag)):.2e}, {np.max(np.diag(gradient_diag)):.2e}]")
            return prev_Sigma  # Return PREVIOUS valid state, not current corrupted Sigma

        # 4. update and project to psd
        Sigma_temp = Sigma - eta_adjusted * gradient
        
        if np.any(np.isnan(Sigma_temp)) or np.any(np.isinf(Sigma_temp)):
            print(f"  [PGD DEBUG] Iteration {t}: Sigma_temp has NaN or Inf after gradient update!")
            print(f"    eta_adjusted: {eta_adjusted:.2e}")
            print(f"    gradient range: [{np.min(gradient):.2e}, {np.max(gradient):.2e}]")
            return Sigma  # Return previous Sigma
        
        Sigma = project_to_PSD(Sigma_temp, thresh=1e-3)
        
        if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
            print(f"  [PGD DEBUG] Iteration {t}: Sigma has NaN or Inf after PSD projection!")
            print(f"    Sigma_temp range: [{np.min(Sigma_temp):.2e}, {np.max(Sigma_temp):.2e}]")
            eigvals_temp = np.linalg.eigvalsh(Sigma_temp)
            print(f"    Sigma_temp eigenvalues: [{np.min(eigvals_temp):.2e}, {np.max(eigvals_temp):.2e}]")
            return prev_Sigma  # Return previous valid Sigma

        if np.linalg.norm(Sigma - prev_Sigma, 'fro') < tol:
            break
        prev_Sigma = Sigma.copy()

    return Sigma

def pgd_maxent(Sigma_tilde, inds, noise_scales=None,max_iters=1000, eta=0.01, tol=1e-8, verbose=False):
    """
    Two-step PGD + Max Entropy pipeline.

    Args:
        Sigma_true: Ground truth covariance matrix
        Sigma_tilde: Observed noisy covariance matrix
        inds: Array of observed indices [[i,j], ...]
        max_iters: Max iterations for PGD
        eta: Step size for PGD
        tol: Convergence tolerance for PGD
        verbose: Print progress
    """
    d = Sigma_tilde.shape[0]

    # After (correct 2D mask)
    mask = np.zeros(Sigma_tilde.shape, dtype=bool)  
    for (i, j) in inds:
        mask[i, j] = True
        mask[j, i] = True  # Ensure symmetry

    Sigma_obs = np.zeros_like(Sigma_tilde)
    Sigma_obs[mask] = Sigma_tilde[mask]  # Fill observed entries, others remain 0

    # Compute weight matrix (1/σ²) if noise scales provided
    if noise_scales is not None:
        weight_matrix = np.zeros_like(noise_scales)
        # Only compute weights for observed entries (avoid division by zero)
        raw_weights = 1.0 / (noise_scales[mask]**2)
        
        # Apply weight exponent to reduce weight variation
        # weight_exponent=1.0 → standard inverse-variance weighting
        # weight_exponent=0.5 → sqrt of inverse-variance (less aggressive)
        # weight_exponent=0.0 → uniform weights
        weight_exponent = 0.3  # Compress weight variation (0.3 reduces 150x to ~4x)
        weight_matrix[mask] = raw_weights ** weight_exponent
    else:
        weight_matrix = None

    # Stage A: Projected Gradient Descent
    if verbose:
        print("\nRunning Projected Gradient Descent...")
    Sigma_psd = projected_gradient_descent(Sigma_obs, mask, weights=weight_matrix, eta=eta, max_iters=max_iters, tol=tol)
    
    # DEBUG: Check PGD result
    if verbose:
        print(f"PGD result - min: {np.min(Sigma_psd):.6f}, max: {np.max(Sigma_psd):.6f}")
        eigvals_pgd = np.linalg.eigvalsh(Sigma_psd)
        print(f"PGD result - min eigval: {np.min(eigvals_pgd):.2e}, max eigval: {np.max(eigvals_pgd):.2e}")
        print(f"PGD result - condition number: {np.max(eigvals_pgd)/np.min(eigvals_pgd) if np.min(eigvals_pgd) > 1e-10 else np.inf:.2e}")
    
    if np.any(np.isnan(Sigma_psd)) or np.any(np.isinf(Sigma_psd)):
        if verbose:
            print(f"ERROR: PGD already returned NaN/inf!")
            print(f"  Input Sigma_obs - NaN: {np.any(np.isnan(Sigma_obs))}, Inf: {np.any(np.isinf(Sigma_obs))}")
            print(f"  Weight matrix - NaN: {np.any(np.isnan(weight_matrix)) if weight_matrix is not None else 'None'}")
            print(f"  Weight matrix - Inf: {np.any(np.isinf(weight_matrix)) if weight_matrix is not None else 'None'}")
            if weight_matrix is not None:
                print(f"  Weight matrix - min: {np.min(weight_matrix[mask]):.2e}, max: {np.max(weight_matrix[mask]):.2e}")
        return Sigma_psd

    # Stage B: Max Entropy Optimization
    if verbose:
        print("\nRunning Max Entropy Refinement...")
    
    try:
        # Direct method (using your existing optimization pipeline)
        rows, cols = inds[:, 0], inds[:, 1]
        Sigma_psd_vec = Sigma_psd[rows, cols]
        
        z_init = initialize_z(Sigma_psd, inds)
        z_opt, g_vals, obj_vals = optimize_lbfgs(Sigma_psd_vec, z_init, inds, d, max_iters=1000, verbose=False)
        
        K_z_opt = get_K(z_opt, inds, d)
        
        Sigma_opt_before = np.linalg.inv(K_z_opt)
        Sigma_opt = project_to_PSD(Sigma_opt_before)  # Ensure PSD

    except Exception as e:
        if verbose:
            print(f"\nMax Entropy FAILED with exception: {str(e)}")
            print(f"Returning PGD-only result from Sigma_psd")
            print(f"Sigma_psd has NaN: {np.any(np.isnan(Sigma_psd))}, has Inf: {np.any(np.isinf(Sigma_psd))}")
            print("Optimization failed: {str(e)}")
            print("Falling back to PGD-only result")
        return Sigma_psd  # Return PGD result instead of None

    return Sigma_opt


class IterativePrivateGGM:
    def __init__(self, rng, d, rho, alpha, n_measurements, sensitivity_type, norm_bound, optimizer="pgd-maxent"):
        """
        Initialize Iterative Private GGM
        
        Args:
            rng: Random number generator
            d (int): Number of features (dimension)
            rho (float): Privacy parameter  
            alpha (float): Budget split parameter (default 0.2)
            n_measurements (int): Number of off-diagonal entries to measure adaptively
            sensitivity_type (str): Type of sensitivity to use ('l2' or 'linf')
            norm_bound (float): R for L2 norm bound or B for Linf norm bound
            optimizer (str): Optimization method to use. Default is pgd-maxent
        """
        self.rng = rng
        self.d = d
        self.rho = rho
        self.alpha = alpha
        self.n_measurements = n_measurements
        self.sensitivity_type = sensitivity_type
        self.norm_bound = norm_bound
        self.optimizer = optimizer
        
        # Split privacy budget
        # For now, only split for covariance. TO-D0: add mean if needed
        # self.rho_m = self.rho * alpha / 2  # mean estimation
        # self.rho_v = self.rho * alpha / 2  # variance estimation
        self.rho_v = self.rho * alpha
        # Following AIM paper: use 10% of budget for selection, 90% for measurement
        # AIM paper suggests 10% for selection is a good rule of thumb
        self.rho_c = self.rho * (1-alpha) * 0.9  # covariance measurement (90% of remaining budget)
        self.rho_s = self.rho * (1-alpha) * 0.1  # selection mechanism (10% of remaining budget)
        
        # Init storage for measurements
        self.measured_entries = set()  
        self.Sigma_tilde = np.zeros((d, d))  # noisy measurements
        self.noise_scales = np.zeros((d, d))  # noise scale σ_{jk} for each measured entry
        self.empirical_cov = None
        
        # Track noise scales used (for debugging/analysis)
        self.actual_diag_noise_scale = None
        self.actual_offdiag_noise_scale = None

    def get_sensitivity_cov_jk(self, n):
        """Get the sensitivity for single entry measurements"""
        from privateggm.privacy import cov_jk_sensitivity_l2, cov_jk_sensitivity_linf
        
        if self.sensitivity_type == 'l2':
            return cov_jk_sensitivity_l2(n, self.norm_bound)
        elif self.sensitivity_type == 'linf':
            return cov_jk_sensitivity_linf(n, self.norm_bound)
        else:
            raise ValueError(f"Unknown sensitivity type: {self.sensitivity_type}")

    def get_score_sensitivity(self, n):
        """
        Compute sensitivity for the scoring function |Sigma_jk(X) - hat_Sigma_jk|
        This needs careful analysis based on your sensitivity bounds
        """
        # For L2 bounded data: the sensitivity of a single covariance entry
        base_sensitivity = self.get_sensitivity_cov_jk(n)
        # The score function has sensitivity base_sensitivity since it's |f(X) - constant|
        # TO-DO: double check analysis
        return base_sensitivity

    def compute_entry_scores(self, Sigma_hat):
        """
        Compute scores for ALL off-diagonal entries (allows remeasuring)
        Score is based on discrepancy between empirical and current estimate
        """
        scores = np.zeros((self.d, self.d))
        
        # Score all off-diagonal entries (both measured and unmeasured)
        for i in range(self.d):
            for j in range(i+1, self.d):  # Only upper triangular, off-diagonal
                score = np.abs(self.empirical_cov[i,j] - Sigma_hat[i,j])
                # Handle NaN values by using a large score (high priority to measure)
                if np.isnan(score):
                    score = np.abs(self.empirical_cov[i,j])  # Use empirical value as score
                scores[i,j] = score
                    
        return scores

    def fit(self, X):
        """
        Fit the private GGM model to data
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        """
        n, d = X.shape
        assert d == self.d
        
        # Center the data first
        X_centered = X - X.mean(axis=0)
        
        # Compute empirical covariance matrix once and store it
        # CHANGED IT TO CENTERED
        self.empirical_cov = (X_centered.T @ X_centered) / n
        
        # 1. Private mean estimation (assuming centered for now)
        # TODO: Implement if needed
        
        # 2. Private diagonal entries - measure all at once
        diag_sensitivity = self.get_sensitivity_cov_jk(n)
        for i in range(d):
            self.Sigma_tilde[i,i], _, noise_scale = gaussian_mechanism(
                self.rng,
                self.empirical_cov[i,i],
                diag_sensitivity, 
                self.rho_v / d
            )
            self.noise_scales[i,i] = noise_scale  # save noise scale for this entry
            self.measured_entries.add((i,i))
        self.actual_diag_noise_scale = noise_scale  # Save for output
        print(f"%%%%% noise scale PGD-MaxEnt (diagonal): {noise_scale} %%%%%")

        # 3. Iterative off-diagonal measurements
        cov_entry_sensitivity = self.get_sensitivity_cov_jk(n)
        score_sensitivity = self.get_score_sensitivity(n)
        
        # Print expected noise scale for off-diagonal measurements (constant across iterations)
        if self.n_measurements > 0:
            expected_offdiag_noise_scale = cov_entry_sensitivity / np.sqrt(2 * self.rho_c / self.n_measurements)
        else:
            expected_offdiag_noise_scale = 0.0  # No off-diagonal measurements
        
        for t in range(self.n_measurements):
            print(f"Iteration {t+1}/{self.n_measurements}")
            
            # Get current estimate using consistent optimization method
            Sigma_hat = self.get_current_estimate()
            
            # Compute scores for unmeasured off-diagonal entries
            scores = self.compute_entry_scores(Sigma_hat)
            
            # Flatten only the upper triangular part for selection
            # Note: We allow remeasuring entries to refine estimates (as per AIM paper)
            candidate_pairs = []
            candidate_scores = []
            for i in range(self.d):
                for j in range(i+1, self.d):
                    candidate_pairs.append((i,j))
                    candidate_scores.append(scores[i,j])
            
            if not candidate_pairs:
                print(f"No more entries to measure at iteration {t+1}")
                break
                
            # Select entry using exponential mechanism
            selected_idx = exponential_mechanism(
                self.rng,
                np.array(candidate_scores),
                score_sensitivity,
                self.rho_s / self.n_measurements
            )
            
            i, j = candidate_pairs[selected_idx]
            print(f"Selected entry ({i},{j}) with score {candidate_scores[selected_idx]:.6f}")
            
            # Measure selected entry using stored empirical covariance
            empirical_cov_ij = self.empirical_cov[i,j]
            noisy_cov, noise_sample, noise_scale = gaussian_mechanism(
                self.rng,
                empirical_cov_ij,
                cov_entry_sensitivity,
                self.rho_c / self.n_measurements
            )
            self.actual_offdiag_noise_scale = noise_scale  # Save for output
            
            self.Sigma_tilde[i,j] = noisy_cov
            self.Sigma_tilde[j,i] = noisy_cov  # Symmetric
            self.noise_scales[i,j] = noise_scale  # Save noise scale for this entry
            self.noise_scales[j,i] = noise_scale  # Symmetric
            self.measured_entries.add((i,j))
            
            print(f"Empirical: {empirical_cov_ij:.6f}, Noisy: {noisy_cov:.6f}, Noise std: {noise_scale:.6f}")

        print(f"%%%%% noise scale PGD-MaxEnt (measurement): {expected_offdiag_noise_scale} %%%%%")
        
        # Final estimate
        self.Sigma_hat = self.get_current_estimate()
        return self.Sigma_hat

    def get_current_estimate(self):
        """Get current covariance estimate using specified optimization method"""
        if self.optimizer == "pgd-maxent":
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            return pgd_maxent(self.Sigma_tilde, inds, noise_scales=self.noise_scales, max_iters=1000, eta=0.1, tol=1e-8, verbose=False)
        
        elif self.optimizer == "lbfgs":
            # Convert measured_entries to array format expected by optimize_lbfgs
            measured_array = np.array([[i,j] for i,j in self.measured_entries])
            z_init = initialize_z(self.Sigma_tilde, measured_array)
            
            z_opt, _, _ = optimize_lbfgs(
                self.Sigma_tilde[measured_array[:,0], measured_array[:,1]], 
                z_init, 
                measured_array, 
                self.d, 
                max_iters=100, 
                verbose=False
            )
            K_z = get_K(z_opt, measured_array, self.d)
            return np.linalg.inv(K_z)
        
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")


class AnnealingIterativePrivateGGM(IterativePrivateGGM):
    """
    Iterative Private GGM with Annealing
    """
    
    def __init__(self, rng, d, rho, alpha, beta, max_rounds, sensitivity_type, norm_bound, 
                 optimizer="pgd-maxent", anneal_threshold_factor=None, allow_remeasurement=True):
        """
        Initialize Annealing Iterative Private GGM
        
        Args:
            rng: Random number generator
            d (int): Number of features (dimension)
            rho (float): Total privacy budget
            alpha (float): Fraction of budget for diagonal measurements
            beta (float): Within each round, fraction for selection vs measurement
            max_rounds (int): Number of adaptive measurement rounds
            sensitivity_type (str): Type of sensitivity to use ('l2' or 'linf')
            norm_bound (float): R for L2 norm bound or B for Linf norm bound
            optimizer (str): Optimization method to use
            anneal_threshold_factor (float): Convergence threshold as multiple of measurement std.
                                            Default: sqrt(2/pi) ≈ 0.798
            allow_remeasurement (bool): If True, allow remeasuring entries to refine estimates.
                                       If False, each entry measured only once.
        """
        # Initialize parent with dummy n_measurements (we'll override fit anyway)
        super().__init__(rng, d, rho, alpha, max_rounds, sensitivity_type, norm_bound, optimizer)
        
        self.beta = beta
        self.max_rounds = max_rounds
        self.allow_remeasurement = allow_remeasurement
        
        # Annealing threshold: diff <= threshold_factor * measurement_std triggers doubling
        self.anneal_threshold_factor = anneal_threshold_factor or np.sqrt(2 / np.pi)
        
        # Track per-round budgets (will be updated dynamically)
        self.rho_per_round_select = None
        self.rho_per_round_measure  = None
        self.rho_used = 0.0
        
    def fit(self, X):
        """
        Fit with annealing: dynamically adjust budgets based on convergence
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        """
        n, d = X.shape
        assert d == self.d
        
        # Center the data (NOTE: mean will eventually be released privately!)
        X_centered = X - X.mean(axis=0)
        self.empirical_cov = (X_centered.T @ X_centered) / n

        # Initialize storage for measurements and noise_scales
        self.measurements = {}
        self.noise_scales = {}
        
        # ========================================
        # 1. Measure diagonal entries
        # ========================================
        rho_diag_init = self.rho * self.alpha
        rho_per_diag = rho_diag_init / d
        
        diag_sensitivity = self.get_sensitivity_cov_jk(n)
        for i in range(d):
            self.Sigma_tilde[i,i], _, noise_scale = gaussian_mechanism(
                self.rng,
                self.empirical_cov[i,i],
                diag_sensitivity,
                rho_per_diag
            )
            self.noise_scales[(i,i)] = noise_scale
            self.measurements[(i,i)] = self.Sigma_tilde[i,i]
            self.measured_entries.add((i,i))  # Track diagonal measurements
        
        self.rho_used = rho_diag_init
        self.actual_diag_noise_scale = noise_scale
        print(f"%%%%% Diagonal noise scale: {noise_scale} (rho_used: {self.rho_used:.6f}) %%%%%")
        
        # Get initial model estimate with just diagonals
        print("\nComputing initial model estimate with diagonal entries...")
        Sigma_hat_init = self.get_current_estimate()
        self.Sigma_hat = Sigma_hat_init  # Store initial estimate
        print(f"Initial estimate computed (condition number: {np.linalg.cond(Sigma_hat_init):.2e})")
        
        # ========================================
        # 2. Initialize per-round budgets
        # ========================================
        remaining_rho = self.rho - self.rho_used
        rho_per_round = remaining_rho / self.max_rounds
        
        self.rho_per_round_select = rho_per_round * self.beta
        self.rho_per_round_measure  = rho_per_round * (1 - self.beta)
        
        print(f"Initial per-round budget: select={self.rho_per_round_select:.6f}, measure={self.rho_per_round_measure :.6f}")
        
        # ========================================
        # 3. Iterative measurements with annealing
        # ========================================
        cov_entry_sensitivity = self.get_sensitivity_cov_jk(n)
        score_sensitivity = self.get_score_sensitivity(n)
        
        round_num = 0
        Sigma_hat_old = Sigma_hat_init  # Use initial diagonal-only estimate for first round

        while self.rho_used < self.rho and round_num < self.max_rounds:
            round_num += 1
            print(f"\n========== Round {round_num}/{self.max_rounds} (rho_used: {self.rho_used:.6f}/{self.rho:.6f}) ==========")
            
            # Check if this is the final round
            # avoid leaving spare budget
            remaining_budget = self.rho - self.rho_used
            if remaining_budget < 2 * (self.rho_per_round_select + self.rho_per_round_measure):
                print("Final round: reallocating remaining budget")
                self.rho_per_round_select = remaining_budget * self.beta
                self.rho_per_round_measure  = remaining_budget * (1 - self.beta)
            
            # Compute scores for candidate entries (using previous round's estimate)
            scores = self.compute_entry_scores(Sigma_hat_old)

            # Collect candidate entries (if remeasure allowed, diagonal included)
            candidate_pairs = []
            candidate_scores = []
            for i in range(self.d):
                for j in range(i, self.d):  # Start from i to include diagonal
                    # If remeasurement not allowed, skip already measured entries
                    if not self.allow_remeasurement and (i, j) in self.measurements:
                        continue
                    candidate_pairs.append((i,j))
                    candidate_scores.append(scores[i,j])
            
            # Check if no candidates available (when remeasurement disabled and all measured)
            if len(candidate_pairs) == 0:
                print("No more entries to measure (all entries measured, remeasurement disabled)")
                break
            
            # compute score sensitivity
            score_sensitivity = self.get_score_sensitivity(n)
                
            # Select entry using exponential mechanism
            selected_idx = exponential_mechanism(
                self.rng,
                np.array(candidate_scores),
                score_sensitivity,
                self.rho_per_round_select
            )
            
            i, j = candidate_pairs[selected_idx]
            
            self.rho_used += self.rho_per_round_select
            print(f"Selected entry ({i},{j}) with score {candidate_scores[selected_idx]:.6f}")
            
            # ========================================
            # Measure with Gaussian mechanism
            # ========================================
            empirical_cov_ij = self.empirical_cov[i,j]
            noisy_cov, _, new_noise_scale = gaussian_mechanism(
                self.rng,
                empirical_cov_ij,
                cov_entry_sensitivity,
                self.rho_per_round_measure 
            )
            self.rho_used += self.rho_per_round_measure 
            
            # ========================================
            # Combine with previous measurement if exists
            # ========================================
            if (i,j) in self.measurements:
                # Already measured: combine using inverse-variance weighting
                old_measurement = self.measurements[(i,j)]
                old_sigma2 = self.noise_scales[(i,j)]**2
                new_sigma2 = new_noise_scale**2
                
                # Combined variance: 1/σ²_combined = 1/σ²_old + 1/σ²_new
                # So: σ²_combined = 1 / (1/σ²_old + 1/σ²_new)
                updated_sigma2 = 1.0 / (1.0/old_sigma2 + 1.0/new_sigma2)
                updated_measurement = ((old_measurement / old_sigma2) + (noisy_cov / new_sigma2)) * updated_sigma2
                
                print(f"Remeasured ({i},{j}): old={old_measurement:.6f} (σ={np.sqrt(old_sigma2):.6f}), " +
                      f"new={noisy_cov:.6f} (σ={new_noise_scale:.6f}), " +
                      f"combined={updated_measurement:.6f} (σ={np.sqrt(updated_sigma2):.6f})")

                # store information
                self.Sigma_tilde[i,j] = updated_measurement
                self.Sigma_tilde[j,i] = updated_measurement
                self.measurements[(i,j)] = updated_measurement
                self.noise_scales[(i,j)] = np.sqrt(updated_sigma2)
                self.noise_scales[(j,i)] = np.sqrt(updated_sigma2)

                new_noise_scale = np.sqrt(updated_sigma2) # TO DO check if it needs to be the updated or raw one

            else:

                # First measurement of this entry, store information
                self.Sigma_tilde[i,j] = noisy_cov
                self.Sigma_tilde[j,i] = noisy_cov
                self.measurements[(i,j)] = noisy_cov
                self.noise_scales[(i,j)] = new_noise_scale
                self.noise_scales[(j,i)] = new_noise_scale
                self.measured_entries.add((i,j))
                
                print(f"First measurement ({i},{j}): empirical={empirical_cov_ij:.6f}, " +
                      f"noisy={noisy_cov:.6f}, noise_std={new_noise_scale:.6f}")
            
            # ========================================
            # Update model and check convergence
            # ========================================
            Sigma_hat_new = self.get_current_estimate()
            
            # Check if optimization failed (returned previous estimate due to NaN) TO-DO: review
            if np.any(np.isnan(Sigma_hat_new)) or np.any(np.isinf(Sigma_hat_new)):
                print("WARNING: Optimization returned NaN/inf - skipping this round, keeping previous estimate")
                # Budget already spent, measurement already stored
                # Don't update Sigma_hat_old, don't trigger annealing
                # Continue to next round
                continue
            
            old_estimate = Sigma_hat_old[i,j]
            new_estimate = Sigma_hat_new[i,j]
            diff = np.abs(new_estimate - old_estimate)
            
            threshold = self.anneal_threshold_factor * new_noise_scale
            print(f"MEASUREMENT UTILITY CHECK: |Δ estimate|={diff:.6f}, threshold={threshold:.6f}")
            
            # ========================================
            # Annealing: double budgets if small gain 
            # ========================================
            if diff <= threshold:
                print("Small gain: doubling future budgets (select x2, measure x4)")
                self.rho_per_round_select *= 2
                self.rho_per_round_measure  *= 4
            else:
                print("Large enough gain: keeping current budget allocation")
            
            # Update Sigma_hat_old for next iteration
            Sigma_hat_old = Sigma_hat_new
            
            # Track the last successful noise scale
            self.actual_offdiag_noise_scale = new_noise_scale
        
        print(f"\n%%%%% Final rho used: {self.rho_used:.6f}/{self.rho:.6f} %%%%%")
        
        # 4. Final estimate
        self.Sigma_hat = self.get_current_estimate()
        
        return self.Sigma_hat
    
    def get_current_estimate(self):
        """
        Override parent method to convert dictionary noise_scales to numpy array
        """
        # Convert dictionary noise_scales to numpy array
        noise_scales_array = np.zeros((self.d, self.d))
        for (i, j), scale in self.noise_scales.items():
            noise_scales_array[i, j] = scale
            if i != j:
                noise_scales_array[j, i] = scale  # Ensure symmetry
        
        if self.optimizer == "pgd-maxent":
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            Sigma_hat = pgd_maxent(self.Sigma_tilde, inds, noise_scales=noise_scales_array, 
                            max_iters=1000, eta=0.1, tol=1e-8, verbose=False)
            
            # TODO: double check with Dan
            # Check if pgd_maxent returned NaN/inf
            if np.any(np.isnan(Sigma_hat)) or np.any(np.isinf(Sigma_hat)):
                print("WARNING: pgd_maxent returned NaN/inf - keeping previous estimate")
                return self.Sigma_hat
            
            return Sigma_hat
        
        elif self.optimizer == "lbfgs":
            measured_array = np.array([[i,j] for i,j in self.measured_entries])
            z_init = initialize_z(self.Sigma_tilde, measured_array)
            
            z_opt, _, _ = optimize_lbfgs(
                self.Sigma_tilde[measured_array[:,0], measured_array[:,1]], 
                z_init, 
                measured_array, 
                self.d, 
                max_iters=100, 
                verbose=False
            )
            K_z = get_K(z_opt, measured_array, self.d)
            return np.linalg.inv(K_z)
        
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")