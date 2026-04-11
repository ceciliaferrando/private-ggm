import numpy as np
from scipy.linalg import solve, cholesky, eigh
from scipy.optimize import line_search
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
import optax
from functools import partial
from collections import deque

from privateggm.utils import *
from privateggm.privacy import gaussian_mechanism, exponential_mechanism, cov_jk_sensitivity_l2, cov_jk_sensitivity_linf

# Import optimized solvers
from privateggm.solvers import pgd_maxent as pgd_maxent_solver, ipm_chol
from privateggm.solve import solve as ggm_solve

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

def optimize_lbfgs(Sigma_tilde_vec, z, inds, d, max_iters=500, verbose=False):
    """
    Sigma_tilde_vec [np.array]: matrix of observed (privatized) covariance entries
    z [jnp.array]: vector supporting K(z)
    inds [np.array]: vector of indices of nonzero covariance entries
    d [int]: number of attributes
    """

    # rng_key = jrandom.PRNGKey(42)

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

    # DEBUG: Final K(z) construction diagnostics (commented out for speed)
    # K_z = get_K(z, inds, d)
    # eigvals = np.linalg.eigvalsh(K_z)
    # min_eigval = np.min(eigvals)
    # max_eigval = np.max(eigvals)
    # cond_number = max_eigval / min_eigval if min_eigval > 1e-10 else np.inf
    # is_non_pd = not jnp.all(eigvals > 0)
    # is_ill_conditioned = cond_number > 1e10
    # is_nearly_singular = min_eigval < 1e-8
    # if is_non_pd or is_ill_conditioned or is_nearly_singular:
    #     print("-" * 60)
    #     print("Issue detected with optimized K(z):")
    #     print(f"  • Non-PD:            {is_non_pd}")
    #     print(f"  • Ill-conditioned:   {is_ill_conditioned} (cond #: {cond_number:.2e})")
    #     print(f"  • Nearly singular:   {is_nearly_singular} (min eigval: {min_eigval:.2e})")
    #     print("-" * 60)

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


def bb_step_size(Sig, Sig_prev, g, g_prev, accelerate=1.5):
    """
    Compute Barzilai-Borwein step size with optional acceleration.
    
    Args:
        Sig: Current point
        Sig_prev: Previous point
        g: Current gradient
        g_prev: Previous gradient
        accelerate: Acceleration factor (>1 means more aggressive steps)
        
    Returns:
        step: BB step size
    """
    s = Sig - Sig_prev
    y = g - g_prev
    
    # First BB formula: α = (s^T s) / (s^T y)
    s_norm_sq = np.sum(s * s)
    sy = np.sum(s * y)
    
    # Safeguard: ensure positive step size
    if sy > 1e-10:
        alpha = s_norm_sq / sy
        # Apply acceleration factor
        alpha = alpha * accelerate
        # Clip to reasonable range [0.001, 50.0] (higher upper bound for speed)
        alpha = np.clip(alpha, 0.001, 50.0)
    else:
        alpha = 0.1 * accelerate  # Fallback with acceleration
    
    return alpha


def projected_bb_line_search(Sig, g, Sig_prev, g_prev, obj_fn, project_fn, debug_iter=None):
    """
    Projected Barzilai-Borwein line search with acceleration - uses BB step size with projection.
    
    Args:
        Sig: Current point (matrix)
        g: Current gradient
        Sig_prev: Previous point
        g_prev: Previous gradient
        obj_fn: Objective function
        project_fn: Projection function (e.g., project_to_PSD)
        debug_iter: If provided, print debug info for this iteration
        
    Returns:
        Sig_new: New point after line search
        step: Step size used
        ls: Number of iterations (always 1 for BB)
    """
    debug_this = debug_iter is not None and debug_iter <= 3
    
    # Compute BB step size with acceleration (1.5x)
    step = bb_step_size(Sig, Sig_prev, g, g_prev, accelerate=1.5)
    
    if debug_this:
        print(f"    [BB LS iter {debug_iter}] BB step size: {step:.6e}")
    
    # Take step
    Sig_candidate = Sig - step * g
    
    if debug_this and project_fn is not None:
        print(f"    [BB LS iter {debug_iter}] Before projection: diag={np.diag(Sig_candidate)[:3]}")
    
    # Apply projection
    Sig_new = project_fn(Sig_candidate)
    
    if debug_this and project_fn is not None:
        print(f"    [BB LS iter {debug_iter}] After projection: diag={np.diag(Sig_new)[:3]}")
        print(f"    [BB LS iter {debug_iter}] Objective: {obj_fn(Sig_new):.6e}")
    
    return Sig_new, step, 1  # BB takes exactly 1 iteration (no backtracking)

def projected_line_search(Sig, g, obj_fn, project_fn, prev_step=None, max_bt=20):
    """
    Backtracking line search with projection.
    Uses 1.1 * prev_step as initial guess if available.
    
    Args:
        Sig: Current point (matrix)
        g: Current gradient
        obj_fn: Objective function
        project_fn: Projection function (e.g., project_to_PSD)
        prev_step: Step size from previous iteration (for warm start)
        max_bt: Maximum backtracking iterations
        
    Returns:
        Sig_new: New point after line search
        step: Step size used
        ls: Number of line search iterations
    """
    # 1. SET THE INITIAL STEP FOR THIS ITERATION
    if prev_step is None or prev_step < 1e-10:
        # If no history, or if the step has decayed into nothingness,
        # we must reset to a reasonable starting value 
        step = 0.1 
    else:
        # Start at 1.1x of the last successful step (warm start)
        step = prev_step * 1.1
    
    obj_current = obj_fn(Sig)
    
    # 2. BACKTRACKING LOOP
    for bt_iter in range(max_bt):
        # Apply step and project 
        Sig_candidate = Sig - step * g
        Sig_new = project_fn(Sig_candidate)
        obj_new = obj_fn(Sig_new)
        
        # 3. SUCCESS CONDITION: Armijo condition
        predicted_decrease = -np.sum(g * (Sig_new - Sig))
        if obj_new <= obj_current - 0.5 * predicted_decrease:
            # Progress: return this step so the NEXT call
            # can use 1.1 * this_step as its starting point.
            return Sig_new, step, bt_iter + 1
        
        # 4. FAIL CONDITION: Shrink step size
        step *= 0.5
    
    # If we exhaust max_bt without progress, return best attempt
    # This happens when gradient is very small or we're already at optimum
    Sig_candidate = Sig - step * g
    Sig_new = project_fn(Sig_candidate)
    return Sig_new, step, max_bt

def gold_standard_line_search(Sig, g, Sig_prev, g_prev, obj_fn, project_fn, history_objs, max_bt=20):
    """
    Improved Projected Line Search: BB-initialization + Non-monotone Backtracking.
    
    Args:
        history_objs: A deque(maxlen=5) containing recent objective values.
    """
    # 1. DYNAMIC INITIALIZATION (The "Smart" 1.1x)
    # Instead of 1.1, we use the BB formula to sense the local curvature
    s = Sig - Sig_prev
    y = g - g_prev
    sy = np.sum(s * y)
    
    if sy > 1e-10:
        step = np.sum(s * s) / sy  # BB Step
    else:
        step = 0.5 # Fallback if geometry is flat
    
    # Safety clip to prevent extreme jumps
    step = np.clip(step, 1e-4, 50.0)
    
    # 2. NON-MONOTONE REFERENCE
    # Simple LS fails because it's too strict (obj_new < obj_current).
    # We compare to the MAX of the last 5 iterations to allow "breathing room."
    f_max = np.max(history_objs) if len(history_objs) > 0 else obj_fn(Sig)
    
    # 3. THE PROJECTED BACKTRACKING LOOP
    for bt_iter in range(max_bt):
        # Step + Projection (Crucial: Project inside the loop!)
        Sig_candidate = project_fn(Sig - step * g)
        obj_candidate = obj_fn(Sig_candidate)
        
        # Check against the "High Water Mark" (f_max)
        if obj_candidate <= f_max:
            return Sig_candidate, step, obj_candidate, bt_iter + 1
        
        # Shrink if we didn't beat the high water mark
        step *= 0.5
        
    # Final fallback: return the last attempt
    return Sig_candidate, step, obj_candidate, max_bt


def projected_gradient_descent(S_observed, mask, weights=None, eta=0.1, max_iters=5000, tol=1e-8, verbose=False):
    """
    Projected gradient descent with backtracking line search.
    """
    import time
    start_time = time.time()
    
    d = S_observed.shape[0]
    assert mask.shape == S_observed.shape
    
    # Setup
    weight_matrix = np.ones_like(S_observed) if weights is None else weights.copy()
    S = np.zeros_like(S_observed)
    S[mask] = S_observed[mask]
    
    # Define objective and gradient functions early for debug
    def obj(Sig): return 0.5 * np.sum(weight_matrix[mask] * (Sig[mask] - S[mask])**2)
    
    def grad(Sig):
        g = np.zeros_like(Sig)
        g[mask] = weight_matrix[mask] * (Sig[mask] - S[mask])
        g += np.diag(np.diag(g))  # Double diagonal
        return g
    
    # DEBUG: Check if S needs PSD projection
    S_eigvals = np.linalg.eigvalsh(S)
    needs_projection = np.any(S_eigvals < 0)
    if needs_projection:
        print(f"  [PGD INIT] S needs PSD projection")
    
    Sigma = project_to_PSD(S, thresh=1e-3)
    
    if needs_projection and verbose:
        print(f"  [PGD INIT] S diagonal before projection: {np.diag(S)}")
        print(f"  [PGD INIT] Sigma diagonal after projection: {np.diag(Sigma)}")
        
        # Check objective and gradient at initial point
        init_obj = obj(Sigma)
        init_grad = grad(Sigma)
        init_grad_norm = np.linalg.norm(init_grad)
        print(f"  [PGD INIT] Initial objective: {init_obj:.6e}")
        print(f"  [PGD INIT] Initial gradient norm: {init_grad_norm:.6e}")
        print(f"  [PGD INIT] Initial gradient diagonal: {np.diag(init_grad)}")
    
    # # COMMENTED OUT: Random control for line search validation
    # # Create random control (fixed seed, truly random PSD matrix)
    # rng = np.random.default_rng(42)
    # A = (rng.normal(0, 1, (d, d)) + rng.normal(0, 1, (d, d)).T) / 2
    # eigvals, eigvecs = np.linalg.eigh(A)
    # # Use random eigenvalues scaled to be similar magnitude to S
    # random_eigvals = np.abs(eigvals) * np.mean(np.abs(np.diag(S)))
    # Sigma_rand = eigvecs @ np.diag(random_eigvals + 0.1) @ eigvecs.T
    # Sigma_rand = project_to_PSD(Sigma_rand, thresh=1e-3)
    
    # Projection function for line search
    def project_psd(Sig):
        return project_to_PSD(Sig, thresh=1e-3)
    
    prev = Sigma.copy()
    g_prev = grad(Sigma)
    conv = {'objectives': [], 'grad_norms': [], 'step_norms': [], 'ls_iters': [], 'converged': False}
    
    # No progress tracking
    no_progress_count = 0
    no_progress_threshold = 100  # Stop if no progress for 100 iterations
    
    # Initialize history for non-monotone line search (gold standard)
    history_objs = deque(maxlen=5)
    history_objs.append(obj(Sigma))
    
    # Main loop
    for t in range(max_iters):
        # Current gradient
        g = grad(Sigma)
        g_norm = np.linalg.norm(g)
        obj_before = obj(Sigma)
        
        # Use gold standard line search (BB initialization + non-monotone backtracking)
        Sig_new, step, obj_candidate, ls = gold_standard_line_search(
            Sigma, g, prev, g_prev, obj, project_fn=project_psd, 
            history_objs=history_objs, max_bt=20
        )
        
        obj_after = obj(Sig_new)
        history_objs.append(obj_after)
        
        conv['objectives'].append(obj_after)
        conv['grad_norms'].append(g_norm)
        conv['step_norms'].append(np.linalg.norm(Sig_new - prev, 'fro'))
        conv['ls_iters'].append(ls)
        
        # Print every 100 iterations instead of every iteration
        if (t + 1) % 100 == 0:
            print(f"  [LS {t+1:4d}] obj={obj_after:.2e} grad={g_norm:.2e} Δ={conv['step_norms'][-1]:.2e} α={step:.2e}")
        
        # Check for no progress (objective not changing)
        if abs(obj_after - obj_before) < 1e-12:
            no_progress_count += 1
            if no_progress_count >= no_progress_threshold:
                conv['converged'] = True
                conv['termination_reason'] = f'no_progress_{no_progress_threshold}_iters'
                print(f"  [LS] No progress for {no_progress_count} iterations (Δobj < 1e-12)")
                break
        else:
            no_progress_count = 0
        
        # Check convergence based on gradient norm
        if not conv['converged'] and g_norm < 1e-10:
            conv['converged'] = True
            conv['termination_reason'] = 'gradient_converged'
            print(f"  [LS] Converged at iter {t+1} (grad={g_norm:.2e})")
            break
        
        # Update for next iteration
        prev = Sigma.copy()
        g_prev = g.copy()
        Sigma = Sig_new
    
    conv['n_iters'] = t + 1
    # conv_rand['n_iters'] = t + 1
    
    # Add timing info (just the good initialization now, random control commented out)
    elapsed_time = time.time() - start_time
    conv['time_seconds'] = elapsed_time
    
    # Add termination reason if not set
    if 'termination_reason' not in conv:
        conv['termination_reason'] = 'max_iters_reached'
    
    # Verify KKT conditions at final point
    final_gradient = grad(Sigma)
    is_optimum, kkt_dict = verify_pgd_optimum(Sigma, final_gradient, tol=1e-4)
    conv['kkt'] = {
        'is_optimum': bool(is_optimum),
        **{k: float(v) for k, v in kkt_dict.items()}
    }
    
    # Convert numpy arrays to lists for JSON serialization
    for key in ['objectives', 'grad_norms', 'step_norms', 'ls_iters']:
        if key in conv:
            conv[key] = [float(x) for x in conv[key]]
        # if key in conv_rand:
        #     conv_rand[key] = [float(x) for x in conv_rand[key]]
    
    # # Add random control convergence info to main dict
    # conv['random_control'] = conv_rand
    
    return Sigma, conv

def pgd_maxent(Sigma_tilde, inds, noise_scales=None, max_iters=1000, eta=0.1, tol=1e-8, verbose=False):
    """
    Two-step PGD + Max Entropy pipeline.

    Args:
        Sigma_true: Ground truth covariance matrix
        Sigma_tilde: Observed noisy covariance matrix
        inds: Array of observed indices [[i,j], ...]
        noise_scales: Noise scale matrix for weighted optimization
        max_iters: Max iterations for PGD
        eta: Step size for PGD
        tol: Convergence tolerance for PGD
        verbose: Print progress
        
    Returns:
        Sigma_opt: Optimized covariance matrix
        full_convergence_info: Dictionary with convergence info from both stages
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
        
        # Standard inverse-variance weighting (no compression)
        weight_matrix[mask] = raw_weights
        
        # Normalize weights so maximum is 1 (prevents numerical issues)
        max_weight = np.max(weight_matrix[mask])
        if max_weight > 0:
            weight_matrix[mask] = weight_matrix[mask] / max_weight
    else:
        weight_matrix = None

    full_convergence_info = {}

    # Stage A: Projected Gradient Descent
    if verbose:
        print("\n" + "="*60)
        print("STAGE A: Projected Gradient Descent")
        print("="*60)
    
    Sigma_psd, pgd_convergence = projected_gradient_descent(
        Sigma_obs, mask, weights=weight_matrix, eta=eta, max_iters=max_iters, tol=tol, verbose=verbose
    )
    
    full_convergence_info['pgd'] = pgd_convergence
    
    # Print PGD summary
    if verbose:
        print("\nPGD Summary:")
        print(f"  Converged: {pgd_convergence['converged']}")
        print(f"  Iterations: {pgd_convergence['n_iters']}")
        print(f"  Termination: {pgd_convergence['termination_reason']}")
        print(f"  Final objective: {pgd_convergence['objectives'][-1]:.6e}")
        print(f"  Final grad norm: {pgd_convergence['grad_norms'][-1]:.6e}")
        print(f"  Avg line search iters: {np.mean(pgd_convergence['ls_iters']):.2f}")
        print(f"  Time: {pgd_convergence['time_seconds']:.2f} seconds")
    
    # DEBUG: Check PGD result
    if verbose:
        print(f"\nPGD result check:")
        print(f"  Value range: [{np.min(Sigma_psd):.6f}, {np.max(Sigma_psd):.6f}]")
    
    if np.any(np.isnan(Sigma_psd)) or np.any(np.isinf(Sigma_psd)):
        if verbose:
            print(f"\nERROR: PGD returned NaN/inf!")
            print(f"  Input Sigma_obs - NaN: {np.any(np.isnan(Sigma_obs))}, Inf: {np.any(np.isinf(Sigma_obs))}")
            if weight_matrix is not None:
                print(f"  Weight matrix range: [{np.min(weight_matrix[mask]):.2e}, {np.max(weight_matrix[mask]):.2e}]")
        return Sigma_psd, full_convergence_info

    # Stage B: Max Entropy Optimization
    if verbose:
        print("\n" + "="*60)
        print("STAGE B: Max Entropy Refinement (L-BFGS)")
        print("="*60)
    
    import time
    lbfgs_start_time = time.time()
    
    try:
        # Direct method (using your existing optimization pipeline)
        rows, cols = inds[:, 0], inds[:, 1]
        Sigma_psd_vec = Sigma_psd[rows, cols]
        
        z_init = initialize_z(Sigma_psd, inds)
        z_opt, g_vals, obj_vals = optimize_lbfgs(Sigma_psd_vec, z_init, inds, d, max_iters=1000, verbose=verbose)
        
        lbfgs_elapsed = time.time() - lbfgs_start_time
        
        full_convergence_info['lbfgs'] = {
            'grad_vals': g_vals,
            'obj_vals': obj_vals,
            'n_iters': len(obj_vals),
            'converged': len(g_vals) > 0 and g_vals[-1] < 1e-8,
            'time_seconds': lbfgs_elapsed
        }
        
        if verbose:
            print(f"\nL-BFGS Summary:")
            print(f"  Iterations: {len(obj_vals)}")
            print(f"  Final objective: {obj_vals[-1]:.6e}")
            print(f"  Final grad sum: {g_vals[-1]:.6e}")
            print(f"  Time: {lbfgs_elapsed:.2f} seconds")
        
        K_z_opt = get_K(z_opt, inds, d)
        
        Sigma_opt_before = np.linalg.inv(K_z_opt)
        Sigma_opt = project_to_PSD(Sigma_opt_before)  # Ensure PSD
        
        if verbose:
            print(f"\nMax Entropy result check: completed")

    except Exception as e:
        if verbose:
            print(f"\nMax Entropy FAILED: {str(e)}")
            print(f"Returning PGD-only result")
        full_convergence_info['lbfgs'] = {'error': str(e)}
        return Sigma_psd, full_convergence_info

    return Sigma_opt, full_convergence_info


def verify_pgd_optimum(W, gradient, tol=1e-6):
    """
    Verify KKT conditions for PGD optimum on the PSD cone.
    
    Args:
        W: Matrix to verify (should be PSD)
        gradient: Gradient at the point
        tol: Tolerance threshold for KKT conditions
    """

    # 1. Primal Feasibility: W must be in the PSD cone (W >= 0)
    # We check the most negative eigenvalue.
    eig_W = np.linalg.eigvalsh(W)
    min_eig_W = np.min(eig_W)
    primal_resid = max(0, -min_eig_W)
    
    # 2. Dual Feasibility: The gradient must be in the Dual Cone (gradient >= 0)
    # In the case of the PSD cone, the dual cone is the PSD cone itself.
    eig_G = np.linalg.eigvalsh(gradient)
    min_eig_G = np.min(eig_G)
    dual_resid = max(0, -min_eig_G)
    
    # 3. Complementary Slackness: <gradient, W> = 0
    # Using the Frobenius Inner Product (trace(gradient.T @ W)) -> for sym matrices this is just np.sum(gradient * W)
    comp_slackness = np.abs(np.sum(gradient * W))
    
    # Summary logic
    is_optimum = (primal_resid < tol and 
                  dual_resid < tol and 
                  comp_slackness < tol)
    
    kkt_dict = {
        "primal_feasibility": primal_resid,
        "dual_feasibility": dual_resid,
        "complementary_slackness": comp_slackness,
        "min_eig_W": min_eig_W,
        "min_eig_grad": min_eig_G
    }
    
    return is_optimum, kkt_dict


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

    def compute_entry_scores(self, Sigma_hat, allow_remeasurement=False):
        """
        Compute scores for entries (allows remeasuring)
        Score is based on discrepancy between empirical and current estimate
        
        Args:
            Sigma_hat: Current covariance estimate
            allow_remeasurement: If True, include diagonal entries. If False, only off-diagonal.
        """
        scores = np.zeros((self.d, self.d))
        
        # Determine which entries to score
        if allow_remeasurement:
            # Include diagonal and upper triangular
            for i in range(self.d):
                for j in range(i, self.d):  # Start from i to include diagonal
                    score = np.abs(self.empirical_cov[i,j] - Sigma_hat[i,j])
                    scores[i,j] = score
        else:
            # Only off-diagonal entries
            for i in range(self.d):
                for j in range(i+1, self.d):  # Only upper triangular, off-diagonal
                    score = np.abs(self.empirical_cov[i,j] - Sigma_hat[i,j])
                    scores[i,j] = score
        
        # Handle NaN values: replace with 2x the max of non-NaN scores
        non_nan_scores = scores[~np.isnan(scores)]
        if len(non_nan_scores) > 0:
            max_non_nan = np.max(non_nan_scores)
            nan_value = 2.0 * max_non_nan
        else:
            # All scores are NaN, use a default large value
            nan_value = 1e6
        
        scores[np.isnan(scores)] = nan_value
                    
        return scores

    def fit(self, X, pre_centered=False):
        """
        Fit the private GGM model to data

        Args:
            X: Data matrix of shape (n_samples, n_features)
            pre_centered: if True, treat X as already centered (skip internal mean subtraction).
                          Use this when the caller has centered X with a private mean.
        """
        n, d = X.shape
        assert d == self.d

        # Center the data (skip if caller already centered with a private mean)
        X_centered = X if pre_centered else X - X.mean(axis=0)

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
        # Convert numpy array noise_scales to form expected by solvers
        noise_scales_array = self.noise_scales if isinstance(self.noise_scales, np.ndarray) else np.array(self.noise_scales)
        
        if self.optimizer == "pgd_maxent_solver":
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            rows, cols = inds[:, 0], inds[:, 1]
            vals = self.Sigma_tilde[rows, cols]
            tau  = noise_scales_array[rows, cols]

            Sigma_hat, convergence_hist = ggm_solve(
                rows, cols, vals, tau, self.d, solver='pgd_maxent',
                eta=0.1, pgd_iters=2000, pgd_eps=1e-8,
                lbfgs_iters=1000, eps=1e-8,
                track_kkt=False
            )

            if not hasattr(self, 'convergence_history'):
                self.convergence_history = []
            self.convergence_history.append({'solver': 'pgd_maxent_solver', 'hist': convergence_hist})

            return Sigma_hat

        elif self.optimizer == "pgd-maxent":
            # Old pgd_maxent from model.py (dense, no component decomposition)
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            Sigma_hat, convergence_hist = pgd_maxent(
                self.Sigma_tilde, inds, noise_scales=noise_scales_array,
                max_iters=5000, eta=0.1, tol=1e-8, verbose=False
            )

            if not hasattr(self, 'convergence_history'):
                self.convergence_history = []
            self.convergence_history.append({'solver': 'pgd_maxent_model', 'hist': convergence_hist})

            return Sigma_hat

        elif self.optimizer == "ipm_chol":
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            rows, cols = inds[:, 0], inds[:, 1]
            vals = self.Sigma_tilde[rows, cols]
            tau  = noise_scales_array[rows, cols]

            Sigma_hat, convergence_hist = ggm_solve(
                rows, cols, vals, tau, self.d, solver='ipm_chol',
                mu_init=1.0, mu_factor=0.5, eps=1e-8,
                max_outer=40, max_inner=200
            )

            if not hasattr(self, 'convergence_history'):
                self.convergence_history = []
            self.convergence_history.append({'solver': 'ipm_chol_solver', 'hist': convergence_hist})

            return Sigma_hat
        
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
                 optimizer="pgd-maxent", anneal_threshold_factor=None, allow_remeasurement=True,
                 target_budget_fraction=0.0):
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
        self.target_budget_fraction = target_budget_fraction
        
        # Annealing threshold: diff <= threshold_factor * measurement_std triggers doubling
        self.anneal_threshold_factor = anneal_threshold_factor or np.sqrt(2 / np.pi)
        
        # Track per-round budgets (will be updated dynamically)
        self.rho_per_round_select = None
        self.rho_per_round_measure  = None
        self.rho_used = 0.0
        
    def fit(self, X, pre_centered=False):
        """
        Fit with annealing: dynamically adjust budgets based on convergence

        Args:
            X: Data matrix of shape (n_samples, n_features)
            pre_centered: if True, treat X as already centered (skip internal mean subtraction).
                          Use this when the caller has centered X with a private mean.
        """
        import time
        trial_start_time = time.time()

        n, d = X.shape
        assert d == self.d

        # Center the data (skip if caller already centered with a private mean)
        X_centered = X if pre_centered else X - X.mean(axis=0)
        self.empirical_cov = (X_centered.T @ X_centered) / n

        # Initialize storage for measurements and noise_scales
        self.measurements = {}
        self.noise_scales = {}
        self.entry_to_rounds = {}  # Track which rounds each entry was measured in
        self.entry_to_noise_scales = {}  # Track raw noise scale for each round of measurement
        
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
            self.entry_to_rounds[(i,i)] = [0]  # Round 0 for diagonal
            self.entry_to_noise_scales[(i,i)] = [noise_scale]
        
        self.rho_used = rho_diag_init
        self.actual_diag_noise_scale = noise_scale
        print(f"%%%%% Diagonal noise scale: {noise_scale} (rho_used: {self.rho_used:.6f}) %%%%%")
        print(f"  True diagonal: {np.diag(self.empirical_cov)}")
        print(f"  Noisy diagonal: {np.diag(self.Sigma_tilde)}")

        # ========================================
        # 1b. Measure target pairs (last row/col) with dedicated budget
        # ========================================
        if self.target_budget_fraction > 0.0:
            rho_target = self.rho * self.target_budget_fraction
            rho_per_target = rho_target / (d - 1)
            target_col = d - 1
            target_sensitivity = diag_sensitivity  # same L-inf sensitivity: 2*T^2/n
            for j in range(d - 1):
                val, _, noise_scale_t = gaussian_mechanism(
                    self.rng,
                    self.empirical_cov[j, target_col],
                    target_sensitivity,
                    rho_per_target,
                )
                self.Sigma_tilde[j, target_col] = val
                self.Sigma_tilde[target_col, j] = val
                self.noise_scales[(j, target_col)] = noise_scale_t
                self.measurements[(j, target_col)] = val
                self.measured_entries.add((j, target_col))
                self.measured_entries.add((target_col, j))
                self.entry_to_rounds[(j, target_col)] = [0]
                self.entry_to_noise_scales[(j, target_col)] = [noise_scale_t]
            self.rho_used += rho_target
            print(f"%%%%% Target pairs noise scale: {noise_scale_t:.4e} (rho_used: {self.rho_used:.6f}) %%%%%")

        # Get initial model estimate with just diagonals (+ target pairs if applicable)
        print("\nComputing initial model estimate with diagonal entries...")
        Sigma_hat_init = self.get_current_estimate()
        self.Sigma_hat = Sigma_hat_init  # Store initial estimate
        self.Sigma_baseline = Sigma_hat_init.copy()  # Save baseline (diagonal-only estimate)
        
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
            scores = self.compute_entry_scores(Sigma_hat_old, allow_remeasurement=self.allow_remeasurement)

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
                self.entry_to_rounds[(i,j)].append(round_num)  # Track remeasurement
                self.entry_to_noise_scales[(i,j)].append(new_noise_scale)  # Raw noise scale for this round

                new_noise_scale = np.sqrt(updated_sigma2) # TO DO check if it needs to be the updated or raw one

            else:

                # First measurement of this entry, store information
                self.Sigma_tilde[i,j] = noisy_cov
                self.Sigma_tilde[j,i] = noisy_cov
                self.measurements[(i,j)] = noisy_cov
                self.noise_scales[(i,j)] = new_noise_scale
                self.noise_scales[(j,i)] = new_noise_scale
                self.measured_entries.add((i,j))
                self.entry_to_rounds[(i,j)] = [round_num]  # Track first measurement
                self.entry_to_noise_scales[(i,j)] = [new_noise_scale]
                
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
            
            # ========================================
            # Check measurement utility
            # ========================================
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
        
        # Record total trial time
        trial_elapsed_time = time.time() - trial_start_time
        self.trial_time_seconds = trial_elapsed_time
        print(f"\n%%%%% Total trial time: {trial_elapsed_time:.2f} seconds %%%%%")
        
        return self.Sigma_hat
    
    def get_current_estimate(self):
        """
        Override parent method to convert dictionary noise_scales to numpy array
        and support multiple solvers
        """
        # Convert dictionary noise_scales to numpy array
        noise_scales_array = np.zeros((self.d, self.d))
        for (i, j), scale in self.noise_scales.items():
            noise_scales_array[i, j] = scale
            if i != j:
                noise_scales_array[j, i] = scale  # Ensure symmetry
        
        if self.optimizer == "pgd_maxent_solver":
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            rows, cols = inds[:, 0], inds[:, 1]
            vals = self.Sigma_tilde[rows, cols]
            tau  = noise_scales_array[rows, cols]

            Sigma_hat, convergence_hist = ggm_solve(
                rows, cols, vals, tau, self.d, solver='pgd_maxent',
                eta=0.1, pgd_iters=2000, pgd_eps=1e-8,
                lbfgs_iters=1000, eps=1e-8,
                track_kkt=False
            )

            if not hasattr(self, 'convergence_history'):
                self.convergence_history = []
            self.convergence_history.append({'solver': 'pgd_maxent_solver', 'hist': convergence_hist})

            if np.any(np.isnan(Sigma_hat)) or np.any(np.isinf(Sigma_hat)):
                print("WARNING: pgd_maxent returned NaN/inf - keeping previous estimate")
                return self.Sigma_hat

            return Sigma_hat

        elif self.optimizer == "pgd-maxent":
            # Old pgd_maxent from model.py (dense, no component decomposition)
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            Sigma_hat, convergence_hist = pgd_maxent(
                self.Sigma_tilde, inds, noise_scales=noise_scales_array,
                max_iters=5000, eta=0.1, tol=1e-8, verbose=False
            )

            if not hasattr(self, 'convergence_history'):
                self.convergence_history = []
            self.convergence_history.append({'solver': 'pgd_maxent_model', 'hist': convergence_hist})

            if np.any(np.isnan(Sigma_hat)) or np.any(np.isinf(Sigma_hat)):
                print("WARNING: pgd-maxent (model.py) returned NaN/inf - keeping previous estimate")
                return self.Sigma_hat

            return Sigma_hat

        elif self.optimizer == "ipm_chol":
            inds = get_observed_Sigma_entries(self.Sigma_tilde)
            rows, cols = inds[:, 0], inds[:, 1]
            vals = self.Sigma_tilde[rows, cols]
            tau  = noise_scales_array[rows, cols]

            Sigma_hat, convergence_hist = ggm_solve(
                rows, cols, vals, tau, self.d, solver='ipm_chol',
                mu_init=1.0, mu_factor=0.5, eps=1e-8,
                max_outer=40, max_inner=200
            )

            if not hasattr(self, 'convergence_history'):
                self.convergence_history = []
            self.convergence_history.append({'solver': 'ipm_chol_solver', 'hist': convergence_hist})

            if np.any(np.isnan(Sigma_hat)) or np.any(np.isinf(Sigma_hat)):
                print("WARNING: ipm_chol returned NaN/inf - keeping previous estimate")
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