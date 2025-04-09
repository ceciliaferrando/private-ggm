import numpy as np
from scipy.linalg import solve, cholesky, eigh
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
import optax
from functools import partial

from privateggm.utils import *

# DEBUG_MODE = True

# DIRECT LBFGS OPTIMIZATION METHOD

def loss_fn(z, inds, Sigma_tilde_vec, d):
    """
    z [jnp.array]: vector supporting precision matrix estimate K(z)
    inds [np.array]: list of the lower triangular nonzero element indices in Sigma_tilde
    Sigma_tilde_vec [np.array]: vector of Sigma_tilde indexed at inds
    d [int]: static argument, number of attributes
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

def mirror_descent(Sigma_init, Sigma_tilde, inds, tau, max_iters=20000, eta_init=0.01, tol=1e-8, verbose=False):
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


def projected_gradient_descent(S_observed, mask, eta=0.1, max_iters=1000, tol=1e-6):
    d = S_observed.shape[0]

    # Ensure mask is 2D boolean array w/ matching shape
    assert mask.shape == S_observed.shape, "Mask must match matrix dimensions"

    # 1. set unobserved entries to 0
    S = np.zeros_like(S_observed)
    S[mask] = S_observed[mask]  # Correct 2D indexing

    # 2. PSD projection
    Sigma = project_to_PSD(S, thresh=1e-3)

    prev_Sigma = Sigma.copy()
    for t in range(max_iters):
        # 3. compute gradient ONLY on observed entries
        gradient = np.zeros_like(Sigma)
        gradient[mask] = Sigma[mask] - S[mask]  # 2D indexing

        # 4. update and project to psd
        Sigma_temp = Sigma - eta * gradient
        Sigma = project_to_PSD(Sigma_temp, thresh=1e-3)

        if np.linalg.norm(Sigma - prev_Sigma, 'fro') < tol:
            break
        prev_Sigma = Sigma.copy()

    return Sigma

def pgd_maxent(Sigma_tilde, inds, max_iters=1000, eta=0.01, tol=1e-6, verbose=True):
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
    mask = np.zeros(Sigma_tilde.shape, dtype=bool)  # Explicit 2D shape
    for (i, j) in inds:
        mask[i, j] = True
        mask[j, i] = True  # Ensure symmetry

    Sigma_obs = np.zeros_like(Sigma_tilde)
    Sigma_obs[mask] = Sigma_tilde[mask]  # Fill observed entries, others remain 0

    # Stage A: Projected Gradient Descent
    if verbose:
        print("\nRunning Projected Gradient Descent...")
    Sigma_psd = projected_gradient_descent(Sigma_obs, mask, eta=eta, max_iters=max_iters, tol=tol)

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
        Sigma_opt = np.linalg.inv(K_z_opt)

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return

    return Sigma_opt
