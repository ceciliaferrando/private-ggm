import numpy as np
from scipy.linalg import solve, cholesky
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

    # linesearch
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)

    # set up optimizer
    opt = optax.chain(
        optax.lbfgs(),
        linesearch,
    )

    # init state
    state = opt.init(z)

    # loss function wrapper
    loss_fn_wrapper = partial(loss_fn, inds=inds, Sigma_tilde_vec=Sigma_tilde_vec, d=d)

    # Create the value and gradient function using the partially applied function (compiled once)
    v_and_g = jax.value_and_grad(loss_fn_wrapper)

    # jax.jitted optimization step
    @jax.jit
    def optimization_step(z, state, step_size):
        v, g = v_and_g(z)
        updates, state = opt.update(g, state, z, value=v, grad=g, value_fn=loss_fn_wrapper)
        z_new = z + step_size * updates  # Apply updates with step size
        grad_sum = jnp.sum(jnp.abs(g))
        return z_new, state, v, grad_sum

    # optimization loop
    g_vals = []
    obj_vals = []
    prev_v = None  # Track previous loss value
    tol = 1e-8  # Tolerance for stopping criterion
    step_size = 1.0  # Initial step size

    for i in range(max_iters):
        z_new, state, v, grad_sum = optimization_step(z, state, step_size)

        # Backtracking line search: Reduce step size if loss increases
        if prev_v is not None and v > prev_v:
            step_size *= 0.5  # Reduce step size
            if verbose:
                print(f"Iteration {i}: Loss increased. Reducing step size to {step_size:.7e}")
            continue

        # Update z and step size
        z = z_new
        #jax.debug.print("{z} ", z=z[:10])

        step_size = min(step_size * 1.1, 1.0)  # Increase step size slightly

        # Append values for tracking
        obj_vals.append(v.item())
        g_vals.append(grad_sum.item())

        # Print progress
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: Loss = {v:.6e}, Grad Sum = {grad_sum:.6e}, Step Size = {step_size:.6e}")

        # Stopping criterion: Check gradient norm
        if grad_sum < tol:
            if verbose:
                print(f"Optimization converged at iteration {i}: Grad Sum = {grad_sum:.7e}")
            break

        # Stopping criterion: Check loss change
        if prev_v is not None and abs(v - prev_v) < tol:
            if verbose:
                print(f"Optimization converged at iteration {i}: Loss Change = {abs(v - prev_v):.7e}")
            break

        # Update previous loss value
        prev_v = v

    #jax.debug.print("{z} ", z=z[:10])
    K_z = get_K(z, inds, d)

    eigvals = np.linalg.eigvalsh(K_z)
    min_eigval = np.min(eigvals)
    max_eigval = np.max(eigvals)
    cond_number = max_eigval / min_eigval if min_eigval > 1e-10 else np.inf

    is_non_pd = not jnp.all(eigvals > 0)
    is_ill_conditioned = cond_number > 1e10
    is_nearly_singular = min_eigval < 1e-8

    if is_non_pd or is_ill_conditioned or is_nearly_singular:
        print("PROBLEM DETECTED WITH K_z:")
        print(f"is_non_pd? {is_non_pd}")
        print(f"is_ill_conditioned? {is_ill_conditioned}")
        print(f"is_nearly_singular? {is_nearly_singular}")

    return z, g_vals, obj_vals


# MIRROR DESCENT METHOD

def mirror_descent(Sigma_init, Sigma_tilde, inds, tau, max_iters=20000, eta_init=0.1, tol=1e-8, verbose=False):
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
    n = Sigma.shape[0]

    def objective(Sigma):
        loss = 0.0
        for (i, j) in processed_inds:
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
                current_eta *= 0.5  # Reduce step size

        if not success:
            raise RuntimeError(f"Failed to find PD K matrix after {iter} iterations")

        # update
        K_prev = K.copy()
        K = K_candidate
        Sigma = np.linalg.inv(K)
        eta = current_eta

        # step size decay regardless of the conditions above
        eta *= 0.995

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
