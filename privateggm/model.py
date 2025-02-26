import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from functools import partial

from privateggm.utils import *

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
    # linesearch
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)

    # set up optimizer
    opt = optax.chain(
        optax.lbfgs(),
        linesearch,
    )

    # init state
    state = opt.init(z)

    # choose loss
    # opt_loss = loss_fn_psd_constrained
    opt_loss = loss_fn

    # loss function wrapper
    loss_fn_wrapper = partial(opt_loss, inds=inds, Sigma_tilde_vec=Sigma_tilde_vec, d=d)

    # Create the value and gradient function using the partially applied function (compiled once)
    v_and_g = jax.value_and_grad(loss_fn_wrapper)

    # PD check function
    def is_positive_definite(matrix):
        eigenvalues = jnp.linalg.eigvalsh(matrix)
        return jnp.all(eigenvalues > 0)

    # jax.jitted optimization step
    @jax.jit
    def optimization_step(z, state, step_size):
        v, g = v_and_g(z)
        updates, state = opt.update(g, state, z, value=v, grad=g, value_fn=loss_fn_wrapper)
        z = z + step_size * updates  # Manual step size adjustment
        grad_sum = jnp.sum(jnp.abs(g))
        return z, state, v, grad_sum

    # Track last stable step
    last_stable_z = z
    last_stable_K = get_K(z, inds, d)

    # Optimization loop with PD check
    g_vals = []
    obj_vals = []

    for i in range(max_iters):
        step_size = 1.0
        max_retries = 5  # Prevent infinite loop
        retries = 0

        while retries < max_retries:

            z_new, state, v, grad_sum = optimization_step(z, state, step_size)
            curr_K = get_K(z_new, inds, d)

            # Check matrix stability
            eigvals = np.linalg.eigvalsh(curr_K)
            min_eigval = np.min(eigvals)
            max_eigval = np.max(eigvals)
            cond_number = max_eigval / min_eigval if min_eigval > 1e-10 else np.inf

            is_non_pd = not jnp.all(eigvals > 0)
            is_ill_conditioned = cond_number > 1e10
            is_nearly_singular = min_eigval < 1e-8

            # If no pathology, move forward
            if not is_non_pd and not is_ill_conditioned and not is_nearly_singular:
                last_stable_z = z_new  # Store stable step
                last_stable_K = curr_K
                break  # exit retry loop, proceed to next iteration

            # Print pathology details
            debug_print(f"⚠️ Iteration {i}, Retry {retries}: Pathology Detected!")
            debug_print(
                f"   Min Eigval: {min_eigval:.4e}, Max Eigval: {max_eigval:.4e}, Condition Number: {cond_number:.2e}")
            if is_non_pd:
                debug_print(f"   ❌ K_z is NOT PD!")
            if is_ill_conditioned:
                debug_print(f"   ⚠️ K_z is Ill-Conditioned!")
            if is_nearly_singular:
                debug_print(f"   ❗ K_z is Nearly Singular!")

            # **Rollback to last stable step**
            debug_print(f"🔄 Rolling back to previous stable iteration {i - 1} and retrying...")
            z = last_stable_z
            curr_K = last_stable_K + 1e-3 * jnp.eye(d)  # Apply stronger regularization

            # **Reduce step size**
            step_size *= 0.5  # reduce step size
            retries += 1  # increase retry count

        # **if retries exceed max limit, apply stronger fallback**
        if retries == max_retries:
            debug_print(f"🚨 Iteration {i}: Max retries reached. Applying emergency correction.")
            curr_K = project_to_PSD(curr_K, thresh=1e-3)  # stronger projection
            curr_K += 1e-3 * jnp.eye(d)  # and stronger regularization

        # Move forward to next iteration
        z = z_new
        obj_vals.append(v.item())
        g_vals.append(grad_sum.item())

        # Debug logging
        if verbose:
            print(f"Iteration {i}: Loss: {v:.7e}, Grad Sum: {grad_sum:.7e}, Condition Number: {cond_number:.2e}")

    return z, g_vals, obj_vals