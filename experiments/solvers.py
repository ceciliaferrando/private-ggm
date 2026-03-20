"""solvers.py — uniform-interface optimization algorithms for covariance estimation.

Uniform signature::

    solver(A, inds, tau, **kwargs) -> (Sigma_hat, hist)

    A         : (d, d) observation matrix (zero at unobserved entries)
    inds      : (m, 2) observed index pairs, lower-triangle including diagonal
    tau       : (m,)   noise std-dev for each entry in inds

    Sigma_hat : (d, d) estimated covariance matrix
    hist      : list of per-iteration dicts with at least:
                  time    — cumulative wall-clock seconds since solver start
                  obj     — weighted objective f(X_E)  (bare; no barrier)
                  entropy — log|X| (Gaussian entropy up to a constant)

Solvers
-------
pgd              Stage A PGD (PSD projection baseline; no max-entropy structure)
pgd_maxent       Two-stage: Stage A PGD then Stage B max-entropy completion
ipm_chol         IPM with Cholesky parameterization X = LL^T
sparse_prec_ipm  IPM in K-space (K = X^{-1} sparse), JAX/optax L-BFGS
clarabel_solver  Off-the-shelf conic IPM via Clarabel (requires clarabel>=0.11.1)
"""

import time

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def project_psd(X, thresh=1e-8, return_evals=False):
    """Project X to the PSD cone by clipping eigenvalues to thresh."""
    vals, vecs = np.linalg.eigh(X)
    vals_c = np.maximum(vals, thresh)
    X_proj = vecs @ np.diag(vals_c) @ vecs.T
    return (X_proj, vals_c) if return_evals else X_proj


def _weights(tau):
    """Normalized weights w̃ = (1/τ²) / max(1/τ²)."""
    w = 1.0 / tau ** 2
    return w / w.max()


def _weighted_obj(X, A, inds, w):
    """Weighted squared-error objective f(X_E)."""
    r = X[inds[:, 0], inds[:, 1]] - A[inds[:, 0], inds[:, 1]]
    return float(np.sum(w * r ** 2))


# ── History utilities ──────────────────────────────────────────────────────────

def strip_compile_iter(hist):
    """Drop the first history entry (JIT compile overhead) and re-zero times.

    For JAX-based solvers the first iteration includes JIT compilation overhead.
    Under the approximation that first_iter ≈ compile_time, dropping it and
    re-zeroing the clock gives a cleaner picture of per-iteration cost.

    Returns
    -------
    stripped_hist : list of dicts with times re-zeroed from after the compile iter
    compile_time  : wall-clock seconds consumed by the first (compile) iteration
    """
    if len(hist) < 2:
        return hist, 0.0
    compile_time = hist[0]['time']
    t_shift = hist[1]['time']
    return [{**e, 'time': e['time'] - t_shift} for e in hist[1:]], compile_time


# ── pgd ───────────────────────────────────────────────────────────────────────

def pgd(A, inds, tau, eta=0.1, max_iters=2000, eps=1e-8, track_kkt=False):
    """
    Stage A: Projected Gradient Descent onto the PSD cone.

    Minimizes f(X_E) subject to X ⪰ 0.  Does NOT enforce max-entropy structure;
    use pgd_maxent for that.

    Parameters
    ----------
    eta       : initial step size
    max_iters : maximum PGD iterations
    eps       : convergence tolerance on step size ‖X_new − X‖_F
    track_kkt : if True, record min_eig_X, min_eig_G, comp_slack each iter
                (adds one eigendecomposition per step — disable for speed)

    Hist fields
    -----------
    Uniform : time, obj, entropy
    PGD     : delta (‖X_new − X‖_F), eta (step size)
    KKT     : min_eig_X, min_eig_G, comp_slack  (only if track_kkt=True)
    """
    d = A.shape[0]
    w = _weights(tau)

    W = np.zeros((d, d))
    W[inds[:, 0], inds[:, 1]] = w
    W[inds[:, 1], inds[:, 0]] = w
    mask = W > 0

    armijo = 0.5
    beta   = 0.5
    gamma  = 1.05

    X    = project_psd(A.copy())
    hist = []
    t0   = time.perf_counter()

    for _ in range(max_iters):
        G   = np.where(mask, W * (X - A), 0.0)
        G[np.arange(d), np.arange(d)] *= 2
        obj = _weighted_obj(X, A, inds, w)

        # Armijo backtracking line search for PGD
        for _ in range(50):
            X_new, evals_new = project_psd(X - eta * G, return_evals=True)
            predicted_decrease = -np.sum(G * (X_new - X))
            if _weighted_obj(X_new, A, inds, w) <= obj - armijo * predicted_decrease:
                break
            eta *= beta
            if eta < 1e-14:
                break

        delta = float(np.linalg.norm(X_new - X, 'fro'))
        entry = {
            'time':    time.perf_counter() - t0,
            'obj':     obj,
            'entropy': float(np.sum(np.log(evals_new))),
            'delta':   delta,
            'eta':     eta,
        }
        if track_kkt:
            entry['min_eig_X']  = float(np.linalg.eigvalsh(X).min())
            entry['min_eig_G']  = float(np.linalg.eigvalsh(G).min())
            entry['comp_slack'] = float(abs(np.sum(G * X)))

        hist.append(entry)
        X    = X_new
        eta *= gamma

        if delta < eps:
            break

    return X, hist


# ── pgd_maxent ────────────────────────────────────────────────────────────────

def pgd_maxent(A, inds, tau,
               eta=0.1, pgd_iters=2000, pgd_eps=1e-8,
               lbfgs_iters=500, eps=1e-8,
               track_kkt=False):
    """
    Two-stage pipeline: Stage A PGD then Stage B max-entropy completion.

    Stage A finds X̃ ⪰ 0 fitting the observations; Stage B solves the
    dual of max-entropy completion given X̃_S as constraints.

    Parameters
    ----------
    eta         : Stage A initial step size
    pgd_iters   : Stage A max PGD iterations
    pgd_eps     : Stage A convergence tolerance on step size
    lbfgs_iters : Stage B max L-BFGS iterations
    eps         : Stage B L-BFGS gradient tolerance
    track_kkt   : if True, add KKT diagnostics to Stage A hist entries

    Returns
    -------
    Sigma_hat : (d, d) estimated covariance (max-entropy completion of X̃_S)
    hist      : combined per-iteration list; Stage A then Stage B entries

    Hist fields
    -----------
    Uniform : time, obj, entropy
    Stage A : delta, eta  (+ min_eig_X, min_eig_G, comp_slack if track_kkt)
    Stage B : dual_loss, constraint_viol, grad_norm  (dual-domain diagnostics)
    """
    d    = A.shape[0]
    w    = _weights(tau)
    rows = inds[:, 0]
    cols = inds[:, 1]

    t0    = time.perf_counter()
    X_psd, hist = pgd(A, inds, tau, eta=eta, max_iters=pgd_iters, eps=pgd_eps,
                      track_kkt=track_kkt)
    for entry in hist:
        entry['stage'] = 'A'

    # ── Stage B: max-entropy completion ───────────────────────────────────────
    diag_mask = rows == cols
    mult      = np.where(diag_mask, 1.0, 2.0)

    def _build_K(z):
        K = np.zeros((d, d))
        K[rows, cols] = z
        K[cols, rows] = z
        return K

    def loss_and_grad(z):
        K = _build_K(z)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            shift = -np.linalg.eigvalsh(K).min() + 1e-4
            K_reg = K + shift * np.eye(d)
            Sigma_reg = np.linalg.inv(K_reg)
            grad = mult * (X_psd[rows, cols] - Sigma_reg[rows, cols])
            return 1e15, grad
        logdet_K = 2.0 * np.sum(np.log(np.diag(L)))
        Sigma = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(d)))
        loss  = -logdet_K + np.sum(K * X_psd)
        grad  = mult * (X_psd[rows, cols] - Sigma[rows, cols])
        return float(loss), grad

    def callback(z):
        K = _build_K(z)
        sign, logdet_K = np.linalg.slogdet(K)
        if sign <= 0:
            return
        Sigma = np.linalg.inv(K)
        grad  = mult * (X_psd[rows, cols] - Sigma[rows, cols])
        hist.append({
            'time':             time.perf_counter() - t0,
            'obj':              _weighted_obj(Sigma, A, inds, w),
            'entropy':          float(-logdet_K),
            'dual_loss':        float(-logdet_K + np.sum(_build_K(z) * X_psd)),
            'constraint_viol':  float(np.sqrt(np.mean((Sigma[rows, cols] - X_psd[rows, cols]) ** 2))),
            'grad_norm':        float(np.linalg.norm(grad)),
            'stage':            'B',
        })

    z0 = np.where(diag_mask, 1.0 / np.maximum(X_psd[rows, cols], 1e-6), 0.0)
    result = minimize(
        loss_and_grad, z0,
        method='L-BFGS-B', jac=True,
        callback=callback,
        options={'maxiter': lbfgs_iters, 'ftol': 0, 'gtol': eps},
    )

    K_opt     = _build_K(result.x)
    Sigma_hat = np.linalg.inv(K_opt)
    return Sigma_hat, hist


# ── ipm_chol ──────────────────────────────────────────────────────────────────

def ipm_chol(A, inds, tau,
             mu_init=1.0, mu_factor=0.5, eps=1e-8, eps_coarse=1e-2,
             max_outer=40, max_inner=200):
    """
    IPM with Cholesky parameterization X = LL^T  (L lower triangular, L[i,i] > 0).

    Minimizes  φ_μ(X) = f(X_E) − μ log|X|  over L, decreasing μ → 0.  The
    central path converges to the max-entropy solution because −log|X| is the
    negative Gaussian entropy.

    Key identity: log|LL^T| = 2 Σᵢ log L[i,i], so the barrier gradient is
    diagonal-only — no matrix inverse required, O(|S|·d) gradient.
    X ≻ 0 is guaranteed by the bound L[i,i] ≥ ε.

    Parameters
    ----------
    mu_init    : initial barrier weight
    mu_factor  : geometric decrease factor for μ (< 1)
    eps        : termination: stop outer loop when μ ≤ eps
    eps_coarse : coarse inner L-BFGS tolerance (early outer levels)
    max_outer  : max outer (μ-reduction) iterations
    max_inner  : max inner L-BFGS steps per outer iteration

    Hist fields  (one entry per outer iteration)
    -----------
    Uniform  : time, obj, entropy
    IPM      : mu, inner_iters
    KKT      : min_eig_G, comp_slack
    """
    d = A.shape[0]
    rows_obs, cols_obs = inds[:, 0], inds[:, 1]
    diag_d = np.arange(d)

    w = _weights(tau)
    W = np.zeros((d, d))
    W[rows_obs, cols_obs] = w
    W[cols_obs, rows_obs] = w
    mask = W > 0

    tril_r, tril_c = np.tril_indices(d)
    is_diag = (tril_r == tril_c)
    bounds  = [(1e-8, None) if is_diag[k] else (None, None) for k in range(len(tril_r))]

    def vec_to_L(z):
        L = np.zeros((d, d))
        L[tril_r, tril_c] = z
        return L

    # Initialize: X₀ = α I  →  L₀ = √α I
    alpha = max(float(np.mean(np.abs(np.diag(A)))), 1.0)
    z  = np.where(is_diag, np.sqrt(alpha), 0.0)
    mu = float(mu_init)

    hist = []
    t0   = time.perf_counter()

    for _outer in range(max_outer):
        eps_inner = min(mu, eps_coarse)

        def phi_and_grad(z, _mu=mu):
            L       = vec_to_L(z)
            diag_L  = L[diag_d, diag_d]
            logdet  = 2.0 * np.sum(np.log(diag_L))
            X       = L @ L.T
            r       = X[rows_obs, cols_obs] - A[rows_obs, cols_obs]
            fval    = float(np.sum(w * r ** 2))
            G       = np.where(mask, W * (X - A), 0.0)
            G[diag_d, diag_d] *= 2
            GL      = G @ L
            grad_z  = 2.0 * GL[tril_r, tril_c]
            grad_z[is_diag] -= 2.0 * _mu / diag_L
            return float(fval - _mu * logdet), grad_z

        result = minimize(phi_and_grad, z, method='L-BFGS-B', jac=True,
                          bounds=bounds,
                          options={'maxiter': max_inner, 'ftol': 0, 'gtol': eps_inner})
        z     = result.x
        L_opt = vec_to_L(z)
        X     = L_opt @ L_opt.T

        r      = X[rows_obs, cols_obs] - A[rows_obs, cols_obs]
        fval   = float(np.sum(w * r ** 2))
        G      = np.where(mask, W * (X - A), 0.0)
        G[diag_d, diag_d] *= 2
        logdet = 2.0 * float(np.sum(np.log(z[is_diag])))

        hist.append({
            'time':        time.perf_counter() - t0,
            'obj':         fval,
            'entropy':     logdet,
            'mu':          mu,
            'inner_iters': result.nit,
            'min_eig_G':   float(np.linalg.eigvalsh(G).min()),
            'comp_slack':  float(abs(np.sum(G * X))),
        })

        if mu <= eps:
            break
        mu *= mu_factor

    return project_psd(X), hist


# ── sparse_prec_ipm ───────────────────────────────────────────────────────────

def sparse_prec_ipm(A, inds, tau,
                    mu_init=1.0, mu_factor=0.3, eps=1e-6, eps_coarse=1e-2,
                    max_outer=50, max_inner=200, K_barrier=False):
    """
    IPM in K-space: minimize  f(X_E) + μ log|K|  over z = (K_ij)_{(i,j)∈E},
    where K = X^{-1} is the sparse precision matrix.

    The term +μ log|K| = −μ log|X| is the log-barrier for X ≻ 0.  The
    max-entropy structure (X = K^{-1} is the max-entropy completion of X_E) is
    built into the parameterization.

    K_barrier=True uses −μ log|K| instead (tested empirically — causes
    divergence; K collapses toward singular and entropy → −∞).

    Non-PD steps are rejected via Cholesky failure returning 1e30.

    Parameters
    ----------
    mu_init    : initial barrier weight
    mu_factor  : geometric decrease factor for μ (< 1)
    eps        : termination: stop outer loop when μ ≤ eps
    eps_coarse : coarse inner L-BFGS tolerance (early outer levels)
    max_outer  : max outer (μ-reduction) iterations
    max_inner  : max inner L-BFGS steps per outer iteration
    K_barrier  : if True, use −μ log|K|; if False (default), +μ log|K|

    Hist fields  (one entry per outer iteration)
    -----------
    Uniform  : time, obj, entropy
    IPM      : mu, inner_iters
    Debug    : grad_norm (last inner gradient norm)

    The first outer iteration includes JAX JIT compilation overhead.
    """
    d = A.shape[0]
    rows, cols = inds[:, 0], inds[:, 1]
    diag_mask  = rows == cols

    w      = _weights(tau)
    rows_j = jnp.array(rows)
    cols_j = jnp.array(cols)
    w_j    = jnp.array(w)
    A_j    = jnp.array(A)

    def build_K_jax(z):
        K = jnp.zeros((d, d))
        K = K.at[rows_j, cols_j].set(z)
        K = K.at[cols_j, rows_j].set(z)
        return K

    barrier_sign = -1.0 if K_barrier else +1.0  # K-barrier: -μlog|K|; X-barrier: +μlog|K|

    def loss_fn(z, mu):
        K      = build_K_jax(z)
        L      = jnp.linalg.cholesky(K)
        chol_ok = jnp.all(jnp.isfinite(jnp.diag(L)))

        def valid(_):
            logdet_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            X = jax.scipy.linalg.cho_solve((L, True), jnp.eye(d))
            r = X[rows_j, cols_j] - A_j[rows_j, cols_j]
            return jnp.sum(w_j * r ** 2) + barrier_sign * mu * logdet_K

        def invalid(_):
            return jnp.array(1e30)

        return jax.lax.cond(chol_ok, valid, invalid, None)

    opt = optax.chain(
        optax.lbfgs(),
        optax.scale_by_zoom_linesearch(max_linesearch_steps=30),
    )

    @jax.jit
    def step(z, state, mu):
        loss = lambda z: loss_fn(z, mu)
        v, g = jax.value_and_grad(loss)(z)
        updates, new_state = opt.update(g, state, z, value=v, grad=g, value_fn=loss)
        z_new = optax.apply_updates(z, updates)
        return z_new, new_state, v, g

    z0 = jnp.array(np.where(diag_mask, 1.0 / np.maximum(A[rows, cols], 1e-6), 0.0))
    z  = z0
    mu = float(mu_init)

    hist  = []
    t0    = time.perf_counter()

    for _outer in range(max_outer):
        state      = opt.init(z)
        mu_j       = jnp.array(mu)
        inner_tol  = min(mu, eps_coarse)
        inner_iters = 0
        g = None

        for _ in range(max_inner):
            z, state, v, g = step(z, state, mu_j)
            inner_iters += 1
            if float(jnp.linalg.norm(g)) < inner_tol:
                break

        # Outer-step diagnostics: compute primal obj and entropy from z
        z_np = np.array(z)
        K_np = np.zeros((d, d))
        K_np[rows, cols] = z_np
        K_np[cols, rows] = z_np
        try:
            L_np    = np.linalg.cholesky(K_np)
            logdetK = 2.0 * float(np.sum(np.log(np.diag(L_np))))
            X_np    = np.linalg.solve(L_np.T, np.linalg.solve(L_np, np.eye(d)))
            obj     = _weighted_obj(X_np, A, inds, w)
            entropy = -logdetK   # log|X| = log|K^{-1}| = -log|K|
        except np.linalg.LinAlgError:
            obj     = float('nan')
            entropy = float('nan')

        hist.append({
            'time':        time.perf_counter() - t0,
            'obj':         obj,
            'entropy':     entropy,
            'mu':          mu,
            'inner_iters': inner_iters,
            'grad_norm':   float(jnp.linalg.norm(g)) if g is not None else float('nan'),
        })

        if mu <= eps:
            break
        mu *= mu_factor

    z_np  = np.array(z)
    K_opt = np.zeros((d, d))
    K_opt[rows, cols] = z_np
    K_opt[cols, rows] = z_np
    try:
        Sigma_hat = np.linalg.inv(K_opt)
    except np.linalg.LinAlgError:
        Sigma_hat = project_psd(np.linalg.pinv(K_opt))
    return project_psd(Sigma_hat), hist


# ── sparse_prec_nobarrier ─────────────────────────────────────────────────────

def sparse_prec_nobarrier(A, inds, tau,
                          max_inner=2000, record_every=10):
    """
    K-space L-BFGS with no log-barrier — ablation of the barrier term in sparse_prec_ipm.

    Minimizes f(K^{-1}_E) directly over z = (K_ij)_{(i,j)∈E} in a single L-BFGS run.
    Non-PD steps are rejected (return 1e30 on Cholesky failure) but there is no
    mu*log|K| barrier term and no outer annealing loop.

    Records a hist entry every `record_every` inner steps so the convergence curve
    is visible. The first entry absorbs JIT compile time (stripped by strip_compile_iter).

    Parameters
    ----------
    max_inner    : total L-BFGS steps
    record_every : record a hist entry every this many inner steps
    """
    d = A.shape[0]
    rows, cols = inds[:, 0], inds[:, 1]
    diag_mask  = rows == cols

    w      = _weights(tau)
    rows_j = jnp.array(rows)
    cols_j = jnp.array(cols)
    w_j    = jnp.array(w)
    A_j    = jnp.array(A)

    def build_K_jax(z):
        K = jnp.zeros((d, d))
        K = K.at[rows_j, cols_j].set(z)
        K = K.at[cols_j, rows_j].set(z)
        return K

    def loss_fn(z):
        K       = build_K_jax(z)
        L       = jnp.linalg.cholesky(K)
        chol_ok = jnp.all(jnp.isfinite(jnp.diag(L)))

        def valid(_):
            X = jax.scipy.linalg.cho_solve((L, True), jnp.eye(d))
            r = X[rows_j, cols_j] - A_j[rows_j, cols_j]
            return jnp.sum(w_j * r ** 2)

        def invalid(_):
            return jnp.array(1e30)

        return jax.lax.cond(chol_ok, valid, invalid, None)

    opt = optax.chain(
        optax.lbfgs(),
        optax.scale_by_zoom_linesearch(max_linesearch_steps=30),
    )

    @jax.jit
    def step(z, state):
        v, g = jax.value_and_grad(loss_fn)(z)
        updates, new_state = opt.update(g, state, z, value=v, grad=g, value_fn=loss_fn)
        z_new = optax.apply_updates(z, updates)
        return z_new, new_state, v, g

    z0    = jnp.array(np.where(diag_mask, 1.0 / np.maximum(A[rows, cols], 1e-6), 0.0))
    z     = z0
    state = opt.init(z)
    hist  = []
    t0    = time.perf_counter()

    for inner_iter in range(max_inner):
        z, state, v, g = step(z, state)

        if (inner_iter + 1) % record_every == 0:
            z_np = np.array(z)
            K_np = np.zeros((d, d))
            K_np[rows, cols] = z_np
            K_np[cols, rows] = z_np
            try:
                L_np    = np.linalg.cholesky(K_np)
                logdetK = 2.0 * float(np.sum(np.log(np.diag(L_np))))
                X_np    = np.linalg.solve(L_np.T, np.linalg.solve(L_np, np.eye(d)))
                obj     = _weighted_obj(X_np, A, inds, w)
                entropy = -logdetK
            except np.linalg.LinAlgError:
                obj     = float('nan')
                entropy = float('nan')

            hist.append({
                'time':      time.perf_counter() - t0,
                'obj':       obj,
                'entropy':   entropy,
                'inner_iter': inner_iter + 1,
                'grad_norm': float(jnp.linalg.norm(g)),
            })

        if float(jnp.linalg.norm(g)) < 1e-6:
            break

    if not hist:
        hist.append({'time': time.perf_counter() - t0, 'obj': float('nan'), 'entropy': float('nan'),
                     'inner_iter': 0, 'grad_norm': float('nan')})

    z_np  = np.array(z)
    K_opt = np.zeros((d, d))
    K_opt[rows, cols] = z_np
    K_opt[cols, rows] = z_np
    try:
        Sigma_hat = np.linalg.inv(K_opt)
    except np.linalg.LinAlgError:
        Sigma_hat = project_psd(np.linalg.pinv(K_opt))
    return project_psd(Sigma_hat), hist


# ── clarabel_solver ───────────────────────────────────────────────────────────

def clarabel_solver(A, inds, tau,
                    verbose=False, max_iter=200,
                    tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8):
    """
    Off-the-shelf conic IPM via Clarabel (requires clarabel >= 0.11.1).

    Formulates the problem as a QP with PSD cone constraint:
        min  (1/2) x^T P x + q^T x   s.t.  -D x + s = 0,  s in PSDTriangleCone(d)
    where x is the upper triangle of X in column-major order and D is diagonal
    with entries 1 (diagonal of X) or sqrt(2) (off-diagonal of X).

    Clarabel's cost_primal = f(X_E) - C where C = sum(w * a^2) is the constant
    dropped when expanding the squared-error objective. The callback adds C back
    to recover our objective at each iteration.

    Parameters
    ----------
    verbose     : pass True to enable Clarabel's own iteration output
    max_iter    : maximum IPM iterations
    tol_gap_abs : absolute duality gap tolerance
    tol_gap_rel : relative duality gap tolerance
    tol_feas    : primal/dual feasibility tolerance

    Hist fields  (one entry per IPM iteration)
    -----------
    Uniform  : time, obj, entropy  (entropy = nan except on the last entry)
    Clarabel : mu, res_primal, res_dual, status
    """
    import clarabel
    from scipy import sparse

    d    = A.shape[0]
    nvec = d * (d + 1) // 2

    # Upper-triangle column-major ordering: column c, rows 0..c
    r_ut = np.array([r for c in range(d) for r in range(c + 1)])
    c_ut = np.array([c for c in range(d) for r in range(c + 1)])
    is_diag_ut = (r_ut == c_ut)

    # Map inds (which may be upper- or lower-triangle) to upper-triangle
    # column-major index.  For upper-triangle entry (r,c) with r<=c:
    #   k = c*(c+1)//2 + r  =  big*(big+1)//2 + small
    i_obs, j_obs = inds[:, 0], inds[:, 1]
    big   = np.maximum(i_obs, j_obs)
    small = np.minimum(i_obs, j_obs)
    k_obs = big * (big + 1) // 2 + small

    w     = _weights(tau)
    a_obs = A[i_obs, j_obs]

    # Constant dropped from Clarabel objective: C = sum(w * a^2)
    # Recover our objective via:  obj = cost_primal + C_const
    C_const = float(np.sum(w * a_obs ** 2))

    # P: diagonal sparse, P[k,k] = 2w for observed entries
    P_diag        = np.zeros(nvec)
    P_diag[k_obs] = 2.0 * w
    P = sparse.diags(P_diag, format='csc')

    # q: q[k] = -2*w*a for observed entries
    q        = np.zeros(nvec)
    q[k_obs] = -2.0 * w * a_obs

    # Constraint matrix: A_clar = -D  where D[k,k] = 1 (diag) or sqrt(2) (off-diag)
    # Encodes  -D x + s = 0  =>  s = D x = vec(X)  in PSDTriangleCone
    scale  = np.where(is_diag_ut, 1.0, np.sqrt(2.0))
    A_clar = sparse.diags(-scale, format='csc')
    b      = np.zeros(nvec)
    cones  = [clarabel.PSDTriangleConeT(d)]

    settings             = clarabel.DefaultSettings()
    settings.verbose     = verbose
    settings.max_iter    = max_iter
    settings.tol_gap_abs = tol_gap_abs
    settings.tol_gap_rel = tol_gap_rel
    settings.tol_feas    = tol_feas

    hist = []
    t0   = time.perf_counter()

    def _callback(info):
        hist.append({
            'time':       time.perf_counter() - t0,
            'obj':        info.cost_primal + C_const,
            'entropy':    float('nan'),
            'mu':         info.mu,
            'res_primal': info.res_primal,
            'res_dual':   info.res_dual,
            'status':     str(info.status),
        })
        return False  # never request early termination

    solver = clarabel.DefaultSolver(P, q, A_clar, b, cones, settings)
    solver.set_termination_callback(_callback)
    sol = solver.solve()

    # Reconstruct Sigma_hat from solution vector x (upper triangle, column-major)
    x        = np.array(sol.x)
    Sigma_hat = np.zeros((d, d))
    Sigma_hat[r_ut, c_ut] = x
    Sigma_hat[c_ut, r_ut] = x  # symmetrize

    # Patch final entry with entropy and solved status; earlier entries stay nan
    _, logdet = np.linalg.slogdet(Sigma_hat)
    if hist:
        hist[-1]['entropy'] = float(logdet)
        hist[-1]['status']  = str(sol.status)
    else:
        hist.append({
            'time':    time.perf_counter() - t0,
            'obj':     _weighted_obj(Sigma_hat, A, inds, w),
            'entropy': float(logdet),
            'status':  str(sol.status),
        })

    return Sigma_hat, hist


# ── pgd_maxent_jax ────────────────────────────────────────────────────────────

def pgd_maxent_jax(A, inds, tau,
                   max_iters=1000, eta=0.1, tol=1e-8, verbose=False):
    """
    Wrapper around privateggm.model.pgd_maxent with the uniform benchmark interface.

    Calls the student's two-stage implementation (Stage A: projected gradient descent
    with BB line search; Stage B: max-entropy L-BFGS in K-space via JAX/optax).

    Parameters
    ----------
    max_iters : Stage A max PGD iterations
    eta       : Stage A initial step size
    tol       : Stage A convergence tolerance
    verbose   : print per-stage progress

    Hist fields
    -----------
    Uniform : time, obj, entropy
    Stage   : stage ('A' or 'B')
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from privateggm.model import pgd_maxent as _pgd_maxent

    d = A.shape[0]
    rows, cols = inds[:, 0], inds[:, 1]

    # Convert tau vector → full (d, d) noise_scales matrix
    noise_scales = np.zeros((d, d))
    noise_scales[rows, cols] = tau
    noise_scales[cols, rows] = tau

    Sigma_hat, conv = _pgd_maxent(
        A, inds,
        noise_scales=noise_scales,
        max_iters=max_iters,
        eta=eta,
        tol=tol,
        verbose=verbose,
    )
    return Sigma_hat, conv['hist']
