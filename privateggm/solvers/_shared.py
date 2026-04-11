"""Shared helpers used across all solvers."""

import numpy as np


def project_psd(X, min_eig=0.0, thresh=1e-8, return_evals=False):
    """Project X onto {X : X ⪽ min_eig·I} by clipping eigenvalues.

    Parameters
    ----------
    min_eig : float — minimum eigenvalue to enforce (default 0.0)
    thresh  : float — numerical floor applied on top of min_eig (default 1e-8)
    """
    vals, vecs = np.linalg.eigh(X)
    vals_c = np.maximum(vals, max(thresh, min_eig))
    X_proj = vecs @ np.diag(vals_c) @ vecs.T
    return (X_proj, vals_c) if return_evals else X_proj


def _weights(tau):
    """Normalized weights w̃ = (1/τ²) / max(1/τ²)."""
    w = 1.0 / tau ** 2
    return w / w.max()


def _weighted_obj(X, rows, cols, vals, w):
    """Weighted squared-error objective f(X_E)."""
    r = X[rows, cols] - vals
    return float(np.sum(w * r ** 2))


def dense_to_coo(A, inds, tau):
    """Convert dense observation format to sparse COO format.

    Parameters
    ----------
    A    : (d, d) ndarray — symmetric observation matrix
    inds : (m, 2) int array — observed index pairs
    tau  : (m,) float array — noise std-devs

    Returns
    -------
    rows, cols, vals, tau, d
    """
    rows = inds[:, 0]
    cols = inds[:, 1]
    vals = A[rows, cols]
    return rows, cols, vals, tau, A.shape[0]


def normalize_coo(rows, cols, vals, tau):
    """Normalize COO observations to upper-triangle (rows <= cols).

    Swaps (i,j) pairs where i > j, then combines duplicate entries
    (including symmetric pairs) via inverse-variance weighting:
        val_combined = sum(w_k * v_k) / sum(w_k)
        tau_combined = 1 / sqrt(sum(w_k))   where w_k = 1/tau_k^2

    Returns (rows, cols, vals, tau) with rows <= cols and one entry
    per unique (i,j) pair, sorted by (row, col).
    """
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    vals = np.asarray(vals, dtype=float)
    tau  = np.asarray(tau,  dtype=float)

    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)

    d_enc = int(hi.max()) + 1
    keys  = lo * d_enc + hi          # unique int per (lo, hi) pair

    w     = 1.0 / tau ** 2
    order = np.argsort(keys, kind='stable')
    keys_s = keys[order]
    w_s    = w[order]
    v_s    = vals[order]

    unique_keys, inv = np.unique(keys_s, return_inverse=True)
    w_tot  = np.zeros(len(unique_keys))
    wv_tot = np.zeros(len(unique_keys))
    np.add.at(w_tot,  inv, w_s)
    np.add.at(wv_tot, inv, w_s * v_s)

    rows_n = unique_keys // d_enc
    cols_n = unique_keys %  d_enc
    vals_n = wv_tot / w_tot
    tau_n  = 1.0 / np.sqrt(w_tot)
    return rows_n, cols_n, vals_n, tau_n
