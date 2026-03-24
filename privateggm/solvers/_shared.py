"""Shared helpers used across all solvers."""

import numpy as np


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
