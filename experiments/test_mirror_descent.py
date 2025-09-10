import numpy as np
from scipy.linalg import cholesky

from privateggm.model import *


def test_mirror_descent():
    # generate synthetic ground truth covariance matrix
    np.random.seed(42)

    n = 5
    A = np.random.randn(n, n)
    Sigma_true = A @ A.T + np.eye(n)  # ensure well-conditioned

    # Select observations (only observe diagonal and some lower triangle)
    noise = 0.1 * np.random.randn(n, n)
    Sigma_tilde = Sigma_true + noise
    Sigma_tilde = (Sigma_tilde + Sigma_tilde.T) / 2  # symmetrize
    inds = [(i, j) for i in range(n) for j in range(n) if (i >= j and i%2==0)]  # Lower triangle

    # nitialize parameters
    tau = np.ones((n, n))  # Uniform weights
    Sigma_init = np.eye(n)  # Start from identity matrix

    # Run mirror descent
    Sigma_opt, K_opt = mirror_descent(
        Sigma_init=Sigma_init,
        Sigma_tilde=Sigma_tilde,
        inds=inds,
        tau=tau,
        max_iters=1000,
        eta_init=0.1
    )

    # Check positive definiteness
    assert np.all(np.linalg.eigvalsh(Sigma_opt) > 0), "Sigma_opt not PD"
    assert np.all(np.linalg.eigvalsh(K_opt) > 0), "K_opt not PD"

    # check inverse relationship
    assert np.allclose(Sigma_opt @ K_opt, np.eye(n)), "Sigma and K not inverses"

    # Check loss decrease

    def compute_loss(Sigma):
        return sum((Sigma[i, j] - Sigma_tilde[i, j]) ** 2 / tau[i, j] ** 2 for (i, j) in inds)

    initial_loss = compute_loss(Sigma_init)
    final_loss = compute_loss(Sigma_opt)
    assert final_loss < initial_loss, f"Loss increased: {final_loss:.2e} vs {initial_loss:.2e}"

    # Check symmetry
    assert np.allclose(Sigma_opt, Sigma_opt.T), "Sigma_opt not symmetric"
    assert np.allclose(K_opt, K_opt.T), "K_opt not symmetric"

    print(f"Test passed! Loss decreased from {initial_loss:.2e} to {final_loss:.2e}")


# Run the test
test_mirror_descent()