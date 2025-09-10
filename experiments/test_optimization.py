import numpy as np
import argparse
from scipy.linalg import cholesk

from privateggm.model import optimize_lbfgs, mirror_descent


def generate_toy_example():
    """
    Generate a small 3x3 synthetic covariance matrix and parameters for testing.
    """
    np.random.seed(42)

    # Small positive definite covariance matrix
    Sigma_true = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, 0.4],
                           [0.3, 0.4, 1.0]])

    Sigma_init = np.eye(3)  # Start with the identity matrix

    # Noisy observed Sigma
    Sigma_tilde = Sigma_true + 0.1 * np.random.randn(3, 3)
    Sigma_tilde = (Sigma_tilde + Sigma_tilde.T) / 2  # Ensure symmetry

    # Indices of observed entries
    inds = np.array([(i, j) for i in range(3) for j in range(i, 3)])

    # Noise levels (weights)
    tau = np.ones((3, 3)) * 0.1  # Assume uniform noise level

    return Sigma_init, Sigma_tilde, inds, tau


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test L-BFGS and Mirror Descent on a 3x3 toy example.")

    parser.add_argument("--method", type=str, choices=["lbfgs", "mirror"], default="lbfgs",
                        help="Optimization method to use (lbfgs or mirror)")
    parser.add_argument("--max_iters", type=int, default=500, help="Maximum number of iterations")
    parser.add_argument("--eta", type=float, default=0.1, help="Initial step size (only for mirror descent)")
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance for convergence")

    args = parser.parse_args()

    # Generate toy example
    Sigma_init, Sigma_tilde, inds, tau = generate_toy_example()

    print("=== Running Optimization ===")
    print(f"Method: {args.method}")
    print(f"Max Iterations: {args.max_iters}")
    print(f"Tolerance: {args.tol}")

    if args.method == "lbfgs":
        print("Running L-BFGS Optimization")
        Sigma_opt, g_vals, obj_vals = optimize_lbfgs(Sigma_tilde, Sigma_init.flatten(), inds, 3,
                                                     max_iters=args.max_iters, verbose=True)
        print("L-BFGS Optimization Complete")
        print("Final Optimized Covariance Matrix:", Sigma_opt)

    elif args.method == "mirror":
        print("Running Mirror Descent Optimization")
        Sigma_opt, K_opt = mirror_descent(Sigma_init, Sigma_tilde, inds, tau,
                                          max_iters=args.max_iters, eta_init=args.eta, tol=args.tol)
        print("Mirror Descent Optimization Complete")
        print("Final Optimized Covariance Matrix:\n", Sigma_opt)

    print("Testing Complete!\n")
