import argparse
import os
import numpy as np
from privateggm.driver import run_driver
from privateggm.utils import *

# Argument parser for flexible experiment configuration
parser = argparse.ArgumentParser(description="Run covariance matrix estimation experiment.")

# Ensure data path is correctly resolved


parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset")
parser.add_argument("--n_iters", type=int, default=500, help="Number of optimization iterations")
parser.add_argument("--rho", type=float, default=0.5, help="Differential privacy parameter")
parser.add_argument("--structure", type=str, default="rndpercent", choices=["chain", "random", "rndpercent"], help="Graph structure")
parser.add_argument("--apply_dp", action="store_true", help="Apply differential privacy")
parser.add_argument("--num_trials", type=int, default=10, help="Number of trials")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

args = parser.parse_args()

data_path = os.path.join(os.path.dirname(__file__), "..", "data", args.dataset_name + ".csv")

# Set seed
np.random.seed(42)

# Results for each trial are going to be stored here
MaxEnt_results = []
SSP_results = []

# Run experiment with chosen dataset
for t in range(args.num_trials):
    results = run_driver(
        data_path=data_path,
        structure=args.structure,
        structure_dict=None,
        max_iters=args.n_iters,
        iterative=False,
        max_off_diag_percentage=0.5,
        apply_dp=args.apply_dp,
        rho=args.rho,
        alpha=0.5
    )

    # Store results
    Sigma_true = results['Sigma_true']
    Sigma_tilde = results['Sigma_tilde']
    Sigma_SSP = results['Sigma_SSP']
    Sigma_MaxEnt = results['Sigma_opt']

    MaxEnt_metric = float(mahalanobis_cov_error(Sigma_true, Sigma_MaxEnt))
    SSP_metric = float(mahalanobis_cov_error(Sigma_true, Sigma_SSP))

    print(MaxEnt_metric, SSP_metric)

    MaxEnt_results.append(MaxEnt_metric)
    SSP_results.append(SSP_metric)

print(f"MaxEnt results: {MaxEnt_results}")
print(f"SSP results: {SSP_results}")

save_results(MaxEnt_results, SSP_results, args.dataset_name)