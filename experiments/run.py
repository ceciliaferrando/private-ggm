import argparse
import os
import numpy as np
from privateggm.driver import run_driver
from privateggm.utils import *
from privateggm.data import *

# Argument parser for flexible experiment configuration
parser = argparse.ArgumentParser(description="Run covariance matrix estimation experiment.")

# Ensure data path is correctly resolved


parser.add_argument("--d", type=int, required=True, help="dimension of synthetic data")
parser.add_argument("--optimizer", type=str, required=True, help="maxent, mirror-descent, or pgd-maxent?")
parser.add_argument("--n_iters", type=int, default=500, help="Number of optimization iterations")
parser.add_argument("--rho", type=float, default=1.0, help="Differential privacy parameter")
parser.add_argument("--structure", type=str, default="rndpercent", choices=["chain", "random", "rndpercent"], help="Graph structure")
parser.add_argument("--apply_dp", action="store_true", help="Apply differential privacy")
parser.add_argument("--num_trials", type=int, default=10, help="Number of trials")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

args = parser.parse_args()

clip_type = "linf"
rescale = True
offdiagpct = 0.5

# set rng
rng = np.random.default_rng(seed=42)
# PASS TO FUN

# Results storage
results_dict = {
    "MaxEnt_mahalanobis": [],
    "SSP_mahalanobis": [],
    "MaxEnt_l2": [],
    "SSP_l2": [],
    "MaxEnt_mse": [],
    "SSP_mse": []
}

# Generate data
outdir = "data/"
generate_well_conditioned_gaussian(rng, outdir, 1000, args.d, rescale=rescale, cond_max=10.0)
data_path = os.path.join(os.path.dirname(__file__), "..", "data",
                         "synthdata" + str(args.d) + "rescale" + str(rescale) + ".csv")
bounds_path = os.path.join(os.path.dirname(__file__), "..", "data", "L_bounds_rescale" + str(rescale) + ".json")

# Load and process data
data, _, _, Sigma_true, T = load_and_process_data(data_path, bounds_path, clip_type=clip_type)

# Run experiment across multiple trials
for t in range(args.num_trials):

    print(f"TRIAL N. {t}")

    results = run_driver(
        rng = rng,
        data=data,
        Sigma_true=Sigma_true,
        T=T,
        structure=args.structure,
        structure_dict=None,
        max_iters=args.n_iters,
        iterative=False,
        max_off_diag_percentage=offdiagpct,
        apply_dp=True,
        rho=args.rho,
        clip_type=clip_type,
        rescale=rescale,
        alpha=None,
        optimizer=args.optimizer
    )

    Sigma_true = results['Sigma_true']
    Sigma_tilde = results['Sigma_true']
    Sigma_SSP = results['Sigma_SSP']
    Sigma_MaxEnt = results['Sigma_opt']

    # Compute error metrics
    results_dict["MaxEnt_mahalanobis"].append(float(mahalanobis_cov_error(Sigma_true, Sigma_MaxEnt)))
    results_dict["SSP_mahalanobis"].append(float(mahalanobis_cov_error(Sigma_true, Sigma_SSP)))
    results_dict["MaxEnt_l2"].append(float(np.linalg.norm(Sigma_true - Sigma_MaxEnt)))
    results_dict["SSP_l2"].append(float(np.linalg.norm(Sigma_true - Sigma_SSP)))
    results_dict["MaxEnt_mse"].append(float(np.mean((Sigma_true - Sigma_MaxEnt) ** 2)))
    results_dict["SSP_mse"].append(float(np.mean((Sigma_true - Sigma_SSP) ** 2)))

# Save results to JSON file
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
suffix = f"{args.optimizer}_synthdata_d{args.d}_rho{args.rho}_offdiagpct{offdiagpct}_{clip_type}_rescale{rescale}"
output_filename = os.path.join(results_dir, f"{suffix}_errors.json")
with open(output_filename, "w") as f:
    json.dump(results_dict, f, indent=4)
print(f"Results saved to {output_filename}")




