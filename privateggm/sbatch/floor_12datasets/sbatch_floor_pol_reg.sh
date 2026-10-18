#!/bin/bash
#SBATCH --job-name=floor_pol_reg
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

# Override these if the repo lives elsewhere:
#   export PRIVATE_GGM_REPO=/path/to/private-ggm
#   export PRIVATE_GGM_CONDA=/path/to/conda_env
REPO=${PRIVATE_GGM_REPO:-/mnt/nfs/work1/sheldon/cferrando/private-ggm}
CONDA=${PRIVATE_GGM_CONDA:-/mnt/nfs/work1/sheldon/cferrando/conda_env}

cd "$REPO"
source "$CONDA/bin/activate"

python -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name pol_reg \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/floor_12datasets --seed 42

echo "Done: pol_reg (floor, alpha=0.3 beta=0.5)"
