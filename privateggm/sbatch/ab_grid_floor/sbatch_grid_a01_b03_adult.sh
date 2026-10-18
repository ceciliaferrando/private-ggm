#!/bin/bash
#SBATCH --job-name=gridf_a01_b03_adult
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

# Override if the repo lives elsewhere:
#   export PRIVATE_GGM_REPO=/path/to/private-ggm
#   export PRIVATE_GGM_CONDA=/path/to/conda_env
REPO=${PRIVATE_GGM_REPO:-/mnt/nfs/work1/sheldon/cferrando/private-ggm}
CONDA=${PRIVATE_GGM_CONDA:-/mnt/nfs/work1/sheldon/cferrando/conda_env}

cd "$REPO"
source "$CONDA/bin/activate"

python -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name adult \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.1 --beta 0.3 --max-rounds-pct 2.0 \
    --n-trials 5 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --diag-noise-floor \
    --output-dir results/ab_grid_floor/a01_b03/ipm_solver --seed 42

echo "Done: adult alpha=0.1 beta=0.3 (floor)"
