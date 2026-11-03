#!/bin/bash
#SBATCH --job-name=densebig2
#SBATCH --output=exp_%A_%a.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=200:00:00
#SBATCH --array=0-1

# indian_pines (d=187) and madeline (d=260) on the DENSE solver.
#
# These are the only two datasets in results/cr_floor_12datasets that ran on
# ipm_chol_sparse; every other dataset used ipm_chol. Re-running them dense
# makes the main results solver-consistent. Written to a separate output
# directory so the existing sparse results are left untouched.
#
# All other settings match results/cr_floor_12datasets exactly: alpha=0.3,
# beta=0.5, 10 trials, floor on, the same 7 rho values, seed 42.
#
# Runtime warning: indian_pines took ~3400s per trial at rho=10 on the SPARSE
# solver, and the dense/sparse crossover sits near d=512, so dense may well be
# slower still at d=187. This can run for days; results are written per dataset
# so madeline will land long before indian_pines.

REPO=${PRIVATE_GGM_REPO:-/mnt/nfs/work1/sheldon/cferrando/private-ggm}
CONDA=${PRIVATE_GGM_CONDA:-/mnt/nfs/work1/sheldon/cferrando/conda_env}

cd "$REPO"
source "$CONDA/bin/activate"

DATASETS=(madeline indian_pines)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "task $SLURM_ARRAY_TASK_ID -> $DS on ipm_chol (dense)"

python -u -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name "$DS" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/cr_dense_bigtwo --seed 42

echo "Done: $DS (dense, alpha=0.3 beta=0.5)"
