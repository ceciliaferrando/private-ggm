#!/bin/bash
#SBATCH --job-name=privmean5
#SBATCH --output=exp_%A_%a.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --array=0-4

# Private-mean covariance estimation on the five reference datasets used in the
# main figure and Table 1. Settings are identical to results/cr_floor_12datasets
# (alpha=0.3, beta=0.5, 10 trials, floor on, same 7 rho values, seed 42); the
# only difference is --private-mean, which splits rho 50/50 between estimating
# the mean and estimating the covariance.
#
# Supersedes results/a50b50_privmean_5trials_v2 (2026-04-11), which ran at
# alpha=0.5 with 5 trials, no diagonal noise floor, under the pre-rebuttal
# sensitivities, and omitted communities_crime.
#
# Cheap: the main run's slowest dataset here is communities_crime at ~32s per
# trial at rho=10, so the whole array is well under an hour.

REPO=${PRIVATE_GGM_REPO:-/mnt/nfs/work1/sheldon/cferrando/private-ggm}
CONDA=${PRIVATE_GGM_CONDA:-/mnt/nfs/work1/sheldon/cferrando/conda_env}

cd "$REPO"
source "$CONDA/bin/activate"

DATASETS=(adult thyroid_ann ibm bank32nh communities_crime)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "task $SLURM_ARRAY_TASK_ID -> $DS private mean, alpha=0.3 beta=0.5"

python -u -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name "$DS" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --private-mean \
    --output-dir results/cr_privmean_5datasets --seed 42

echo "Done: $DS (private mean, alpha=0.3 beta=0.5)"
