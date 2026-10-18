#!/bin/bash
#SBATCH --job-name=reg_pm_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --array=0-8

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

datasets=("adult" "SeoulBikeData" "LifeExpectancy" "Parkinsons" "ibm" "pol_reg" "Ailerons" "spambase" "communities_crime")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Private mean regression: $dataset (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

# Rho values are 2x the original so that after 50/50 split,
# covariance gets the same budget as the non-private-mean experiment.
python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0002 0.002 0.02 0.2 2.0 4.0 20.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
    --private-mean \
    --output-dir results/regression_privatemean/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset private mean regression"
echo "======================================"
