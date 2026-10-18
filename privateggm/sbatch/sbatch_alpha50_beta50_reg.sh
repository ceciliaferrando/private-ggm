#!/bin/bash
#SBATCH --job-name=a50b50_reg_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --array=0-8

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Regression datasets — all have defined or last-column targets
# communities_crime needs 32GB (d=102); others are fine with less but using 32G for safety
datasets=("adult" "SeoulBikeData" "LifeExpectancy" "Parkinsons" "ibm" "pol_reg" "Ailerons" "spambase" "communities_crime")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "alpha=0.5 beta=0.5 REGRESSION: $dataset (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
    --output-dir results/alpha50_beta50/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset alpha=0.5 beta=0.5 regression"
echo "======================================"
