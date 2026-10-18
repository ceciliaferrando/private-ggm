#!/bin/bash
#SBATCH --job-name=a50b50_noreg_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-1

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Non-regression datasets (no meaningful regression target)
datasets=("bank32nh" "thyroid_ann")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "alpha=0.5 beta=0.5 NO-REG: $dataset (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/alpha50_beta50/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset alpha=0.5 beta=0.5 no-regression"
echo "======================================"
