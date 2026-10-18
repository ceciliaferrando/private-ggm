#!/bin/bash
#SBATCH --job-name=reg_test_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-1

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

datasets=("Ailerons" "spambase")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Regression test: $dataset (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --max-rounds-pct 2.0 \
    --n-trials 1 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/regression/ipm_solver --seed 42 \
    --regression

echo "======================================"
echo "Completed: $dataset"
echo "======================================"
