#!/bin/bash
#SBATCH --job-name=reg_ailerons
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-1

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "Ailerons regression (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # Public mean
    python -m experiments.test_iterative_annealing \
        --data-type real --data-name "Ailerons" \
        --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
        --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
        --n-trials 5 --sensitivity-type linf \
        --allow-remeasurement --optimizer ipm_chol \
        --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
        --output-dir results/reg_pubmean_a50_5trials/ipm_solver --seed 42
else
    # Private mean
    python -m experiments.test_iterative_annealing \
        --data-type real --data-name "Ailerons" \
        --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
        --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
        --n-trials 5 --sensitivity-type linf \
        --allow-remeasurement --optimizer ipm_chol \
        --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
        --private-mean \
        --output-dir results/reg_privmean_a50_5trials/ipm_solver --seed 42
fi

echo "======================================"
echo "Completed: Ailerons regression (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"
