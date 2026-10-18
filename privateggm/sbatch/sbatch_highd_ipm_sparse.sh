#!/bin/bash
#SBATCH --job-name=highd_sparse_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --array=0-7

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# 0-3: communities_crime, 4-7: madeline
# within each dataset: 0=pub_cov, 1=priv_cov, 2=pub_reg, 3=priv_reg

if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
    dataset="communities_crime"
    task_local=$SLURM_ARRAY_TASK_ID
else
    dataset="madeline"
    task_local=$(( SLURM_ARRAY_TASK_ID - 4 ))
fi

echo "======================================"
echo "High-d ipm_chol_sparse: $dataset task_local=${task_local} (Array ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

BASE_ARGS="--data-type real --data-name $dataset \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol_sparse --seed 42"

if [ "$task_local" -eq 0 ]; then
    # Public mean covariance
    python -m experiments.test_iterative_annealing $BASE_ARGS \
        --output-dir results/highd_sparse/${dataset}_pub_cov/ipm_sparse_solver

elif [ "$task_local" -eq 1 ]; then
    # Private mean covariance
    python -m experiments.test_iterative_annealing $BASE_ARGS \
        --private-mean \
        --output-dir results/highd_sparse/${dataset}_priv_cov/ipm_sparse_solver

elif [ "$task_local" -eq 2 ]; then
    # Public mean regression
    python -m experiments.test_iterative_annealing $BASE_ARGS \
        --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
        --output-dir results/highd_sparse/${dataset}_pub_reg/ipm_sparse_solver

elif [ "$task_local" -eq 3 ]; then
    # Private mean regression
    python -m experiments.test_iterative_annealing $BASE_ARGS \
        --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
        --private-mean \
        --output-dir results/highd_sparse/${dataset}_priv_reg/ipm_sparse_solver
fi

echo "======================================"
echo "Completed: $dataset task_local=${task_local}"
echo "======================================"
