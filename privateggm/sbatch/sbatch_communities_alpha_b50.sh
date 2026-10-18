#!/bin/bash
#SBATCH --job-name=comm_a_b50_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --array=0-4

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

alphas=(0.1 0.2 0.3 0.4 0.5)
alpha_dirs=(alpha_b50_10 alpha_b50_20 alpha_b50_30 alpha_b50_40 alpha_b50_50)

alpha=${alphas[$SLURM_ARRAY_TASK_ID]}
alpha_dir=${alpha_dirs[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "communities_crime alpha=$alpha beta=0.5 (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "communities_crime" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha $alpha --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/$alpha_dir/ipm_solver --seed 42

echo "======================================"
echo "Completed: communities_crime alpha=$alpha"
echo "======================================"
