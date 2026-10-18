#!/bin/bash
#SBATCH --job-name=alpha_miss_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-5

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Missing combinations: Ailerons at alpha 0.1,0.2,0.5 + communities_crime at alpha 0.1,0.2,0.5
# Task 0: Ailerons alpha=0.1
# Task 1: Ailerons alpha=0.2
# Task 2: Ailerons alpha=0.5
# Task 3: communities_crime alpha=0.1
# Task 4: communities_crime alpha=0.2
# Task 5: communities_crime alpha=0.5

datasets=("Ailerons" "Ailerons" "Ailerons" "communities_crime" "communities_crime" "communities_crime")
alphas=(0.1 0.2 0.5 0.1 0.2 0.5)
alpha_dirs=(baseline_alpha10 baseline_alpha20 baseline_alpha50 baseline_alpha10 baseline_alpha20 baseline_alpha50)

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
alpha=${alphas[$SLURM_ARRAY_TASK_ID]}
alpha_dir=${alpha_dirs[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Alpha ablation: $dataset alpha=$alpha (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha $alpha --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/$alpha_dir/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset alpha=$alpha"
echo "======================================"
