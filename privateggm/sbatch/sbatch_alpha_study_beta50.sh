#!/bin/bash
#SBATCH --job-name=alpha_b50_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --array=0-54

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# 8 datasets x 5 alphas = 40 tasks
# Excluded: Facebook, Job, KDD
datasets=("adult" "Ailerons" "ibm" "LifeExpectancy" "Parkinsons" "SeoulBikeData" "spambase" "communities_crime" "pol_reg" "thyroid_ann" "bank32nh")
alphas=(0.1 0.2 0.3 0.4 0.5)
alpha_dirs=(alpha_b50_10 alpha_b50_20 alpha_b50_30 alpha_b50_40 alpha_b50_50)

n_datasets=${#datasets[@]}
dataset_idx=$((SLURM_ARRAY_TASK_ID % n_datasets))
alpha_idx=$((SLURM_ARRAY_TASK_ID / n_datasets))

dataset=${datasets[$dataset_idx]}
alpha=${alphas[$alpha_idx]}
alpha_dir=${alpha_dirs[$alpha_idx]}

echo "======================================"
echo "Alpha study (beta=0.5): $dataset alpha=$alpha (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha $alpha --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/$alpha_dir/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset alpha=$alpha"
echo "======================================"
