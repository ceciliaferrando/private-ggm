#!/bin/bash
#SBATCH --job-name=alpha_study_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=200:00:00
#SBATCH --array=0-51

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# 13 datasets (adult → spambase, ordered by d)
datasets=(
    "adult"
    "adult2"
    "SeoulBikeData"
    "LifeExpectancy"
    "Parkinsons"
    "thyroid_ann"
    "ibm"
    "pol_reg"
    "Job"
    "KDD"
    "BreastCancerWisconsin"
    "Facebook"
    "spambase"
)

# 4 alphas (0.3 already running in baseline_alpha30)
alphas=(0.1 0.2 0.4 0.5)

n_datasets=${#datasets[@]}   # 13
alpha_idx=$(( SLURM_ARRAY_TASK_ID / n_datasets ))
dataset_idx=$(( SLURM_ARRAY_TASK_ID % n_datasets ))

dataset=${datasets[$dataset_idx]}
alpha=${alphas[$alpha_idx]}

# Folder name: alpha as integer * 10 (e.g. 0.1 -> alpha10)
alpha_int=$(echo "$alpha * 10" | bc | cut -d. -f1)
output_dir="results/baseline_alpha${alpha_int}0/ipm_solver"

echo "======================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: $dataset  alpha=$alpha  -> $output_dir"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha "$alpha" --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir "$output_dir" --seed 42

echo "======================================"
echo "Completed: $dataset  alpha=$alpha"
echo "======================================"
