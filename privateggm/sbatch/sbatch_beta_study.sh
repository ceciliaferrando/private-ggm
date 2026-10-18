#!/bin/bash
#SBATCH --job-name=beta_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-59

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

datasets=("adult" "Ailerons" "Facebook" "ibm" "Job" "LifeExpectancy" "Parkinsons" "SeoulBikeData" "spambase" "communities_crime" "pol_reg" "thyroid_ann")
betas=(0.1 0.2 0.3 0.4 0.5)
beta_dirs=(baseline_beta10 baseline_beta20 baseline_beta30 baseline_beta40 baseline_beta50)

n_datasets=${#datasets[@]}
dataset_idx=$((SLURM_ARRAY_TASK_ID % n_datasets))
beta_idx=$((SLURM_ARRAY_TASK_ID / n_datasets))

dataset=${datasets[$dataset_idx]}
beta=${betas[$beta_idx]}
beta_dir=${beta_dirs[$beta_idx]}

echo "======================================"
echo "Beta study: $dataset beta=$beta (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta $beta --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/$beta_dir/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset beta=$beta"
echo "======================================"
