#!/bin/bash
#SBATCH --job-name=beta10_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-11

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

datasets=("adult" "Ailerons" "Facebook" "ibm" "Job" "LifeExpectancy" "Parkinsons" "SeoulBikeData" "spambase" "communities_crime" "pol_reg" "thyroid_ann")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Beta study beta=0.1: $dataset (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.1 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/baseline_beta10/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset beta=0.1"
echo "======================================"
