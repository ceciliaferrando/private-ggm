#!/bin/bash
#SBATCH --job-name=rerun_miss_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-1

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

datasets=("spambase" "BreastCancerWisconsin")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "RERUN alpha=0.3 beta=0.5 PUBLIC MEAN NO REG: $dataset"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/rerun_a30b50/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset"
echo "======================================"
