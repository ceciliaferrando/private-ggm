#!/bin/bash
#SBATCH --job-name=measurements_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-2

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# 3 datasets with IPM solver
datasets=(
    "adult"
    "SeoulBikeData"
    "ibm"
)

# Get dataset for this array job
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
solver="ipm_chol"

echo "======================================"
echo "Running: $dataset with $solver (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer "$solver" \
    --output-dir results/measurements --seed 42

echo "======================================"
echo "Completed: $dataset with $solver"
echo "======================================"
