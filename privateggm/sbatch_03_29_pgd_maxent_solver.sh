#!/bin/bash
#SBATCH --job-name=pgd_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-13

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Array of datasets to process
datasets=(
    "adult"
    "SeoulBikeData"
    "LifeExpectancy"
    "thyroid_ann"
    "ibm"
    "BreastCancerWisconsin"
    "communities_crime"
    "spambase"
    "hill_valley"
    "madeline"
    "Facebook"
    "pol_reg"
    "KDD"
    "Job"
)

# Get dataset for this array job
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Running: $dataset (pgd_maxent_solver)"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd_maxent_solver \
    --output-dir results/03_29/pgd_maxent_solver --seed 42

echo "======================================"
echo "Completed: $dataset"
echo "======================================"

