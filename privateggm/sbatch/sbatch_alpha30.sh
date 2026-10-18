#!/bin/bash
#SBATCH --job-name=alpha30_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=500:00:00
#SBATCH --array=0-18

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

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
    "mice"
    "bankruptcy"
    "Ailerons"
    "communities_crime"
    "hill_valley"
    "madeline"
)

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Running: $dataset (Task ${SLURM_ARRAY_TASK_ID})"
echo "alpha=0.3, seed=42, seed_formula=42+t"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/baseline_alpha30/ipm_solver --seed 42

echo "======================================"
echo "Completed: $dataset"
echo "======================================"
