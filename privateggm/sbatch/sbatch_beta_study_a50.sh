#!/bin/bash
#SBATCH --job-name=beta_a50_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-10

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

datasets=("adult" "SeoulBikeData" "LifeExpectancy" "Parkinsons" "thyroid_ann" "ibm" "pol_reg" "bank32nh" "Ailerons" "spambase" "BreastCancerWisconsin")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Beta sweep alpha=0.5: $dataset (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

for beta in 0.1 0.2 0.3 0.4 0.5; do
    echo ""
    echo "--- beta=${beta} ---"
    python -m experiments.test_iterative_annealing \
        --data-type real --data-name "$dataset" \
        --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
        --alpha 0.5 --beta ${beta} --max-rounds-pct 2.0 \
        --n-trials 10 --sensitivity-type linf \
        --allow-remeasurement --optimizer ipm_chol \
        --output-dir results/beta_a50_$(echo $beta | tr -d '.0')/ipm_solver --seed 42
done

echo "======================================"
echo "Completed all betas: $dataset"
echo "======================================"
