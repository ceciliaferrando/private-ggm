#!/bin/bash
#SBATCH --job-name=beta10_bank
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=48:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "Beta study bank32nh beta=0.1"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "bank32nh" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.1 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/baseline_alpha30/ipm_solver --seed 42

echo "======================================"
echo "Completed: bank32nh beta=0.1"
echo "======================================"
