#!/bin/bash
#SBATCH --job-name=mad_sparse_pub_reg
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "madeline ipm_chol_sparse public mean regression"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "madeline" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol_sparse \
    --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
    --output-dir results/highd_sparse/madeline_pub_reg/ipm_sparse_solver --seed 42

echo "Done."
