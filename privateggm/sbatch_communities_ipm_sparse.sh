#!/bin/bash
#SBATCH --job-name=comm_ipm_sparse
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "communities_crime ipm_chol_sparse all rho"
echo "======================================"

time python -m experiments.test_iterative_annealing \
    --data-type real --data-name "communities_crime" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 1 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol_sparse \
    --output-dir results/communities_ipm_sparse/ipm_sparse_solver --seed 42

echo "======================================"
echo "Done."
echo "======================================"
