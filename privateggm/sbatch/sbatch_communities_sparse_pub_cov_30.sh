#!/bin/bash
#SBATCH --job-name=comm_sparse_pub_cov
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "communities_crime ipm_chol_sparse public mean covariance"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "communities_crime" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 0.3 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol_sparse \
    --output-dir results/highd_sparse_30/communities_crime_pub_cov/ipm_sparse_solver --seed 42

echo "Done."
