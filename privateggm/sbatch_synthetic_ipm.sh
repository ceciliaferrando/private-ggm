#!/bin/bash
#SBATCH --job-name=ipm_synthetic
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=32G
#SBATCH --time=500:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Run IPM experiments on synthetic data with varying dimensions
for d in 5 10 25 50 100; do
    echo "=========================================="
    echo "Running IPM on synthetic d=$d"
    echo "=========================================="
    python -m experiments.test_iterative_annealing \
        --data-type synthetic --d $d --n 1000 \
        --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
        --alpha 0.2 --max-rounds-pct 2.0 \
        --n-trials 10 --sensitivity-type linf \
        --allow-remeasurement --optimizer ipm_chol \
        --output-dir results/synthetic/ipm_solver --seed 42
done
