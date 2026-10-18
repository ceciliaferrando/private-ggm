#!/bin/bash
#SBATCH --job-name=grid_a01_b01
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name adult LifeExpectancy ibm bank32nh spambase \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.1 --beta 0.1 --max-rounds-pct 2.0 \
    --n-trials 5 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/ab_grid/a01_b01/ipm_solver --seed 42

echo "Done: alpha=0.1 beta=0.1"
