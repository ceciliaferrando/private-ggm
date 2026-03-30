#!/bin/bash
#SBATCH --job-name=pgd_maxent_new
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=16G
#SBATCH --time=500:00:00

# Change to the repository directory
cd /mnt/nfs/work1/sheldon/cferrando/private-ggm

# 1. Activate the conda environment
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# 2. Run the annealing iterative private GGM experiment
# Using -m ensures Python treats the folders as packages
python -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name adult \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 \
    --max-rounds-pct 2.0 \
    --n-trials 10 \
    --sensitivity-type linf \
    --allow-remeasurement \
    --optimizer pgd_maxent_solver \
    --output-dir results/pgd_maxent_new \
    --seed 42
