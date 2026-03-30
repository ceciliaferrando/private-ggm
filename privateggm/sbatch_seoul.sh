#!/bin/bash
#SBATCH --job-name=annealing_seoul
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=16G
#SBATCH --time=500:00:00

# Change to the repository directory
cd /mnt/nfs/work1/sheldon/cferrando/private-ggm

# Activate the conda environment
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Seoul Bike dataset
python -m experiments.test_iterative_annealing \
    --data-type real --data-name SeoulBikeData \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd_maxent_solver \
    --output-dir results/pgd_maxent_new --seed 42

echo "Seoul Bike experiment completed!"
