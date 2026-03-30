#!/bin/bash
#SBATCH --job-name=goldstandard_new_datasets
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=16G
#SBATCH --time=500:00:00

# Change to the repository directory
cd /mnt/nfs/work1/sheldon/cferrando/private-ggm

# Activate the conda environment
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Run experiments on new datasets with pgd-maxent (goldstandard linesearch)
# Facebook dataset
python -m experiments.test_iterative_annealing \
    --data-type real --data-name Facebook \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd-maxent \
    --output-dir results/goldstandard_linesearch --seed 42

# Job dataset
python -m experiments.test_iterative_annealing \
    --data-type real --data-name Job \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd-maxent \
    --output-dir results/goldstandard_linesearch --seed 42

# KDD dataset
python -m experiments.test_iterative_annealing \
    --data-type real --data-name KDD \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd-maxent \
    --output-dir results/goldstandard_linesearch --seed 42

# pol_reg dataset
python -m experiments.test_iterative_annealing \
    --data-type real --data-name pol_reg \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd-maxent \
    --output-dir results/goldstandard_linesearch --seed 42
