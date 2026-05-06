#!/bin/bash
#SBATCH --job-name=paceggm_5datasets
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=500:00:00

cd /path/to/private-ggm
source /path/to/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real --data-name adult \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/neurips_5datasets --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name thyroid_ann \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/neurips_5datasets --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name ibm \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/neurips_5datasets --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name bank32nh \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/neurips_5datasets --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name communities_crime \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/neurips_5datasets --seed 42

echo "All 5 datasets completed."
