#!/bin/bash
#SBATCH --job-name=pm_madeline
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=100:00:00

# cd $HOME/private-ggm
# source $HOME/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name madeline \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 0.5 \
    --n-trials 10 --sensitivity-type linf \
    --private-mean \
    --allow-remeasurement --optimizer ipm_chol_sparse \
    --output-dir results/final_a03_b05/ipm_solver --seed 42

echo "Done: madeline privmean alpha=0.3 beta=0.5"
