#!/bin/bash
#SBATCH --job-name=pm_ibm
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

# cd $HOME/private-ggm
# source $HOME/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name ibm \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --private-mean \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/final_a03_b05_priv/ipm_solver --seed 42

echo "Done: ibm privmean alpha=0.3 beta=0.5"
