#!/bin/bash
#SBATCH --job-name=reg_ada_ibm_nr
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=48:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real --data-name ibm \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --regression --target-budget-frac 0.3333 --ssp-target-frac 0.3333 \
    --ridge-lambda 0.0 \
    --output-dir results/a30_b50_reg_adassp/no_ridge --seed 42

echo "Done: ibm no_ridge"
