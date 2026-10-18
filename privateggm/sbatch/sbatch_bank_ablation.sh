#!/bin/bash
#SBATCH --job-name=bank_ablation
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=50:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real --data-name bank32nh \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd_pure \
    --output-dir results/updates_solvers/pgd_fu0 --seed 41

python -m experiments.test_iterative_annealing \
    --data-type real --data-name bank32nh \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd_pure_fu1 \
    --output-dir results/updates_solvers/pgd_fu1 --seed 41

python -m experiments.test_iterative_annealing \
    --data-type real --data-name bank32nh \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol_nocc \
    --output-dir results/updates_solvers/ipm_nocc --seed 41

echo "Done: bank32nh fill ablation seed 41"
