#!/bin/bash
#SBATCH --job-name=bank_seed41
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
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/updates_bank_seed41 --seed 41

echo "Done: bank32nh seed 41"
