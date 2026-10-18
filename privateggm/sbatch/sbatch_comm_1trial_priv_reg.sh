#!/bin/bash
#SBATCH --job-name=comm_priv_reg_1t
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "communities_crime" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 0.3 \
    --n-trials 1 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --regression --target-budget-frac 0.2 --ssp-target-frac 0.2 \
    --private-mean \
    --output-dir results/highd_ipm_1trial/communities_crime_priv_reg --seed 42

echo "Done."
