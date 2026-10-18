#!/bin/bash
#SBATCH --job-name=debug_comm_rho0001
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=10:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real \
    --data-name communities_crime \
    --rho-values 0.0001 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 1 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/debug_comm_rho0001 --seed 42

echo "Done: communities_crime rho=0.0001 debug"
