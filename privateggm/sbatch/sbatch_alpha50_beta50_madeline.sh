#!/bin/bash
#SBATCH --job-name=a50b50_madeline
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=200:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "alpha=0.5 beta=0.5: madeline (d=260)"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name madeline \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/alpha50_beta50/ipm_solver --seed 42

echo "======================================"
echo "Completed: madeline alpha=0.5 beta=0.5"
echo "======================================"
