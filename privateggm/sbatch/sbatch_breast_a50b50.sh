#!/bin/bash
#SBATCH --job-name=breast_a50b50
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=24:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "BreastCancerWisconsin alpha=0.5 beta=0.5 PUBLIC MEAN NO REG"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "BreastCancerWisconsin" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.5 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/alpha_b50_50/ipm_solver --seed 42

echo "======================================"
echo "Completed: BreastCancerWisconsin alpha=0.5 beta=0.5"
echo "======================================"
