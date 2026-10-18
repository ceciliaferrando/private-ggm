#!/bin/bash
#SBATCH --job-name=thyroid_ann_goldstandard
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=16G
#SBATCH --time=100:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

echo "======================================"
echo "Running: thyroid_ann (goldstandard)"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name thyroid_ann \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer pgd_maxent \
    --output-dir results/03_29/goldstandard_linesearch --seed 42

echo "======================================"
echo "Completed: thyroid_ann"
echo "======================================"
