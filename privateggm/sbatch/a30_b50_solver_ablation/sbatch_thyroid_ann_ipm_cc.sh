#!/bin/bash
#SBATCH --job-name=abl_ipm_cc_thy
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=48:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real --data-name thyroid_ann \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/a30_b50_solver_ablation/ipm_cc --seed 42

echo "Done: thyroid_ann ipm_cc"
