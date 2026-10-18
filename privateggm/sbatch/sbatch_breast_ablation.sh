#!/bin/bash
#SBATCH --job-name=breast_abl_%a
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j_%a.out
#SBATCH --partition=longq
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

alphas=(0.1 0.2 0.3 0.4)
alpha_dirs=(alpha_b50_10 alpha_b50_20 alpha_b50_30 alpha_b50_40)
alpha=${alphas[$SLURM_ARRAY_TASK_ID]}
outdir=${alpha_dirs[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "BreastCancerWisconsin alpha=${alpha} beta=0.5 ablation (Task ${SLURM_ARRAY_TASK_ID})"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "BreastCancerWisconsin" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha ${alpha} --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/${outdir}/ipm_solver --seed 42

echo "======================================"
echo "Completed: BreastCancerWisconsin alpha=${alpha}"
echo "======================================"
