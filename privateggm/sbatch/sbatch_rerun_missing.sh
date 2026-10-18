#!/bin/bash
#SBATCH --job-name=missing_solvers
#SBATCH --output=/mnt/nfs/work1/sheldon/cferrando/private-ggm/exp_%j.out
#SBATCH --partition=longq        
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --array=0-4

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# Define the missing jobs as (dataset, solver, output_dir)
# Goldstandard missing: Facebook (index 10), ibm (index 4)
# PGD_maxent missing: pol_reg (index 11), communities_crime (index 6), thyroid_ann (index 3)

case $SLURM_ARRAY_TASK_ID in
    0)
        dataset="Facebook"
        solver="pgd-maxent"
        output_dir="results/03_29/goldstandard_linesearch"
        ;;
    1)
        dataset="ibm"
        solver="pgd-maxent"
        output_dir="results/03_29/goldstandard_linesearch"
        ;;
    2)
        dataset="pol_reg"
        solver="pgd_maxent_solver"
        output_dir="results/03_29/pgd_maxent_solver"
        ;;
    3)
        dataset="communities_crime"
        solver="pgd_maxent_solver"
        output_dir="results/03_29/pgd_maxent_solver"
        ;;
    4)
        dataset="thyroid_ann"
        solver="pgd_maxent_solver"
        output_dir="results/03_29/pgd_maxent_solver"
        ;;
esac

echo "======================================"
echo "Running: $dataset ($solver)"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "======================================"

python -m experiments.test_iterative_annealing \
    --data-type real --data-name "$dataset" \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer "$solver" \
    --output-dir "$output_dir" --seed 42

echo "======================================"
echo "Completed: $dataset"
echo "======================================"
