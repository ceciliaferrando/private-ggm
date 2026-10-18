#!/bin/bash
#SBATCH --job-name=floor12
#SBATCH --output=exp_%A_%a.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=200:00:00
#SBATCH --array=0-11

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

DATASETS=(adult SeoulBike LifeExpectancy Parkinsons thyroid_ann ibm \
          pol_reg bank32nh Ailerons spambase communities_crime madeline)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== $DS (array task $SLURM_ARRAY_TASK_ID) ==="

python -m experiments.test_iterative_annealing \
    --data-type real --data-name $DS \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/floor_12datasets --seed 42

echo "=== done: $DS ==="
