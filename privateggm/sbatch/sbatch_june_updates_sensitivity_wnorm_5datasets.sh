#!/bin/bash
#SBATCH --job-name=june_upd_wnorm
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=200:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

for DS in adult thyroid_ann ibm bank32nh communities_crime; do
python -m experiments.test_iterative_annealing \
    --data-type real --data-name $DS \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --weight-norm median \
    --output-dir results/june_updates_sensitivity_wnorm --seed 42
done

echo "Done: adult, thyroid_ann, ibm, bank32nh, communities_crime"
