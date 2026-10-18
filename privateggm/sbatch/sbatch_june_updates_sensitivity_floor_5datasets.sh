#!/bin/bash
#SBATCH --job-name=june_upd_sf
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=200:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.test_iterative_annealing \
    --data-type real --data-name adult \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/june_updates_sensitivity_floor --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name thyroid_ann \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/june_updates_sensitivity_floor --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name ibm \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/june_updates_sensitivity_floor --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name bank32nh \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/june_updates_sensitivity_floor --seed 42

python -m experiments.test_iterative_annealing \
    --data-type real --data-name communities_crime \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/june_updates_sensitivity_floor --seed 42

echo "Done: adult, thyroid_ann, ibm, bank32nh, communities_crime"
