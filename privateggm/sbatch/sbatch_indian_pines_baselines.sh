#!/bin/bash
#SBATCH --job-name=privcov_indian_pines
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

# CoinPress t=4 + AdaptiveCov -> main baseline folder
python -m experiments.run_privcovariance_baselines \
    --datasets indian_pines \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 10 --coinpress-t 4 \
    --output-dir results/privcovariance_baselines --seed 42

# CoinPress t=1
python -m experiments.run_privcovariance_baselines \
    --datasets indian_pines \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 10 --coinpress-t 1 \
    --output-dir results/privcovariance_baselines_t1 --seed 42

# CoinPress t=2
python -m experiments.run_privcovariance_baselines \
    --datasets indian_pines \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 10 --coinpress-t 2 \
    --output-dir results/privcovariance_baselines_t2 --seed 42

echo "Done: indian_pines baselines (t=1,2,4 + AdaptiveCov)"
