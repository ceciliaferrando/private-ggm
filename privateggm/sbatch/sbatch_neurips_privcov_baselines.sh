#!/bin/bash
#SBATCH --job-name=privcov_baselines_neurips
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd /path/to/private-ggm
source /path/to/conda_env/bin/activate
export PRIVATE_COV_REPO=/path/to/PrivateCovariance

python -m experiments.run_privcovariance_baselines \
    --datasets adult thyroid_ann ibm bank32nh communities_crime \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 10 --coinpress-t 4 \
    --private-mean \
    --output-dir results/neurips_privcov_baselines --seed 42

echo "Done: AdaptiveCov + CoinPress t=4 with private mean (rho split 50/50)"
