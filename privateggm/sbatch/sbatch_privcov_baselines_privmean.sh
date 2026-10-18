#!/bin/bash
#SBATCH --job-name=privcov_baselines_privmean
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate
export PRIVATE_COV_REPO=/mnt/nfs/home/cferrando/PrivateCovariance

python -m experiments.run_privcovariance_baselines \
    --datasets adult SeoulBikeData LifeExpectancy Parkinsons thyroid_ann ibm pol_reg \
              BreastCancerWisconsin bank32nh Ailerons spambase communities_crime indian_pines madeline \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 10 --coinpress-t 4 \
    --private-mean \
    --output-dir results/privcovariance_baselines_privmean --seed 42

echo "Done: AdaptiveCov + CoinPress t=4 with private mean (rho split 50/50)"
