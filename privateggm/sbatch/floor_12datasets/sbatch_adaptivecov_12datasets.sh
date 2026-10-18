#!/bin/bash
#SBATCH --job-name=adaptivecov_12
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=100:00:00

# OPTIONAL — only needed if the stored AdaptiveCov results are regenerated.
# results/privcovariance_baselines/ already covers all 12 datasets
# (7 rho x 10 trials). AdaptiveCov and CoinPress come from the external
# PrivateCovariance repo and do not depend on the sensitivity/floor changes,
# so the stored results remain valid for the floor comparison.

REPO=${PRIVATE_GGM_REPO:-/mnt/nfs/work1/sheldon/cferrando/private-ggm}
CONDA=${PRIVATE_GGM_CONDA:-/mnt/nfs/work1/sheldon/cferrando/conda_env}
# Clone https://github.com/hkustDB/PrivateCovariance and point this at it:
export PRIVATE_COV_REPO=${PRIVATE_COV_REPO:-/path/to/PrivateCovariance}

cd "$REPO"
source "$CONDA/bin/activate"

python -m experiments.run_privcovariance_baselines \
    --datasets adult SeoulBikeData LifeExpectancy Parkinsons thyroid_ann ibm \
               pol_reg bank32nh Ailerons spambase communities_crime madeline \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 10 --coinpress-t 4 \
    --output-dir results/floor_12datasets_privcov --seed 42

echo "Done: AdaptiveCov + CoinPress t=4, 12 datasets"
