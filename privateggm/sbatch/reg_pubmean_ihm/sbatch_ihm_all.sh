#!/bin/bash
#SBATCH --job-name=reg_ihm_all
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=48:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

python -m experiments.run_regression_baselines \
    --datasets adult SeoulBikeData LifeExpectancy ibm bank32nh Ailerons \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 5 \
    --no-coinpress \
    --output-dir results/reg_ihm_pubmean \
    --seed 42

echo "Done: IHM regression all datasets"
