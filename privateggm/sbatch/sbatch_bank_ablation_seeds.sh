#!/bin/bash
#SBATCH --job-name=bank_ablation
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=32G
#SBATCH --time=100:00:00

cd /mnt/nfs/work1/sheldon/cferrando/private-ggm
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate

for seed in 42 43 44; do
  for optimizer in ipm_chol ipm_chol_nocc pgd_pure pgd_pure_fu1; do
    case $optimizer in
      ipm_chol)     outdir="ipm_cc"   ;;
      ipm_chol_nocc) outdir="ipm_nocc" ;;
      pgd_pure)     outdir="pgd_fu0"  ;;
      pgd_pure_fu1) outdir="pgd_fu1"  ;;
    esac
    python -m experiments.test_iterative_annealing \
        --data-type real --data-name bank32nh \
        --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
        --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
        --n-trials 10 --sensitivity-type linf \
        --allow-remeasurement --optimizer $optimizer \
        --output-dir results/updates_solvers_bank_seed${seed}/${outdir} --seed $seed
  done
done

echo "Done: bank32nh ablation seeds 42 43 44"
