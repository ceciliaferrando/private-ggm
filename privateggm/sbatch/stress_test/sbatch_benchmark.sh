#!/bin/bash
#SBATCH --job-name=stress_sparse
#SBATCH --output=exp_%j.out
#SBATCH --partition=longq
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=4

# Scalability stress test promised to the reviewers: ipm_chol vs ipm_chol_sparse
# over d in {32,...,2048} x average degree in {0.5,...,2.5}, 35 cells.
# The dense solver is skipped at d=2048. Partial results are flushed to the
# pickle after every cell, so the run can be inspected before it completes.

REPO=${PRIVATE_GGM_REPO:-/mnt/nfs/work1/sheldon/cferrando/private-ggm}
CONDA=${PRIVATE_GGM_CONDA:-/mnt/nfs/work1/sheldon/cferrando/conda_env}

cd "$REPO"
source "$CONDA/bin/activate"

python -u experiments/benchmark_sparse_vs_dense.py \
    --output "$REPO/results/cr_stress_test/sparse_vs_dense_results.pkl" \
    --seeds 1

echo "Done: sparse vs dense stress test"
