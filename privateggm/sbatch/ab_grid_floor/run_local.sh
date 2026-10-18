#!/bin/bash
# Run all 45 alpha/beta grid jobs locally, in parallel, without slurm.
#
#   ./run_local.sh            # default 4 at a time
#   JOBS=8 ./run_local.sh     # override concurrency
#
# Set these first if the repo/conda are not at the cluster defaults:
#   export PRIVATE_GGM_REPO=/path/to/private-ggm
#   export PRIVATE_GGM_CONDA=/path/to/conda_env
#
# The sbatch_*.sh files are ordinary bash scripts -- the #SBATCH lines are
# comments -- so this runs exactly what slurm would have run.

cd "$(dirname "$0")" || exit 1

N=${JOBS:-4}

# One BLAS thread per process, otherwise N numpy jobs each grab every core
# and thrash. Parallelism comes from running several jobs, not from threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

echo "Running 45 jobs, $N at a time. Logs in $(pwd)/logs/"
echo

ls sbatch_grid_a*.sh | xargs -P "$N" -I{} sh -c \
  'echo "  start {}"; if bash {} > "logs/{}.log" 2>&1; then echo "  done  {}"; else echo "  FAIL  {}  (see logs/{}.log)"; fi'

echo
echo "All finished. Results in results/ab_grid_floor/"
echo "Check for failures:  grep -l Traceback logs/*.log"
