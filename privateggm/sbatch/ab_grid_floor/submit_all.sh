#!/bin/bash
# Submit all 45 alpha/beta grid jobs (9 cells x 5 datasets).
#   ./submit_all.sh
# Override paths first if the repo is not at the default location:
#   export PRIVATE_GGM_REPO=/path/to/private-ggm
#   export PRIVATE_GGM_CONDA=/path/to/conda_env

cd "$(dirname "$0")" || exit 1

for f in sbatch_grid_a*.sh; do
    echo "submitting $f"
    sbatch "$f"
done

echo
echo "Submitted 45 jobs. Watch with:  squeue -u \$USER"
