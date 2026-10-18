#!/bin/bash
# Submit all 12 floor-experiment jobs. Run from anywhere:
#   ./submit_all.sh
# Override paths first if the repo is not at the default location:
#   export PRIVATE_GGM_REPO=/path/to/private-ggm
#   export PRIVATE_GGM_CONDA=/path/to/conda_env

cd "$(dirname "$0")" || exit 1

for f in sbatch_floor_*.sh; do
    echo "submitting $f"
    sbatch "$f"
done

echo
echo "Submitted. Watch with:  squeue -u \$USER"
