#!/bin/bash
# Submit all 8 regression jobs.
#   ./submit_all.sh
cd "$(dirname "$0")" || exit 1
for f in sbatch_reg_*.sh; do echo "submitting $f"; sbatch "$f"; done
echo; echo "Watch with:  squeue -u \$USER"
