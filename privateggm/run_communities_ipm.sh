#!/bin/bash
source /mnt/nfs/work1/sheldon/cferrando/conda_env/bin/activate && time python -m experiments.test_iterative_annealing \
    --data-type real --data-name "communities_crime" \
    --rho-values 1.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 1 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/test_highd_solver/ipm_solver --seed 42 2>&1 | grep -E "rho|Trial|trial_time|Mah|Frob|SSP|total|real|user|sys|Round|measurements|ipm"
