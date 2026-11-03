#!/bin/bash
#SBATCH --job-name=dense2048
#SBATCH --output=exp_%A_%a.out
#SBATCH --partition=longq
#SBATCH --mem=128G
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4

# The dense solver at d=2048, which benchmark_sparse_vs_dense.py skips.
# One array task per average degree so the five cells run in parallel;
# d=1024 at degree 2.5 took 178s, so each task should be well under an hour.
#
# PYTHONPATH: privateggm is editable-installed pointing at a stale node-local
# /tmp copy, and invoking by script path puts experiments/ at sys.path[0]
# rather than the cwd, so the repo has to be put first explicitly.

REPO=${PRIVATE_GGM_REPO:-/mnt/nfs/work1/sheldon/cferrando/private-ggm}
CONDA=${PRIVATE_GGM_CONDA:-/mnt/nfs/work1/sheldon/cferrando/conda_env}

cd "$REPO"
source "$CONDA/bin/activate"
export PYTHONPATH="$REPO:$PYTHONPATH"

DEGREES=(0.5 1.0 1.5 2.0 2.5)
DEG=${DEGREES[$SLURM_ARRAY_TASK_ID]}

python -u - "$DEG" <<'PYEOF'
import sys, pickle, os
sys.path.insert(0, 'experiments')
from benchmark_sparse_vs_dense import make_problem_by_degree, time_solver

deg = float(sys.argv[1])
d = 2048
prob = make_problem_by_degree(d, deg, seed=0)
print(f"d={d}  avg_deg={deg:.1f}  |off|={prob['n_off']}", flush=True)
t, o = time_solver('ipm_chol', prob)
print(f"    ipm_chol  {t:.2f}s" if t is not None else "    ipm_chol  FAIL", flush=True)

out = f"results/cr_stress_test/dense_d2048_deg{deg}.pkl"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'wb') as f:
    pickle.dump({'d': d, 'avg_degree': deg, 'seed': 0, 'n_edges': prob['n_off'],
                 'time_dense': t, 'time_sparse': None,
                 'obj_dense': o, 'obj_sparse': None}, f)
print(f"Saved -> {out}")
PYEOF

echo "Done: dense d=2048 degree $DEG"
