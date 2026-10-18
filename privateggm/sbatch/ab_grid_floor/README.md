# Alpha/beta grid ablation — with floor

Re-runs the alpha/beta grid against the current code: tightened per-entry
sensitivities, free diagonal refinement, and the diagonal noise floor
(`--diag-noise-floor`).

3x3 grid, alpha and beta each in {0.1, 0.3, 0.5}, across 5 datasets
(adult d=6, LifeExpectancy d=15, ibm d=23, bank32nh d=33, spambase d=58).
One job per cell per dataset = 45 jobs.

## Running

```bash
# if the repo is not at the cluster default, set these first
export PRIVATE_GGM_REPO=/path/to/private-ggm
export PRIVATE_GGM_CONDA=/path/to/conda_env

./submit_all.sh                        # all 45
sbatch sbatch_grid_a03_b05_adult.sh    # or one at a time
```

Results land in `results/ab_grid_floor/a{01,03,05}_b{01,03,05}/ipm_solver/`.

## Differences from the original grid

The original scripts live loose in `privateggm/sbatch/sbatch_grid_*.sh` and
wrote to `results/ab_grid/`. These differ in three ways:

- `--diag-noise-floor` is passed; the originals predate the flag.
- Output goes to `results/ab_grid_floor/`, so the original results survive
  for comparison. Filenames are otherwise identical and would have been
  overwritten in place.
- Repo and conda paths come from `PRIVATE_GGM_REPO` / `PRIVATE_GGM_CONDA`
  rather than being hardcoded.

Unchanged: 5 trials, no `--random-baseline`, `--max-rounds-pct 2.0`,
`ipm_chol`, 32G, seed 42, same 7 rho values.

The updated sensitivities and the free diagonal refinement apply
automatically — the refinement is unconditional in the model and has no
flag.

Note the original `results/ab_grid/` was produced by the pre-sensitivity
-change code, so those numbers are not reproducible from the current tree.
Keep them if you want the before/after comparison.

## Ignore the un-suffixed originals

`privateggm/sbatch/sbatch_grid_a{01,03,05}_b{01,03,05}.sh` (no dataset
suffix) pass five names to `--data-name`, which takes one value. They are
a superseded earlier attempt and are not what produced `results/ab_grid/`.
