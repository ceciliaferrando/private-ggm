# Floor experiment — 12 datasets

PACE-GGM with the diagonal noise floor (`--diag-noise-floor`), run across the
full dataset list. Each run also produces the SSP, AdaSSP, diagonal-only and
random-selection baselines, so one job per dataset covers everything except
AdaptiveCov/CoinPress.

## Running

```bash
# if the repo is not at the cluster default, set these first
export PRIVATE_GGM_REPO=/path/to/private-ggm
export PRIVATE_GGM_CONDA=/path/to/conda_env

./submit_all.sh            # all 12
sbatch sbatch_floor_adult.sh   # or one at a time
```

Results land in `results/floor_12datasets/`.

## What each job runs

`experiments.test_iterative_annealing` with `--random-baseline
--diag-noise-floor`, rho in {1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 10}, 10 trials,
alpha=0.3, beta=0.5, seed 42. SSP and the other baselines are computed
automatically — no extra flag needed.

## Per-dataset settings

| dataset | d | mem | max-rounds-pct | optimizer |
|---|---|---|---|---|
| adult | 6 | 32G | 2.0 | ipm_chol |
| SeoulBikeData | 10 | 32G | 2.0 | ipm_chol |
| LifeExpectancy | 15 | 32G | 2.0 | ipm_chol |
| Parkinsons | 17 | 32G | 2.0 | ipm_chol |
| thyroid_ann | 21 | 32G | 2.0 | ipm_chol |
| ibm | 23 | 32G | 2.0 | ipm_chol |
| pol_reg | 27 | 32G | 2.0 | ipm_chol |
| bank32nh | 33 | 32G | 2.0 | ipm_chol |
| Ailerons | 34 | 32G | 2.0 | ipm_chol |
| spambase | 58 | 32G | 2.0 | ipm_chol |
| communities_crime | 102 | 32G | 2.0 | ipm_chol |
| madeline | 260 | 64G | 0.5 | ipm_chol_sparse |

madeline uses the sparse solver and a reduced round budget, following the
settings used for the earlier full runs. communities_crime uses 2.0 rather
than the older 0.5 — that is what the recent 5-dataset runs used, where it
completed in about 56 minutes.

Runtime scales steeply with d: the small datasets finish in minutes,
communities_crime takes roughly an hour, madeline is the long pole.

## AdaptiveCov

Already stored in `results/privcovariance_baselines/` for all 12 datasets
(7 rho x 10 trials). AdaptiveCov and CoinPress come from the external
PrivateCovariance repo and are untouched by the sensitivity and floor
changes, so those results stay valid and do not need regenerating.
`sbatch_adaptivecov_12datasets.sh` is included only if you want to rerun
them; it needs `PRIVATE_COV_REPO` pointed at a clone of
https://github.com/hkustDB/PrivateCovariance.

## Dataset naming

Use `SeoulBikeData`, not `SeoulBike`. Both load identically (d=10, n=8760),
but every stored result — including the AdaptiveCov baselines — uses
`SeoulBikeData`, and the plotting function matches datasets across folders
by that name.
