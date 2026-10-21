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
| BreastCancerWisconsin | 32 | 32G | 2.0 | ipm_chol |
| communities_crime | 102 | 32G | 2.0 | ipm_chol |
| indian_pines | 187 | 64G | 2.0 | ipm_chol_sparse |
| madeline | 260 | 64G | 2.0 | ipm_chol_sparse |

(The folder name says 12; it is 14. BreastCancerWisconsin and indian_pines
were added later, after they turned out to be needed for the paper figures.)

**`ipm_chol_sparse` needs `scikit-sparse`.** indian_pines and madeline both
use it. If `sksparse` is missing the failure is quiet: every trial raises
`No module named 'sksparse'`, the experiment loop catches it per trial, and
the job exits 0 having written a results file with zero trials in it. Check
the trial counts, not the exit code. The cluster conda env does not have it;
the mac mini does.

`max-rounds-pct` is uniform at 2.0. It is not binding for any dataset here —
the privacy budget is exhausted first — so the value is effectively
presentational. Note the flag is a raw multiplier on the off-diagonal entry
count (`max_rounds = pct * d(d-1)/2`), not a percentage, despite the name
and the `%` in its log line. For madeline that allows 67,340 rounds while
runs actually stop near 272; the closest any dataset comes to the cap is
adult, at 20 rounds against 30.

madeline uses the sparse solver, and that is the setting that matters for
it: at essentially identical round counts, historical runs took 0.04-0.06h
with `ipm_chol_sparse` versus 15.6-34.7h with dense `ipm_chol`.

Runtime otherwise scales steeply with d: the small datasets finish in
minutes, communities_crime takes roughly an hour.

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
