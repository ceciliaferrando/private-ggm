# How the paper's results were produced

Every table and figure in *Private Adaptive Covariance Estimation via Gaussian
Graphical Models* (NeurIPS 2026) comes from one of the commands below. Results
land in `results/<run>/`, which is not tracked: rerun the experiment to
regenerate it, then run the corresponding table or figure command.

Runs were launched on a SLURM cluster, one job per dataset. The launch scripts
were specific to one cluster and one account and are not tracked; the commands
they wrapped are recorded here verbatim.

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
```

`requirements.txt` is a `pip freeze` of the environment the results were
produced in, so it pins more than the package needs. `pyproject.toml` lists the
actual dependencies.

The 20 benchmark datasets are tracked under `data/`. `data/target_columns.json`
designates the regression target for each of the 8 regression datasets.

## Main experiments

The driver is `experiments.test_iterative_annealing`. One invocation sweeps all
seven privacy budgets for one dataset.

**Main results** (14 datasets, `results/cr_floor_12datasets`) — vary
`--data-name` over `adult SeoulBikeData LifeExpectancy Parkinsons thyroid_ann
ibm pol_reg BreastCancerWisconsin bank32nh Ailerons spambase communities_crime
indian_pines madeline`:

```bash
python -m experiments.test_iterative_annealing \
    --data-type real --data-name <dataset> \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --random-baseline --diag-noise-floor \
    --output-dir results/cr_floor_12datasets --seed 42
```

`results/cr_alldense_merged` is the same sweep with every dataset on the dense
`ipm_chol` solver, used for the measurement-count tables.

**Solver ablation** (`results/cr_solver_ablation/<variant>`) — four variants,
`--optimizer` in `ipm_chol` / `pgd_maxent` and connected components on or off:

```bash
python -u -m experiments.test_iterative_annealing \
    --data-type real --data-name <dataset> \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer <opt> \
    --diag-noise-floor \
    --output-dir results/cr_solver_ablation/<variant> --seed 42
```

**Hyperparameter grid** — the nine combinations
$(\alpha,\beta) \in \{0.1,0.2,0.3\} \times \{0.1,0.3,0.5\}$ in
`results/ab_grid_lowalpha`, extended to $\beta \in \{0.7,0.9\}$ at
$\alpha \in \{0.1,0.3\}$ in `results/ab_grid_highbeta`. Same driver with
`--alpha`/`--beta` varied and `--n-trials 5`.

**Regression** (`results/a30_b50_reg_adassp`) — `experiments.run_regression_baselines`,
8 regression datasets, 10 trials.

**Scalability stress test** (`results/cr_stress_test`) — synthetic graphs, 35
cells over $d \in \{32,\dots,2048\}$ and average degree
$\kappa \in \{0.5,\dots,2.5\}$:

```bash
python -u experiments/benchmark_sparse_vs_dense.py \
    --output results/cr_stress_test/sparse_vs_dense_results.pkl --seeds 1
```

The dense solver is skipped above $d=1024$, so the five $d=2048$ cells are
filled in separately, one per average degree:

```bash
python -u experiments/benchmark_dense_d2048.py \
    --output results/cr_stress_test/dense_d2048_deg<deg>.pkl --degree <deg>
```

## Tables

| Paper | Command |
|---|---|
| Measurements (main + appendix) | `python -m experiments.make_measurements_table results/cr_alldense_merged --style both --merge-d --integers --label tab:allmeasurements --output results/0_cr_final/dense_allmeasurements_SINGLE.tex` |
| Trial times | `python -m experiments.make_runtime_table results/cr_floor_12datasets --output results/0_cr_final/main_runtime_table.tex` |
| Hyperparameter relative error | `python -m experiments.make_grid_relerr_table results/ab_grid_lowalpha --output results/0_cr_final/grid_relerr.tex` |
| Stress-test runtimes | emitted by `plot_stress_test.py --table` (see below) |

Drop `--integers` from the measurements command for unrounded means; the
generated caption follows whichever options were passed.

## Figures

| Paper | Command |
|---|---|
| Main results (5 datasets) | `python -m experiments.main_plotting_function results/cr_floor_12datasets --output-dir results/0_cr_final --output-tag dense_5datasets_pace_ssp_adaptivecov` |
| All datasets + diagonal-only | same, with `--indep` and `--group-sizes` |
| Regression | `python -m experiments.reg_plotting_function results/a30_b50_reg_adassp --output-dir results/0_cr_final --output-tag reg10_pace_vs_adassp_2row` |
| Max-entropy vs PGD | `python -m experiments.main_plotting_function results/cr_solver_ablation --ablation-fill --output-dir results/0_cr_final --output-tag cr_ablation_fill_pgd_vs_ipm` |
| Connected components | `python -m experiments.main_plotting_function results/cr_solver_ablation --ablation-cc --output-dir results/0_cr_final --output-tag cr_ablation_ipm_nocc_vs_cc` |
| Hyperparameter grid | `python -m experiments.plot_hyperparam_robustness --grid-dir results/ab_grid_lowalpha results/ab_grid_highbeta --output-dir results/0_cr_final --output-tag abgrid_withhighbeta` |
| Stress test | `python -m experiments.plot_stress_test results/cr_stress_test/sparse_vs_dense_results.pkl --dense-only --extra results/cr_stress_test/dense_d2048_deg*.pkl --table results/0_cr_final/app_runtime_stress.tex` |

## Tests

```bash
pip install pytest && pytest tests/
```

`tests/test_solvers.py` covers the reconstruction solvers: agreement between
`ipm_chol` and the reference solvers, the inverse-variance weighting identity,
and behaviour when the reconstruction problem is ill-posed.

## Notes

- `--diag-noise-floor` and `--allow-remeasurement` are on for every run in the
  paper. Dropping either changes the results.
- `--seed 42` fixes the trial seeds; trial *i* uses seed `42 + i`.
- `indian_pines` at $\rho=10$ takes roughly 30 CPU-minutes per trial. It is
  excluded from the hyperparameter grid for that reason.
- Some early runs used the chordal `ipm_chol_sparse` solver. The paper reports
  the dense `ipm_chol` throughout; `indian_pines` and `madeline` were rerun to
  make the main results solver-consistent. The solver each run actually used is
  recorded in its `convergence_*.json`, not in `parameters.optimizer`.
