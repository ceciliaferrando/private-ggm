# PACE-GGM: Private Adaptive Covariance Estimation via Gaussian Graphical Models

## Requirements

Python 3.10+ is recommended. Install dependencies via pip:

```bash
pip install numpy scipy scikit-learn pandas matplotlib networkx jax optax
```

**Optional — CoinPress/AdaptiveCov baselines only:**

These baselines require PyTorch and the [PrivateCovariance](https://github.com/hkustDB/PrivateCovariance) library:

```bash
pip install torch
git clone https://github.com/hkustDB/PrivateCovariance.git /path/to/PrivateCovariance
export PRIVATE_COV_REPO=/path/to/PrivateCovariance
```

## Repository structure

```
privateggm/          # Core library
  model.py           # PACE-GGM annealing algorithm
  privacy.py         # Gaussian mechanism, SSP, private mean
  baselines.py       # AdaSSP, IHM, diagonal regression, AdaptiveCov, CoinPress
  solvers/           # IPM (default), PGD-MaxEnt, Clarabel
  data.py, data_reg.py, regression.py, utils.py

experiments/
  test_iterative_annealing.py      # Main experiment runner (covariance + regression)
  run_privcovariance_baselines.py  # AdaptiveCov / CoinPress baseline runner
  run_regression_baselines.py      # IHM / CoinPress regression baseline runner
  main_plotting_function.py        # Covariance estimation plots
  reg_plotting_function.py         # Regression MSE plots

data/                # Dataset CSVs and target column mapping
privateggm/sbatch/   # SLURM job scripts
```

## Running the main experiment

From the repository root:

```bash
python -m experiments.test_iterative_annealing \
    --data-type real --data-name adult \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.2 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/my_run --seed 42
```

Replace `adult` with any dataset name in `data/`. Available datasets: `adult`, `thyroid_ann`, `ibm`, `bank32nh`, `communities_crime`.

**Key arguments:**

| Argument | Description |
|---|---|
| `--rho-values` | Privacy budgets ρ (zCDP) to evaluate |
| `--alpha` | Annealing parameter α (fraction of budget per round) |
| `--max-rounds-pct` | Max rounds as a fraction of off-diagonal entries |
| `--n-trials` | Number of independent trials per ρ |
| `--optimizer` | Solver: `ipm_chol` (default, recommended) or `pgd_maxent_solver` |
| `--allow-remeasurement` | Enable adaptive re-measurement of entries |
| `--regression` | Also run linear regression evaluation |
| `--private-mean` | Split ρ 50/50 between mean and covariance estimation |

**SLURM:** To run all 5 datasets on a cluster:

```bash
sbatch privateggm/sbatch/sbatch_neurips_5datasets.sh
```

## Running covariance baselines (AdaptiveCov / CoinPress)

Requires `PRIVATE_COV_REPO` to be set (see Requirements above).

```bash
python -m experiments.run_privcovariance_baselines \
    --datasets adult thyroid_ann ibm bank32nh communities_crime \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --n-trials 10 --coinpress-t 4 --private-mean \
    --output-dir results/baselines --seed 42
```

## Plotting

**Covariance estimation:**

```bash
python -m experiments.main_plotting_function results/my_run \
    --privcovariance-folder results/baselines \
    --output-dir results/my_run --output-tag my_plots
```

**Regression MSE:**

```bash
python -m experiments.reg_plotting_function results/my_run \
    --stat se --n-rows 2 --n-per-plot 5
```
