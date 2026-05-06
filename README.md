# PACE-GGM: Private Adaptive Covariance Estimation via Gaussian Graphical Models

## Requirements

Python 3.10+ is recommended. Install dependencies via pip:

```bash
pip install numpy scipy scikit-learn pandas matplotlib networkx jax optax
```

**Optional — AdaptiveCov/CoinPress baselines only:**

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
  baselines.py       # AdaSSP, diagonal regression, AdaptiveCov, CoinPress wrappers
  solvers/           # IPM (default), PGD-MaxEnt
  data.py, data_reg.py, regression.py, utils.py

experiments/
  test_iterative_annealing.py      # Main experiment runner (covariance + regression)
  run_privcovariance_baselines.py  # AdaptiveCov / CoinPress baseline runner
  main_plotting_function.py        # Covariance estimation plots
  reg_plotting_function.py         # Regression MSE plots

data/                              # Dataset CSVs and target column mapping
privateggm/sbatch/                 # SLURM job scripts (see Step 1–3 below)
```

---

## Reproducing the experiments

All experiments save results to a `results/` folder in the repository root.
Run from the repository root with the conda environment activated.

### Step 1 — Covariance estimation (PACE-GGM vs SSP vs Independent)

```bash
sbatch privateggm/sbatch/sbatch_neurips_5datasets.sh
```

Runs PACE-GGM on all 5 datasets (adult, thyroid_ann, ibm, bank32nh, communities_crime).
Results saved to `results/neurips_5datasets/`.

> **Note:** `communities_crime` uses `--max-rounds-pct 0.5` due to its high dimensionality (d=102); all others use `--max-rounds-pct 2.0`.

To run a single dataset manually:

```bash
python -m experiments.test_iterative_annealing \
    --data-type real --data-name adult \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
    --alpha 0.3 --beta 0.5 --max-rounds-pct 2.0 \
    --n-trials 10 --sensitivity-type linf \
    --allow-remeasurement --optimizer ipm_chol \
    --output-dir results/neurips_5datasets --seed 42
```

### Step 2 — Covariance baselines (AdaptiveCov / CoinPress)

Requires `PRIVATE_COV_REPO` to be set (see Requirements above).

```bash
sbatch privateggm/sbatch/sbatch_neurips_privcov_baselines.sh
```

Results saved to `results/neurips_privcov_baselines/`.

### Step 3 — Regression (PACE-GGM vs SSP vs AdaSSP)

```bash
sbatch privateggm/sbatch/sbatch_neurips_5datasets_reg.sh
```

Runs regression on adult, bank32nh, ibm, communities_crime (80/20 train/test split).
Results saved to `results/neurips_5datasets_reg/`.

---

## Plotting

### Covariance estimation plot (with AdaptiveCov overlay)

Run after Steps 1 and 2:

```bash
python -m experiments.main_plotting_function results/neurips_5datasets \
    --privcovariance-folder results/neurips_privcov_baselines \
    --no-coinpress \
    --n-per-plot 5 \
    --output-tag neurips_5datasets_with_baselines
```

To plot PACE-GGM, SSP and Independent only (no baselines):

```bash
python -m experiments.main_plotting_function results/neurips_5datasets \
    --n-per-plot 5 --output-tag neurips_5datasets
```

### Regression plot

Run after Step 3:

```bash
# PACE-GGM vs AdaSSP
python -m experiments.reg_plotting_function results/neurips_5datasets_reg \
    --stat se --no-ssp --suffix _adassp

# PACE-GGM vs SSP
python -m experiments.reg_plotting_function results/neurips_5datasets_reg \
    --stat se --no-adassp --suffix _ssp
```

Plots are saved alongside the results in `results/neurips_5datasets_reg/`.

---

## Key hyperparameters

| Argument | Value used | Description |
|---|---|---|
| `--alpha` | 0.3 | Fraction of budget per select round |
| `--beta` | 0.5 | Fraction of budget per measure round |
| `--n-trials` | 10 | Independent trials per ρ |
| `--sensitivity-type` | `linf` | L∞ clipping |
| `--optimizer` | `ipm_chol` | Interior point solver (recommended) |
| `--seed` | 42 | Global random seed |
