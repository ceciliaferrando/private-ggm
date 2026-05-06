"""
Run IHM and CoinPress-based regression baselines on real datasets.

Outputs one JSON per dataset in --output-dir, in the format:
  {
    "parameters": { "dataset", "d", "n", "rho_values", "n_trials", "methods", ... },
    "results": {
      "<rho>": [
        { "ihm":        {"mse": float, "trial_time_seconds": float},
          "coinpress":  {"mse": float, "trial_time_seconds": float},
          ... }
      ]
    }
  }

These files can be overlaid on existing regression plots via
reg_plotting_function.py --ihm-folder.

Usage:
    PRIVATE_COV_REPO=/tmp/PrivateCovariance \\
    python -m experiments.run_regression_baselines \\
        --datasets adult LifeExpectancy bank32nh \\
        --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \\
        --n-trials 5 \\
        --coinpress-t 1 \\
        --output-dir results/reg_ihm_coinpress
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd

from privateggm.data import load_data, compute_and_save_bounds, load_and_process_data
from privateggm.regression import prepare_regression_split, regression_mse
from privateggm.baselines import ihm_regression, coinpress_regression

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATASETS_DEFAULT = ['adult', 'LifeExpectancy', 'bank32nh',
                    'SeoulBikeData', 'communities_crime', 'Ailerons', 'ibm']
RHO_VALUES_DEFAULT = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]


def _load_dataset(data_name, seed=42):
    """Load dataset and return (df_full, target_col)."""
    target_columns_path = os.path.join(DATA_DIR, 'target_columns.json')
    target_col = None
    if os.path.exists(target_columns_path):
        with open(target_columns_path) as f:
            target_col = json.load(f).get(data_name)

    df = load_data(DATA_DIR, data_name=data_name, target_column=target_col, drop_columns=None)
    df = df.dropna()

    if target_col and target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]
    elif 'target' in df.columns:
        # load_data auto-attached a _target.csv — already last column
        target_col = 'target'
    else:
        # Fall back to last column (consistent with test_iterative_annealing.py)
        target_col = df.columns[-1]
        print(f"  No explicit target for {data_name}, using last column: {target_col}")

    return df, target_col


def run_dataset(data_name, rho_values, n_trials, output_dir, seed=42,
                coinpress_t=1, ihm_iters=3, ihm_k=6,
                run_ihm=True, run_coinpress=True):
    print(f"\n{'='*70}")
    print(f"Dataset: {data_name}")
    print(f"{'='*70}")

    df, target_col = _load_dataset(data_name, seed=seed)
    if df is None:
        return None

    # prepare_regression_split expects target as last column
    X_train_c, reg_data, Sigma_true, T, d_full, n_train, mu_used = \
        prepare_regression_split(df, seed=seed)

    d_feat = d_full - 1   # features only (target is last column)
    n = X_train_c.shape[0]
    X_feat = X_train_c[:, :-1]   # (n, d_feat)
    y_train = X_train_c[:, -1]    # (n,)

    X_test_feat = reg_data['X_test_feat']
    y_test      = reg_data['y_test']

    print(f"  n_train={n}, d_feat={d_feat}, T={T:.4f}")

    methods = []
    if run_ihm:
        methods.append('ihm')
    if run_coinpress:
        methods.append(f'coinpress_t{coinpress_t}')

    results = {
        'parameters': {
            'dataset':      data_name,
            'd':            d_feat,
            'n':            n,
            'rho_values':   rho_values,
            'n_trials':     n_trials,
            'methods':      methods,
            'ihm_iters':    ihm_iters,
            'ihm_k':        ihm_k,
            'coinpress_t':  coinpress_t,
            'seed':         seed,
        },
        'results': {},
    }

    for rho in rho_values:
        print(f"\n  rho={rho}")
        trials = []
        for trial_idx in range(n_trials):
            trial_seed = seed + trial_idx * 10000 + int(rho * 1e6)
            rng = np.random.default_rng(trial_seed)
            trial = {}

            # ── IHM ──────────────────────────────────────────────────────────
            if run_ihm:
                t0 = time.perf_counter()
                try:
                    beta_ihm = ihm_regression(
                        X_feat, y_train, rho, T,
                        n_iters=ihm_iters, percentage_k=ihm_k,
                        rng=np.random.default_rng(trial_seed),
                    )
                    mse_ihm = regression_mse(X_test_feat, y_test, beta_ihm)
                    trial['ihm'] = {
                        'mse':                 float(mse_ihm),
                        'trial_time_seconds':  float(time.perf_counter() - t0),
                    }
                except Exception as e:
                    print(f"    ihm failed: {e}")
                    trial['ihm'] = {'mse': None, 'trial_time_seconds': None, 'error': str(e)}

            # ── CoinPress regression ──────────────────────────────────────────
            if run_coinpress:
                t0 = time.perf_counter()
                try:
                    beta_cp = coinpress_regression(
                        X_feat, y_train, rho, T,
                        t=coinpress_t, xty_frac=0.5,
                        rng=np.random.default_rng(trial_seed + 1),
                    )
                    mse_cp = regression_mse(X_test_feat, y_test, beta_cp)
                    trial[f'coinpress_t{coinpress_t}'] = {
                        'mse':                float(mse_cp),
                        'trial_time_seconds': float(time.perf_counter() - t0),
                    }
                except Exception as e:
                    print(f"    coinpress_t{coinpress_t} failed: {e}")
                    trial[f'coinpress_t{coinpress_t}'] = {
                        'mse': None, 'trial_time_seconds': None, 'error': str(e)}

            # Print per-trial summary
            parts = []
            if 'ihm' in trial:
                parts.append(f"IHM={trial['ihm']['mse']:.4f}" if trial['ihm']['mse'] else "IHM=err")
            cp_key = f'coinpress_t{coinpress_t}'
            if cp_key in trial:
                parts.append(f"CP={trial[cp_key]['mse']:.4f}" if trial[cp_key]['mse'] else "CP=err")
            print(f"    trial {trial_idx+1}/{n_trials}  {' | '.join(parts)}")

            trials.append(trial)
        results['results'][str(rho)] = trials

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f'reg_baselines_{data_name}_d{d_feat}_{n_trials}trials_cpt{coinpress_t}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS_DEFAULT)
    parser.add_argument('--rho-values', nargs='+', type=float, default=RHO_VALUES_DEFAULT)
    parser.add_argument('--n-trials', type=int, default=5)
    parser.add_argument('--coinpress-t', type=int, default=1,
                        help='CoinPress iterations for regression (default 1)')
    parser.add_argument('--ihm-iters', type=int, default=3)
    parser.add_argument('--ihm-k', type=int, default=6,
                        help='IHM sketch dimension multiplier (default 6)')
    parser.add_argument('--output-dir', type=str, default='results/reg_ihm_coinpress')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-ihm', action='store_true', help='Skip IHM')
    parser.add_argument('--no-coinpress', action='store_true', help='Skip CoinPress regression')
    args = parser.parse_args()

    for ds in args.datasets:
        run_dataset(
            ds,
            args.rho_values,
            args.n_trials,
            args.output_dir,
            seed=args.seed,
            coinpress_t=args.coinpress_t,
            ihm_iters=args.ihm_iters,
            ihm_k=args.ihm_k,
            run_ihm=not args.no_ihm,
            run_coinpress=not args.no_coinpress,
        )

    print(f"\nDone. Results in: {args.output_dir}")
