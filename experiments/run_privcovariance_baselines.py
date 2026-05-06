"""
Run AdaptiveCov and CoinPress baselines on real datasets and save results in the
same JSON format used by test_iterative_annealing.py, so they can be overlaid on
existing plots via main_plotting_function.py --privcovariance-folder.

Usage:
    python -m experiments.run_privcovariance_baselines \
        --datasets adult LifeExpectancy bank32nh \
        --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 10.0 \
        --n-trials 5 \
        --output-dir results/privcovariance_baselines

Prerequisites:
    - Clone https://github.com/hkustDB/PrivateCovariance somewhere and set
      PRIVATE_COV_REPO=/path/to/PrivateCovariance (or edit _PRIVATE_COV_REPO
      in privateggm/baselines.py).
    - PyTorch must be installed.
"""
import argparse
import json
import os
import time
import numpy as np

from privateggm.data import load_data, compute_and_save_bounds, rescale_data, load_and_process_data
from privateggm.utils import mahalanobis_cov_error, frobenius_error
from privateggm.baselines import adaptive_cov, coinpress_cov
from privateggm.privacy import privatize_mean


DATASETS_DEFAULT = ['adult', 'LifeExpectancy', 'bank32nh']
RHO_VALUES_DEFAULT = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 10.0]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def _load_dataset(data_name):
    """Load, rescale, and clip a real dataset. Returns (X, Sigma_true, T, d, n)."""
    target_columns_path = os.path.join(DATA_DIR, 'target_columns.json')
    target_col = None
    if os.path.exists(target_columns_path):
        with open(target_columns_path) as f:
            target_col = json.load(f).get(data_name)

    df = load_data(DATA_DIR, data_name=data_name, target_column=target_col, drop_columns=None)
    df = df.dropna()

    bounds_path = compute_and_save_bounds(df, DATA_DIR, rescale=True)
    df_rescaled = rescale_data(df)

    data_path = os.path.join(DATA_DIR, f'{data_name}_processed.csv')
    df_rescaled.to_csv(data_path, index=False)

    processed, n, d, Sigma_true, T = load_and_process_data(data_path, bounds_path, clip_type='linf')
    X = processed.to_numpy()
    return X, Sigma_true, T, d, n


def _run_one_trial(X, Sigma_true, T, rho, seed, coinpress_t=4, private_mean=False):
    """Run one trial of adaptive_cov and coinpress_cov. Returns trial result dict."""
    n, d = X.shape
    rng = np.random.default_rng(seed)

    if private_mean:
        # Split rho 50/50: half for mean, half for covariance (mirrors PACE-GGM privcov)
        rho_mean = rho / 2.0
        rho_cov  = rho / 2.0
        mu_emp = X.mean(axis=0)
        mu_dp, _ = privatize_mean(rng, mu_emp, n, d, T, rho_mean, sensitivity_type='linf')
        X_c = X - mu_dp
    else:
        rho_cov = rho
        X_c = X - X.mean(axis=0)

    # L2 row-norm bound after L_inf clipping: sqrt(d)*T
    r = float(np.sqrt(d) * T)

    trial = {}

    # ── AdaptiveCov ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        Sigma_adaptive = adaptive_cov(X_c, rho_cov, r=r)
        trial['adaptive_cov'] = {
            'mahalanobis_error': float(mahalanobis_cov_error(Sigma_true, Sigma_adaptive)),
            'frobenius_error':   float(frobenius_error(Sigma_true, Sigma_adaptive)),
            'trial_time_seconds': float(time.perf_counter() - t0),
        }
    except Exception as e:
        print(f"    adaptive_cov failed: {e}")
        trial['adaptive_cov'] = {'mahalanobis_error': None, 'frobenius_error': None,
                                 'trial_time_seconds': None, 'error': str(e)}

    # ── CoinPress ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        Sigma_cp = coinpress_cov(X_c, rho_cov, r=r, t=coinpress_t)
        trial['coinpress'] = {
            'mahalanobis_error': float(mahalanobis_cov_error(Sigma_true, Sigma_cp)),
            'frobenius_error':   float(frobenius_error(Sigma_true, Sigma_cp)),
            'trial_time_seconds': float(time.perf_counter() - t0),
        }
    except Exception as e:
        print(f"    coinpress failed: {e}")
        trial['coinpress'] = {'mahalanobis_error': None, 'frobenius_error': None,
                              'trial_time_seconds': None, 'error': str(e)}

    return trial


def run_dataset(data_name, rho_values, n_trials, output_dir, seed=42, coinpress_t=4,
                private_mean=False):
    print(f"\n{'='*70}")
    print(f"Dataset: {data_name}")
    print(f"{'='*70}")

    X, Sigma_true, T, d, n = _load_dataset(data_name)
    print(f"  X shape: {X.shape}, T={T:.4f}")

    results = {
        'parameters': {
            'dataset': data_name,
            'd': int(d),
            'n': int(n),
            'rho_values': rho_values,
            'n_trials': n_trials,
            'sensitivity_type': 'linf',
            'methods': ['adaptive_cov', 'coinpress'],
            'coinpress_t': coinpress_t,
            'private_mean': private_mean,
            'seed': seed,
        },
        'results': {},
    }

    for rho in rho_values:
        print(f"\n  rho={rho}")
        trials = []
        for t in range(n_trials):
            print(f"    trial {t+1}/{n_trials}", end='', flush=True)
            trial_seed = seed + t * 10000 + int(rho * 1e6)
            res = _run_one_trial(X, Sigma_true, T, rho, trial_seed, coinpress_t=coinpress_t,
                                 private_mean=private_mean)
            trials.append(res)
            mah_a = res['adaptive_cov'].get('mahalanobis_error')
            mah_c = res['coinpress'].get('mahalanobis_error')
            print(f"  adaptive={mah_a:.3f}  coinpress={mah_c:.3f}" if (mah_a and mah_c) else "  (error)")
        results['results'][str(rho)] = trials

    os.makedirs(output_dir, exist_ok=True)
    pm_tag = '_privmean' if private_mean else ''
    out_path = os.path.join(output_dir,
                            f'privcovariance_{data_name}_d{d}_{n_trials}trials_cpt{coinpress_t}{pm_tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS_DEFAULT,
                        help='Dataset names to run')
    parser.add_argument('--rho-values', nargs='+', type=float, default=RHO_VALUES_DEFAULT)
    parser.add_argument('--n-trials', type=int, default=5)
    parser.add_argument('--coinpress-t', type=int, default=4,
                        help='Number of CoinPress iterations (default 4)')
    parser.add_argument('--output-dir', type=str, default='results/privcovariance_baselines')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--private-mean', action='store_true', default=False,
                        help='Split rho 50/50: half for private mean, half for covariance '
                             '(mirrors PACE-GGM private-mean pipeline)')
    args = parser.parse_args()

    for ds in args.datasets:
        run_dataset(ds, args.rho_values, args.n_trials, args.output_dir, args.seed,
                    coinpress_t=args.coinpress_t, private_mean=args.private_mean)

    print(f"\nDone. Results in: {args.output_dir}")
