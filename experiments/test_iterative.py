import numpy as np
import json
import os
import pandas as pd
import argparse

from privateggm.data import (
    generate_well_conditioned_gaussian,
    load_data,
    compute_and_save_bounds,
    load_and_process_data
)
from privateggm.driver import run_iterative_mode
from privateggm.utils import (
    mahalanobis_cov_error,
    frobenius_error
)
from privateggm.privacy import SSP_covariance, cov_sensitivity_l2, cov_sensitivity_linf


def run_experiment(
    data_type='synthetic',
    data_name=None,
    d=10,
    n=1000,
    rho_values=[0.1, 1.0, 10.0],
    off_diag_pct=0.5,
    alpha=0.2,
    n_trials=5,
    sensitivity_type='linf',
    seed=42
):
    """
    Run iterative mode experiments.
    
    Args:
        data_type : str
            'synthetic' or 'real'
        data_name : str
            Name of real dataset (if data_type='real')
        d : int
            Dimension for synthetic data
        n : int
            Number of samples for synthetic data
        rho_values : list
            List of privacy budgets to test
        off_diag_pct : float
            Proportion of off-diagonal entries to measure (e.g., 0.3 = 30%)
        alpha : float
            Budget split parameter (fraction for diagonal)
        n_trials : int
            Number of trials for each configuration
        sensitivity_type : str
            Type of sensitivity ('l2' or 'linf')
        seed : int
            Random seed
    """
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    results = {
        'parameters': {
            'd': d,
            'n': n,
            'rho_values': rho_values,
            'off_diag_pct': off_diag_pct,
            'alpha': alpha,
            'n_trials': n_trials,
            'sensitivity_type': sensitivity_type,
            'seed': seed
        },
        'results': {}
    }
    
    # Set up data directory
    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(outdir, exist_ok=True)

    if data_type == 'synthetic':
        # Generate synthetic data and bounds
        df = generate_well_conditioned_gaussian(
            rng=rng,
            outdir=outdir,
            n_samples=n,
            d=d,
            rescale=True
        )
        data_path = os.path.join(outdir, f"synthdata{d}rescaleTrue.csv")
        bounds_path = os.path.join(outdir, "L_bounds_rescaleTrue.json")
        
        # Process the data
        processed_data, actual_n, actual_d, Sigma_true, T = load_and_process_data(
            data_path, bounds_path, clip_type=sensitivity_type)
        X = processed_data.to_numpy()

    else:  # real data
        # Load target columns from JSON file
        target_columns_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'target_columns.json')
        target_col = None
        drop_cols = None
        
        # Try to load from target_columns.json
        if os.path.exists(target_columns_path):
            with open(target_columns_path, 'r') as f:
                target_columns = json.load(f)
            target_col = target_columns.get(data_name, None)
            print(f"Loaded target column for {data_name}: {target_col}")
        
        if data_name == "communities_crime":
            pass  # data_df already loaded
        else:
            data_df = load_data(outdir, data_name=data_name, target_column=target_col, drop_columns=drop_cols)
        
        # Handle missing values
        print(f"Original data shape: {data_df.shape}")
        print(f"Missing values per column:\n{data_df.isnull().sum()}")
        
        # Drop rows with any missing values
        data_df = data_df.dropna()
        print(f"Data shape after dropping NaNs: {data_df.shape}")
        
        # Compute bounds with rescaling
        bounds_path = compute_and_save_bounds(data_df, outdir, rescale=True)
        
        # Apply rescaling
        from privateggm.data import rescale_data
        data_df_rescaled = rescale_data(data_df)
        
        # Save rescaled data
        temp_data_path = os.path.join(outdir, f"{data_name}_processed.csv")
        data_df_rescaled.to_csv(temp_data_path, index=False)
        
        # Process the data
        processed_data, actual_n, actual_d, Sigma_true, T = load_and_process_data(
            temp_data_path, bounds_path, clip_type=sensitivity_type
        )
        
        X = processed_data.to_numpy()
        _, d = X.shape
    
    # Print data statistics
    print(f"\nData statistics:")
    print(f"X shape: {X.shape}")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Data mean: {X.mean():.3f}")
    print(f"Data std: {X.std():.3f}")
    print(f"Sensitivity bound T: {T:.3f}")
    print(f"Number of NaNs: {np.isnan(X).sum()}")
    print(f"Number of Infs: {np.isinf(X).sum()}")

    # Compute SSP baseline sensitivity
    n_samples, d = X.shape
    if sensitivity_type == 'l2':
        ssp_sensitivity = cov_sensitivity_l2(n_samples, T)
    else:  # linf
        ssp_sensitivity = cov_sensitivity_linf(n_samples, d, T)

    # Compute n_measurements from off_diag_pct
    n_offdiag_total = d * (d - 1) // 2
    n_measurements = max(1, int(off_diag_pct * n_offdiag_total))
    
    print(f"\nIterative mode parameters:")
    print(f"  Total off-diagonal entries: {n_offdiag_total}")
    print(f"  Off-diagonal percentage: {off_diag_pct:.1%}")
    print(f"  Number of measurements: {n_measurements}")

    # Loop over rho values
    for rho in rho_values:
        print(f"\n{'='*80}")
        print(f"Testing rho = {rho}")
        print(f"{'='*80}")
        
        if rho not in results['results']:
            results['results'][rho] = []
        
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            
            try:
                # Create RNG for this trial
                trial_rng = np.random.default_rng(seed=trial * 1000 + 42)
                
                # Run iterative algorithm
                print(f"Running iterative algorithm with:")
                print(f"  d={d}, rho={rho}, alpha={alpha}, n_measurements={n_measurements}")
                print(f"  sensitivity_type={sensitivity_type}, norm_bound={T}")
                
                Sigma_est, model = run_iterative_mode(
                    rng=trial_rng,
                    d=d,
                    X=X,
                    n_measurements=n_measurements,
                    rho=rho,
                    alpha=alpha,
                    sensitivity_type=sensitivity_type,
                    bound=T
                )
                
                # Compute errors for iterative method
                mah_error = mahalanobis_cov_error(Sigma_true, Sigma_est)
                frob_error = frobenius_error(Sigma_true, Sigma_est)
                
                # Compute SSP baseline with same privacy budget
                Sigma_SSP, ssp_noise_scale = SSP_covariance(Sigma_true, ssp_sensitivity, rho, rng=trial_rng)
                mah_error_ssp = mahalanobis_cov_error(Sigma_true, Sigma_SSP)
                frob_error_ssp = frobenius_error(Sigma_true, Sigma_SSP)
                
                # Get actual noise scales from the model
                diag_noise_scale = model.actual_diag_noise_scale
                offdiag_noise_scale = model.actual_offdiag_noise_scale
                
                # Store results
                trial_results = {
                    'iterative': {
                        'mahalanobis_error': float(mah_error),
                        'frobenius_error': float(frob_error),
                        'estimated_cov': Sigma_est.tolist(),
                        'diag_noise_scale': float(diag_noise_scale),
                        'offdiag_noise_scale': float(offdiag_noise_scale),
                        'n_measured_offdiag': int(n_measurements),
                        'n_total_offdiag': int(d * (d-1) // 2)
                    },
                    'ssp': {
                        'mahalanobis_error': float(mah_error_ssp),
                        'frobenius_error': float(frob_error_ssp),
                        'estimated_cov': Sigma_SSP.tolist(),
                        'noise_scale': float(ssp_noise_scale)
                    },
                    'true_cov': Sigma_true.tolist()
                }
                
                results['results'][rho].append(trial_results)
                
                print(f"✓ Trial completed successfully")
                print(f"  Iterative: Mah={mah_error:.4f}, Frob={frob_error:.4f}")
                print(f"  SSP:       Mah={mah_error_ssp:.4f}, Frob={frob_error_ssp:.4f}")
                print(f"  Diag noise scale: {diag_noise_scale:.6e}")
                print(f"  Off-diag noise scale: {offdiag_noise_scale:.6e}")
                print(f"  SSP noise scale: {ssp_noise_scale:.6e}")
                
            except Exception as e:
                print(f"✗ Error in trial {trial + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    dataset_name = data_name if data_type == 'real' else 'synthetic'
    output_file = f'results/iterative/iterative_privacy_analysis_{dataset_name}_d{d}_offdiag{off_diag_pct}_alpha{alpha}.json'
    os.makedirs('results/iterative', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test iterative private GGM')
    
    # Data parameters
    parser.add_argument('--data-type', type=str, default='synthetic',
                        choices=['synthetic', 'real'],
                        help='Type of data to use')
    parser.add_argument('--data-name', type=str, default=None,
                        help='Name of real dataset (required if data-type=real)')
    parser.add_argument('--dimension', type=int, default=None,
                        help='Dimension for synthetic data')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of samples for synthetic data')
    
    # Privacy parameters
    parser.add_argument('--rho-values', type=float, nargs='+',
                        default=[0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0],
                        help='Privacy budget values to test')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Budget split parameter (fraction for diagonal)')
    
    # Iterative parameters
    parser.add_argument('--off-diag', type=float, default=0.3,
                        help='Proportion of off-diagonal entries to measure (e.g., 0.3 = 30%%)')
    
    # Experiment parameters
    parser.add_argument('--n-trials', type=int, default=5,
                        help='Number of trials per configuration')
    parser.add_argument('--sensitivity-type', type=str, default='linf',
                        choices=['l2', 'linf'],
                        help='Type of sensitivity to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.data_type == 'synthetic':
        # Synthetic data experiments
        if args.dimension is not None:
            dimensions = [args.dimension]
        else:
            dimensions = [6, 10, 20]
        
        for d in dimensions:
            print(f"\n{'='*80}")
            print(f"Running synthetic experiment with d={d}")
            print(f"off_diag_pct={args.off_diag} ({args.off_diag*100:.0f}% of off-diagonal entries)")
            print(f"{'='*80}")
            
            results = run_experiment(
                data_type='synthetic',
                d=d,
                n=args.n_samples,
                rho_values=args.rho_values,
                off_diag_pct=args.off_diag,
                alpha=args.alpha,
                n_trials=args.n_trials,
                sensitivity_type=args.sensitivity_type,
                seed=args.seed
            )
    
    else:  # real data
        if args.data_name is None:
            raise ValueError("--data-name is required when --data-type=real")
        
        # For real data, we need to load it first to determine d
        print(f"Loading {args.data_name} to determine dimension...")
        
        # Quick load to get dimension
        outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        target_columns_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'target_columns.json')
        
        if os.path.exists(target_columns_path):
            with open(target_columns_path, 'r') as f:
                target_columns = json.load(f)
            target_col = target_columns.get(args.data_name, None)
        else:
            target_col = None
        
        if args.data_name == "adult":
            target_col = "class"
        elif args.data_name == "life_expectancy":
            target_col = None  # Already processed
        
        data_df = load_data(outdir, data_name=args.data_name, target_column=target_col)
        data_df = data_df.dropna()
        d = data_df.shape[1]
        
        print(f"Dataset dimension: d={d}")
        print(f"off_diag_pct={args.off_diag} ({args.off_diag*100:.0f}% of off-diagonal entries)")
        
        results = run_experiment(
            data_type='real',
            data_name=args.data_name,
            d=d,
            n=None,  # Will be determined from data
            rho_values=args.rho_values,
            off_diag_pct=args.off_diag,
            alpha=args.alpha,
            n_trials=args.n_trials,
            sensitivity_type=args.sensitivity_type,
            seed=args.seed
        )
