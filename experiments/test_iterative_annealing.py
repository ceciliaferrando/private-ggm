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
from privateggm.model import AnnealingIterativePrivateGGM
from privateggm.utils import (
    mahalanobis_cov_error,
    frobenius_error
)
from privateggm.privacy import SSP_covariance, cov_sensitivity_l2, cov_sensitivity_linf


def save_results(results, output_file):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  → Results saved to {output_file}")


def run_experiment(
    data_type='synthetic',
    data_name=None,
    d=10,
    n=1000,
    rho_values=[0.1, 1.0, 10.0],
    alpha=0.2,
    max_rounds_pct=0.3,
    n_trials=5,
    sensitivity_type='linf',
    allow_remeasurement=True,
    output_dir='results/linesearch',
    optimizer='pgd_maxent_solver',
    seed=42
):
    """
    Run annealing experiments.
    
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
        alpha : float
            Budget split parameter (fraction for diagonal)
        max_rounds_pct : float
            Maximum rounds as percentage of off-diagonal entries (e.g., 0.3 for 30%, 1.0 for 100%)
        n_trials : int
            Number of trials for each configuration
        sensitivity_type : str
            Type of sensitivity ('l2' or 'linf')
        allow_remeasurement : bool
            Whether to allow remeasuring entries
        output_dir : str
            Output directory for results
        optimizer : str
            Optimization method ('pgd-maxent', 'ipm_chol', or 'lbfgs')
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
            'alpha': alpha,
            'n_trials': n_trials,
            'sensitivity_type': sensitivity_type,
            'allow_remeasurement': allow_remeasurement,
            'optimizer': optimizer,
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
        
        if os.path.exists(target_columns_path):
            with open(target_columns_path, 'r') as f:
                target_columns = json.load(f)
                target_col = target_columns.get(data_name)
        
        if target_col:
            print(f"Loading {data_name} to determine dimension...")
            print(f"Target column for {data_name}: {target_col}")
        else:
            print(f"Warning: No target column specified for {data_name}")
        
        # Load and process the data
        df = load_data(outdir, data_name=data_name, target_column=target_col, drop_columns=None)
        
        print(f"Loaded {data_name}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if target_col and target_col in df.columns:
            print(f"Dropping target column: {target_col}")
            df = df.drop(columns=[target_col])
            print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print(f"Columns: {list(df.columns)}")
        
        # Get dimension from data
        d = df.shape[1]
        
        # Handle missing values
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Drop rows with any missing values
        df = df.dropna()
        print(f"Data shape after dropping NaNs: {df.shape}")
        
        # Compute bounds with rescaling
        bounds_path = compute_and_save_bounds(df, outdir, rescale=True)
        
        # Apply rescaling
        from privateggm.data import rescale_data
        df_rescaled = rescale_data(df)
        
        # Save rescaled data
        data_path = os.path.join(outdir, f"{data_name}_processed.csv")
        df_rescaled.to_csv(data_path, index=False)
        
        # Process the data
        processed_data, actual_n, actual_d, Sigma_true, T = load_and_process_data(
            data_path, bounds_path, clip_type=sensitivity_type)
        X = processed_data.to_numpy()
    
    # Update results parameters with actual values
    results['parameters']['d'] = d
    results['parameters']['n'] = X.shape[0]
    results['parameters']['dataset'] = data_name if data_type == 'real' else 'synthetic'
    
    # Set up output file path
    dataset_name = data_name if data_type == 'real' else 'synthetic'
    remeas_str = 'remeasure' if allow_remeasurement else 'no_remeasure'
    output_file = f'{output_dir}/annealing_privacy_analysis_{dataset_name}_d{d}_alpha{alpha}_{remeas_str}.json'
    
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

    print(f"\nAnnealing mode parameters:")
    print(f"  Allow remeasurement: {allow_remeasurement}")
    print(f"  Weight exponent: 0.3 (hardcoded in model)")

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
                
                # Run annealing algorithm
                print(f"Running annealing algorithm with:")
                print(f"  d={d}, rho={rho}, alpha={alpha}")
                print(f"  sensitivity_type={sensitivity_type}, norm_bound={T}")
                print(f"  allow_remeasurement={allow_remeasurement}")
                
                # Set max_rounds based on percentage of off-diagonal entries
                n_off_diag = d * (d - 1) // 2
                max_rounds = int(max_rounds_pct * n_off_diag)
                print(f"  max_rounds: {max_rounds} ({max_rounds_pct*100:.0f}% of {n_off_diag} off-diagonal entries)")
                
                model = AnnealingIterativePrivateGGM(
                    rng=trial_rng,
                    d=d,
                    rho=rho,
                    alpha=alpha,
                    beta=0.1,  # 10% for selection, 90% for measurement (AIM paper default)
                    max_rounds=max_rounds,
                    sensitivity_type=sensitivity_type,
                    norm_bound=T,
                    optimizer=optimizer,
                    allow_remeasurement=allow_remeasurement
                )
                
                Sigma_est = model.fit(X)
                
                # Save convergence information for trial 0 only (in same folder as main results)
                if trial == 0 and hasattr(model, 'convergence_history') and model.convergence_history:
                    dataset_label = data_name if data_type == 'real' else f'synthetic_d{d}'
                    
                    # Convert JAX arrays to regular Python lists for JSON serialization
                    def convert_to_serializable(obj):
                        """Convert JAX arrays and numpy arrays to Python lists recursively"""
                        if hasattr(obj, 'tolist'):  # numpy or JAX array
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_to_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_serializable(item) for item in obj]
                        else:
                            return obj
                    
                    conv_data = {
                        'settings': {
                            'dataset': dataset_label,
                            'd': d,
                            'n': X.shape[0],
                            'rho': rho,
                            'alpha': alpha,
                            'beta': 0.1,
                            'max_rounds': max_rounds
                        },
                        'convergence_history': convert_to_serializable(model.convergence_history)
                    }
                    
                    # Save as JSON in same folder as main results
                    json_path = os.path.join(output_dir, f'convergence_{dataset_label}_rho{rho}.json')
                    with open(json_path, 'w') as f:
                        json.dump(conv_data, f, indent=2)
                
                # Compute errors for annealing method
                mah_error = mahalanobis_cov_error(Sigma_true, Sigma_est)
                frob_error = frobenius_error(Sigma_true, Sigma_est)
                
                # Compute diagonal-only baseline errors
                Sigma_baseline = model.Sigma_baseline if hasattr(model, 'Sigma_baseline') else None
                if Sigma_baseline is not None:
                    mah_error_baseline = mahalanobis_cov_error(Sigma_true, Sigma_baseline)
                    frob_error_baseline = frobenius_error(Sigma_true, Sigma_baseline)
                else:
                    mah_error_baseline = None
                    frob_error_baseline = None
                
                # Compute SSP baseline with same privacy budget
                Sigma_SSP, ssp_noise_scale = SSP_covariance(Sigma_true, ssp_sensitivity, rho, rng=trial_rng)
                mah_error_ssp = mahalanobis_cov_error(Sigma_true, Sigma_SSP)
                frob_error_ssp = frobenius_error(Sigma_true, Sigma_SSP)
                
                # Convert entry_to_rounds to JSON-serializable format (dict keys must be strings)
                entry_to_rounds_serialized = {}
                if hasattr(model, 'entry_to_rounds'):
                    for (i, j), rounds_list in model.entry_to_rounds.items():
                        key = f"({i},{j})"  # Convert tuple key to string
                        entry_to_rounds_serialized[key] = rounds_list
                
                # Extract KKT conditions from convergence history
                kkt_summary = {}
                if hasattr(model, 'convergence_history') and model.convergence_history:
                    for round_idx, conv_info in enumerate(model.convergence_history):
                        if 'pgd' in conv_info and 'kkt' in conv_info['pgd']:
                            kkt = conv_info['pgd']['kkt']
                            kkt_summary[f'round_{round_idx}'] = {
                                'is_optimum': bool(kkt['is_optimum']),
                                'primal_feasibility': float(kkt['primal_feasibility']),
                                'dual_feasibility': float(kkt['dual_feasibility']),
                                'complementary_slackness': float(kkt['complementary_slackness']),
                                'min_eig_W': float(kkt['min_eig_W']),
                                'min_eig_grad': float(kkt['min_eig_grad'])
                            }
                
                # Store results
                trial_results = {
                    'annealing': {
                        'mahalanobis_error': float(mah_error),
                        'frobenius_error': float(frob_error),
                        'estimated_cov': Sigma_est.tolist(),
                        'n_rounds': len(model.measurements),
                        'total_measurements': len(model.measurements),
                        'measurements': {f"({i},{j})": float(v) for (i, j), v in model.measurements.items()},
                        'noise_scales': {f"({i},{j})": float(v) for (i, j), v in model.noise_scales.items()},
                        'entry_to_rounds': entry_to_rounds_serialized,
                        'kkt_conditions': kkt_summary
                    },
                    'baseline': {
                        'mahalanobis_error': float(mah_error_baseline) if mah_error_baseline is not None else None,
                        'frobenius_error': float(frob_error_baseline) if frob_error_baseline is not None else None,
                        'estimated_cov': Sigma_baseline.tolist() if Sigma_baseline is not None else None
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
                
                print(f"Trial completed successfully")
                print(f"  Annealing: Mah={mah_error:.4f}, Frob={frob_error:.4f}")
                if mah_error_baseline is not None:
                    print(f"  Baseline:  Mah={mah_error_baseline:.4f}, Frob={frob_error_baseline:.4f}")
                print(f"  SSP:       Mah={mah_error_ssp:.4f}, Frob={frob_error_ssp:.4f}")
                print(f"  Total measurements: {len(model.measurements)}")
                print(f"  SSP noise scale: {ssp_noise_scale:.6e}")
                
                # Print KKT conditions for first and last round
                if hasattr(model, 'convergence_history') and model.convergence_history:
                    print(f"\n  KKT Conditions Summary:")
                    
                    # First round
                    first_round = model.convergence_history[0]
                    if 'pgd' in first_round and 'kkt' in first_round['pgd']:
                        kkt = first_round['pgd']['kkt']
                        print(f"    Round 0 (first):")
                        print(f"      is_optimum={kkt['is_optimum']}, primal_feas={kkt['primal_feasibility']:.2e}, dual_feas={kkt['dual_feasibility']:.2e}")
                        print(f"      comp_slack={kkt['complementary_slackness']:.2e}, min_eig_W={kkt['min_eig_W']:.2e}, min_eig_grad={kkt['min_eig_grad']:.2e}")
                    
                    # Last round
                    last_round = model.convergence_history[-1]
                    if 'pgd' in last_round and 'kkt' in last_round['pgd']:
                        kkt = last_round['pgd']['kkt']
                        print(f"    Round {len(model.convergence_history)-1} (last):")
                        print(f"      is_optimum={kkt['is_optimum']}, primal_feas={kkt['primal_feasibility']:.2e}, dual_feas={kkt['dual_feasibility']:.2e}")
                        print(f"      comp_slack={kkt['complementary_slackness']:.2e}, min_eig_W={kkt['min_eig_W']:.2e}, min_eig_grad={kkt['min_eig_grad']:.2e}")
                
                # Print measurement details
                if hasattr(model, 'entry_to_rounds'):
                    print(f"\n  Measurement tracking (entry_to_rounds):")
                    for (i, j), rounds in sorted(model.entry_to_rounds.items()):
                        meas_val = model.measurements.get((i, j), np.nan)
                        noise_std = model.noise_scales.get((i, j), np.nan)
                        print(f"    ({i},{j}): rounds={rounds}, value={meas_val:.6f}, noise_std={noise_std:.6e}")
                
                # SAVE AFTER EACH TRIAL
                save_results(results, output_file)
                
            except Exception as e:
                print(f"Error in trial {trial + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Still save what we have so far
                save_results(results, output_file)
                continue
    
    print(f"\n{'='*80}")
    print(f"All results saved to {output_file}")
    print(f"{'='*80}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run annealing experiments')
    parser.add_argument('--data-type', type=str, default='synthetic', choices=['synthetic', 'real'],
                        help='Type of data to use')
    parser.add_argument('--data-name', type=str, default=None,
                        help='Name of real dataset (required if data-type=real)')
    parser.add_argument('--d', type=int, default=10,
                        help='Dimension for synthetic data')
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of samples for synthetic data')
    parser.add_argument('--rho-values', nargs='+', type=float, default=[0.1, 1.0, 10.0],
                        help='Privacy budget values to test')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Budget split parameter')
    parser.add_argument('--max-rounds-pct', type=float, default=0.2,
                        help='Maximum rounds as percentage of off-diagonal entries (default: 0.2 for 20%%)')
    parser.add_argument('--n-trials', type=int, default=1,
                        help='Number of trials per configuration')
    parser.add_argument('--sensitivity-type', type=str, default='linf', choices=['l2', 'linf'],
                        help='Sensitivity type')
    parser.add_argument('--allow-remeasurement', action='store_true', default=False,
                        help='Allow remeasuring entries')
    parser.add_argument('--no-remeasurement', action='store_true', default=False,
                        help='Disable remeasuring entries (explicit flag)')
    parser.add_argument('--output-dir', type=str, default='results/linesearch',
                        help='Output directory for results (default: results/linesearch)')
    parser.add_argument('--optimizer', type=str, default='pgd_maxent_solver', choices=['pgd_maxent_solver', 'pgd-maxent', 'ipm_chol', 'lbfgs'],
                        help='Optimization method to use (default: pgd_maxent_solver - Dan\'s optimized PGD solver; pgd-maxent for old model.py version)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Handle remeasurement flags
    if args.no_remeasurement:
        allow_remeasurement = False
    else:
        allow_remeasurement = args.allow_remeasurement
    
    # Validate arguments
    if args.data_type == 'real' and args.data_name is None:
        parser.error("--data-name is required when --data-type=real")
    
    print("="*80)
    print("ANNEALING ITERATIVE PRIVATE GGM EXPERIMENTS")
    print("="*80)
    print(f"Data type: {args.data_type}")
    if args.data_type == 'real':
        print(f"Dataset: {args.data_name}")
    else:
        print(f"Dimension: {args.d}")
        print(f"Samples: {args.n}")
    print(f"Rho values: {args.rho_values}")
    print(f"Alpha: {args.alpha}")
    print(f"Max rounds: {args.max_rounds_pct*100:.0f}% of off-diagonal entries")
    print(f"Trials: {args.n_trials}")
    print(f"Sensitivity type: {args.sensitivity_type}")
    print(f"Allow remeasurement: {allow_remeasurement}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Seed: {args.seed}")
    print("="*80)
    
    run_experiment(
        data_type=args.data_type,
        data_name=args.data_name,
        d=args.d,
        n=args.n,
        rho_values=args.rho_values,
        alpha=args.alpha,
        max_rounds_pct=args.max_rounds_pct,
        n_trials=args.n_trials,
        sensitivity_type=args.sensitivity_type,
        allow_remeasurement=allow_remeasurement,
        output_dir=args.output_dir,
        optimizer=args.optimizer,
        seed=args.seed
    )
