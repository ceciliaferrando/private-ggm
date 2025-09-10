import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from privateggm.data import generate_well_conditioned_gaussian
from privateggm.model import IterativePrivateGGM
from privateggm.utils import mahalanobis_cov_error
from privateggm.privacy import SSP_covariance, cov_sensitivity_l2, cov_sensitivity_linf
from plot import plot_results

def frobenius_error(A, B):
    """Compute the Frobenius norm of the difference between two matrices."""
    return np.linalg.norm(A - B, ord='fro')

def run_experiment(
    d=50,                      # dimension of the data
    n=1000,                    # number of samples
    rho_values=[0.1, 0.5, 1.0, 2.0, 5.0],  # privacy budgets to test
    off_diag_pct=0.3,        # fixed percentage of off-diagonal elements
    n_trials=5,              # number of trials for each rho
    max_iters=None,          # if None, will be set based on off_diag_pct
    alpha=0.2,               # budget split parameter
    sensitivity_type='linf',  # sensitivity type
    norm_bound=1.0,          # norm bound for sensitivity
    seed=42                  # random seed
):
    """
    Run experiment testing the iterative algorithm with different privacy budgets.
    
    Parameters:
    -----------
    d : int
        Dimension of the data
    n : int
        Number of samples
    rho_values : list
        List of privacy budgets to test
    off_diag_pct : float
        Percentage of off-diagonal elements to measure (between 0 and 1)
    n_trials : int
        Number of trials for each rho value
    max_iters : int or None
        Maximum number of iterations. If None, will be set based on off_diag_pct
    alpha : float
        Budget split parameter
    sensitivity_type : str
        Type of sensitivity ('l2' or 'linf')
    norm_bound : float
        Norm bound for sensitivity
    seed : int
        Random seed
    """
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Calculate max_iters based on off_diag_pct if not provided
    if max_iters is None:
        total_off_diag = d * (d-1) // 2  # total number of off-diagonal elements
        max_iters = int(round(total_off_diag * off_diag_pct))  # round instead of floor
        print(f"Total off-diagonal elements: {total_off_diag}")
        print(f"Will measure {max_iters} elements ({off_diag_pct*100}% of off-diagonal elements)")
    
    results = {
        'parameters': {
            'd': d,
            'n': n,
            'rho_values': rho_values,
            'off_diag_pct': off_diag_pct,
            'n_trials': n_trials,
            'max_iters': max_iters,
            'alpha': alpha,
            'sensitivity_type': sensitivity_type,
            'norm_bound': norm_bound,
            'seed': seed
        },
        'results': {}
    }
    
    for rho in rho_values:
        print(f"\nTesting rho = {rho}")
        results['results'][rho] = []
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            
            # Generate synthetic data
            outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(outdir, exist_ok=True)
            df = generate_well_conditioned_gaussian(
                rng=rng,
                outdir=outdir,
                n_samples=n,
                d=d,
                rescale=True
            )
            X = df.to_numpy()
            Sigma_true = df.cov().to_numpy()
            
            # Initialize and run iterative algorithm
            model = IterativePrivateGGM(
                rng=rng,
                d=d,
                rho=rho,
                alpha=alpha,
                max_iters=max_iters,
                sensitivity_type=sensitivity_type,
                norm_bound=norm_bound,
                optimizer="pgd-maxent"
            )
            
            # Fit the model
            model.fit(X)
            
            # Get the estimated covariance matrix
            Sigma_est = model.Sigma_tilde  # or however you get the final estimate
            
            # Compute errors
            mah_error = mahalanobis_cov_error(Sigma_true, Sigma_est)
            frob_error = frobenius_error(Sigma_true, Sigma_est)
            
            # Compute SSP baseline with same privacy budget
            if sensitivity_type == 'l2':
                cov_sensitivity = cov_sensitivity_l2(n, norm_bound)
            else:  # linf
                cov_sensitivity = cov_sensitivity_linf(n, d, norm_bound)
            
            Sigma_SSP = SSP_covariance(Sigma_true, cov_sensitivity, rho)
            mah_error_ssp = mahalanobis_cov_error(Sigma_true, Sigma_SSP)
            frob_error_ssp = frobenius_error(Sigma_true, Sigma_SSP)
            
            # Store results for this trial
            trial_results = {
                'iterative': {
                    'mahalanobis_error': float(mah_error),
                    'frobenius_error': float(frob_error),
                    'measured_entries': len(model.measured_entries),
                    'estimated_cov': Sigma_est.tolist()
                },
                'ssp': {
                    'mahalanobis_error': float(mah_error_ssp),
                    'frobenius_error': float(frob_error_ssp),
                    'estimated_cov': Sigma_SSP.tolist()
                },
                'empirical_cov': model.empirical_cov.tolist(),
                'true_cov': Sigma_true.tolist()
            }
            
            results['results'][rho].append(trial_results)
    
    # Save results
    output_file = f'results/iterative_privacy_analysis_d{d}_offdiag{off_diag_pct}.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results

if __name__ == "__main__":
    # Run experiment with default parameters
    results = run_experiment(
        d=50,                      # small dimension for quick testing
        n=1000,                    # number of samples
        rho_values=[0.1, 0.5, 1.0, 2.0, 5.0],  # privacy budgets
        off_diag_pct=0.2,        # measure 30% of off-diagonal elements
        n_trials=5               # 5 trials per rho value
    )
    
    # Print summary statistics
    print("\nSummary of results:")
    print("------------------")

    # Prepare data for plotting
    plot_data = []
    for rho in results['results']:
        # Iterative method errors
        mah_errors = [trial['iterative']['mahalanobis_error'] for trial in results['results'][rho]]
        frob_errors = [trial['iterative']['frobenius_error'] for trial in results['results'][rho]]
        
        # SSP baseline errors
        mah_errors_ssp = [trial['ssp']['mahalanobis_error'] for trial in results['results'][rho]]
        frob_errors_ssp = [trial['ssp']['frobenius_error'] for trial in results['results'][rho]]
        
        print(f"\nrho = {rho}:")
        print("Iterative method:")
        print(f"  Mahalanobis error: {np.mean(mah_errors):.4f} ± {np.std(mah_errors):.4f}")
        print(f"  Frobenius error: {np.mean(frob_errors):.4f} ± {np.std(frob_errors):.4f}")
        print("SSP baseline:")
        print(f"  Mahalanobis error: {np.mean(mah_errors_ssp):.4f} ± {np.std(mah_errors_ssp):.4f}")
        print(f"  Frobenius error: {np.mean(frob_errors_ssp):.4f} ± {np.std(frob_errors_ssp):.4f}")
        
        # Add data points for plotting
        for err in mah_errors:
            plot_data.append({'rho': rho, 'Error': err, 'Method': 'Iterative', 'metric': 'mahalanobis'})
        for err in mah_errors_ssp:
            plot_data.append({'rho': rho, 'Error': err, 'Method': 'SSP', 'metric': 'mahalanobis'})
        for err in frob_errors:
            plot_data.append({'rho': rho, 'Error': err, 'Method': 'Iterative', 'metric': 'frobenius'})
        for err in frob_errors_ssp:
            plot_data.append({'rho': rho, 'Error': err, 'Method': 'SSP', 'metric': 'frobenius'})

    # Create DataFrame
    df = pd.DataFrame(plot_data)

    # Plot results
    metrics = ['mahalanobis', 'frobenius']
    color_map = {'Iterative': 'blue', 'SSP': 'red'}

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        df_metric = df[df['metric'] == metric]
        
        sns.lineplot(
            data=df_metric,
            x='rho',
            y='Error',
            hue='Method',
            palette=color_map,
            errorbar=('ci', 95),
            marker='o'
        )

        plt.xlabel('Privacy Budget (ρ)')
        plt.ylabel(f'{metric.title()} Error')
        plt.title(f'Iterative vs SSP {metric.title()} Error vs Privacy Budget')
        plt.xscale('log')
        plt.yscale('log')

        # Save the plot
        plot_filename = os.path.join('results', f'iterative_vs_ssp_{metric}_d{results["parameters"]["d"]}_offdiag{results["parameters"]["off_diag_pct"]}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {plot_filename}")
        plt.close()
