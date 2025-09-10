import numpy as np
import json
import os
from pathlib import Path

from privateggm.data import generate_synthetic_data
from privateggm.model import IterativePrivateGGM
from privateggm.utils import mahalanobis_cov_error, frobenius_error

def run_privacy_analysis(
    d=5,                      # dimension
    n=1000,                   # number of samples
    trials=5,                 # number of trials per rho value
    off_diag_pct=0.3,        # fixed percentage of off-diagonal elements
    rho_values=None,         # privacy budgets to test
    max_iters=None,          # maximum iterations (if None, computed based on off_diag_pct)
    seed=42                  # random seed
):
    if rho_values is None:
        rho_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Compute max_iters based on desired percentage of off-diagonal elements if not provided
    if max_iters is None:
        total_off_diag = (d * (d-1)) // 2  # total possible off-diagonal elements
        max_iters = int(off_diag_pct * total_off_diag)
    
    results = {
        'params': {
            'd': d,
            'n': n,
            'trials': trials,
            'off_diag_pct': off_diag_pct,
            'rho_values': rho_values,
            'max_iters': max_iters,
            'seed': seed
        },
        'results': {}
    }
    
    rng = np.random.default_rng(seed)
    
    for rho in rho_values:
        print(f"\nTesting privacy budget rho = {rho}")
        
        trial_errors = {
            'mahalanobis': [],
            'frobenius': [],
            'measured_entries': []
        }
        
        for trial in range(trials):
            print(f"Trial {trial + 1}/{trials}")
            
            # Generate synthetic data
            X, Sigma_true = generate_synthetic_data(
                rng=rng,
                n=n,
                d=d,
                rescale=True
            )
            
            # Initialize and fit the model
            model = IterativePrivateGGM(
                rng=rng,
                d=d,
                rho=rho,
                alpha=0.2,  # budget split parameter
                max_iters=max_iters,
                sensitivity_type='linf',
                norm_bound=1.0,  # assuming normalized data
                optimizer="pgd-maxent"
            )
            
            model.fit(X)
            
            # Get the final estimate
            Sigma_hat = model.get_current_estimate()
            
            # Calculate errors
            m_error = mahalanobis_cov_error(Sigma_true, Sigma_hat)
            f_error = frobenius_error(Sigma_true, Sigma_hat)
            
            trial_errors['mahalanobis'].append(float(m_error))
            trial_errors['frobenius'].append(float(f_error))
            trial_errors['measured_entries'].append(len(model.measured_entries))
            
        # Compute statistics over trials
        results['results'][str(rho)] = {
            'mean_mahalanobis': float(np.mean(trial_errors['mahalanobis'])),
            'std_mahalanobis': float(np.std(trial_errors['mahalanobis'])),
            'mean_frobenius': float(np.mean(trial_errors['frobenius'])),
            'std_frobenius': float(np.std(trial_errors['frobenius'])),
            'mean_measured': float(np.mean(trial_errors['measured_entries'])),
            'std_measured': float(np.std(trial_errors['measured_entries'])),
            'raw_trials': trial_errors
        }
        
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    filename = f"privacy_analysis_d{d}_offdiag{off_diag_pct}_trials{trials}.json"
    with open(results_dir / filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Example usage
    results = run_privacy_analysis(
        d=5,                  # 5-dimensional problem
        n=1000,              # 1000 samples
        trials=5,            # 5 trials per rho value
        off_diag_pct=0.3,    # measure 30% of off-diagonal entries
        rho_values=[0.01, 0.1, 0.5, 1.0, 2.0],  # privacy budgets to test
        seed=42
    )
    
    # Print summary
    print("\nResults Summary:")
    print("---------------")
    for rho, res in results['results'].items():
        print(f"\nPrivacy budget (rho) = {rho}")
        print(f"Average Mahalanobis Error: {res['mean_mahalanobis']:.4f} ± {res['std_mahalanobis']:.4f}")
        print(f"Average Frobenius Error: {res['mean_frobenius']:.4f} ± {res['std_frobenius']:.4f}")
        print(f"Average Measured Entries: {res['mean_measured']:.1f} ± {res['std_measured']:.1f}")
