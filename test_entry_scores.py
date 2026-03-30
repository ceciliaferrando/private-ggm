#!/usr/bin/env python
"""Quick test of compute_entry_scores with allow_remeasurement on adult dataset"""

import numpy as np
import pandas as pd
import sys
from privateggm.data import load_data
from privateggm.model import AnnealingIterativeGGM

# Load adult dataset
print("Loading adult dataset...")
data_dir = "data"
df = load_data(data_dir, data_name="adult", target_column=None, drop_columns=None)
print(f"Loaded adult: {df.shape}")

X = df.values
n, d = X.shape
print(f"Data shape: {n} samples, {d} features")

# Create a small test with limited rho for quick testing
rng = np.random.default_rng(42)
rho = 1.0
alpha = 0.2
beta = 0.1
max_rounds = 3  # Just 3 rounds for quick test
sensitivity_type = "linf"
norm_bound = np.sqrt(d)

print(f"\nInitializing AnnealingIterativePrivateGGM...")
print(f"  rho={rho}, alpha={alpha}, beta={beta}, max_rounds={max_rounds}")
print(f"  allow_remeasurement=True")

# Test WITH remeasurement allowed
print("\n" + "="*60)
print("TEST 1: allow_remeasurement=True (includes diagonal in scores)")
print("="*60)

ggm_remeasure = AnnealingIterativeGGM(
    rng=rng,
    d=d,
    rho=rho,
    alpha=alpha,
    beta=beta,
    max_rounds=max_rounds,
    sensitivity_type=sensitivity_type,
    norm_bound=norm_bound,
    optimizer="pgd-maxent",
    allow_remeasurement=True
)

Sigma_hat = ggm_remeasure.fit(X)
print(f"\nFinal estimate shape: {Sigma_hat.shape}")
print(f"Number of measured entries: {len(ggm_remeasure.measured_entries)}")
print(f"Measured entries include diagonal: {any(i==j for i,j in ggm_remeasure.measured_entries)}")

# Test WITHOUT remeasurement allowed
print("\n" + "="*60)
print("TEST 2: allow_remeasurement=False (excludes diagonal from scores)")
print("="*60)

ggm_no_remeasure = AnnealingIterativeGGM(
    rng=np.random.default_rng(42),  # Reset RNG for fair comparison
    d=d,
    rho=rho,
    alpha=alpha,
    beta=beta,
    max_rounds=max_rounds,
    sensitivity_type=sensitivity_type,
    norm_bound=norm_bound,
    optimizer="pgd-maxent",
    allow_remeasurement=False
)

Sigma_hat_2 = ggm_no_remeasure.fit(X)
print(f"\nFinal estimate shape: {Sigma_hat_2.shape}")
print(f"Number of measured entries: {len(ggm_no_remeasure.measured_entries)}")
print(f"Measured entries include diagonal (besides initial): {any(i==j for i,j in ggm_no_remeasure.measured_entries if (i,j) not in [(k,k) for k in range(d)])}")

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"With remeasurement: {len(ggm_remeasure.measured_entries)} entries measured")
print(f"Without remeasurement: {len(ggm_no_remeasure.measured_entries)} entries measured")
print("✓ Test completed successfully!")
