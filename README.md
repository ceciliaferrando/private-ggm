# PrivateGGM

**Created:** January 9, 2026 at 2:29 PM

This backup contains all the essential files needed to run the iterative private covariance experiments.

## Contents

### privateggm/ (6 files)
- `__init__.py` - Package initialization
- `model.py` - **CRITICAL**: Contains IterativePrivateGGM with covariance centering fix (line 385-392)
- `driver.py` - Contains run_iterative_mode() function
- `privacy.py` - SSP_covariance and sensitivity functions
- `utils.py` - Error metrics (Mahalanobis, Frobenius)
- `data.py` - Data loading and generation functions

### experiments/ (1 file)
- `test_iterative.py` - Main experiment runner script

### Documentation
- TO-DO

## Critical Fix Included

The `model.py` file includes the **covariance centering fix** applied on Jan 9, 2026:

```python
# Line 385-392 in model.py
X_centered = X - X.mean(axis=0)
self.empirical_cov = (X_centered.T @ X_centered) / n
```

This fix resolves the bug where non-centered data (like Adult, SeoulBike, LifeExpectancy) 
was causing flat error curves because the code was computing E[XX^T] instead of 
Cov(X) = E[(X-μ)(X-μ)^T].


## Usage

To restore or deploy these files:

```bash
# Copy back to main directory
cp -r backup/privateggm/* privateggm/
cp backup/experiments/test_iterative.py experiments/

# Or deploy to server
rsync -av backup/ user@server:/path/to/private-ggm/
```

## File Sizes
- Total: ~68 KB of Python code
- model.py: 18 KB (largest file, contains IterativePrivateGGM class)
- utils.py: 14 KB (error metrics and utilities)
- test_iterative.py: 14 KB (experiment runner)
- data.py: 11 KB (data loading/generation)
- driver.py: 8 KB (run_iterative_mode function)
- privacy.py: 5 KB (SSP and sensitivity calculations)
