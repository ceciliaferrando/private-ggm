"""privateggm.solvers — covariance estimation solvers.

Working solvers
---------------
pgd              Projected gradient descent (PSD baseline; no max-entropy structure)
pgd_maxent       Two-stage: Stage A PGD + Stage B max-entropy completion
ipm_chol         IPM with Cholesky parameterization X = LL^T  (recommended)
ipm_chol_sparse  IPM with sparse Cholesky via junction tree  (better for large sparse problems)
dr_splitting     Douglas-Rachford splitting for sparse PSD-nearness  (active development)

Optional (requires clarabel >= 0.11.1)
---------------------------------------
clarabel_solver  Off-the-shelf conic IPM via Clarabel
"""

from .pgd_maxent import pgd, pgd_maxent
from .ipm_chol import ipm_chol
from .ipm_chol_sparse import ipm_chol_sparse
from .dr_splitting import dr_splitting

try:
    from .clarabel import clarabel_solver
except ImportError:
    pass
