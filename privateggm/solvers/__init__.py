"""privateggm.solvers — covariance estimation solvers.

Working solvers
---------------
pgd          Projected gradient descent (PSD baseline; no max-entropy structure)
pgd_maxent   Two-stage: Stage A PGD + Stage B max-entropy completion
ipm_chol     IPM with Cholesky parameterization X = LL^T  (recommended)

Optional (requires clarabel >= 0.11.1)
---------------------------------------
clarabel_solver  Off-the-shelf conic IPM via Clarabel

Experimental (not production-ready)
-------------------------------------
from privateggm.solvers.experimental import ipm_chol_kspace
"""

from .pgd_maxent import pgd, pgd_maxent
from .ipm_chol import ipm_chol

try:
    from .clarabel import clarabel_solver
except ImportError:
    pass
