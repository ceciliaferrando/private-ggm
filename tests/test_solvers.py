"""Tests for solvers and shared utilities.

Run with:  pytest tests/test_solvers.py -v
"""

import numpy as np
import pytest

from privateggm import solve, dense_to_coo
from privateggm.solvers._shared import normalize_coo
from privateggm.solvers.ipm_chol import ipm_chol
from privateggm.solvers.ipm_chol_sparse import ipm_chol_sparse
from privateggm.solvers.pgd_maxent import pgd, pgd_maxent
from privateggm.solvers.dr_splitting import dr_splitting


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_problem(d=12, seed=0, tau_diag=0.05, fraction_off=0.3):
    """Small low-noise problem where solvers should fit well."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = Q @ np.diag(np.clip(np.exp(rng.standard_normal(d)), 0.5, 5.0)) @ Q.T

    n_off = max(1, int(fraction_off * d * (d - 1) / 2))
    pool = [(i, j) for i in range(d) for j in range(i + 1, d)]
    chosen = rng.choice(len(pool), n_off, replace=False)
    off_ij = [pool[k] for k in sorted(chosen)]

    pairs = [(i, i) for i in range(d)] + off_ij
    tau = np.array([tau_diag] * d + [tau_diag * 2] * n_off)
    rows = np.array([i for i, j in pairs])
    cols = np.array([j for i, j in pairs])
    vals = Sigma[rows, cols] + rng.normal(0, tau)

    return Sigma, rows, cols, vals, tau, d


# ── normalize_coo ─────────────────────────────────────────────────────────────

class TestNormalizeCoo:
    def test_upper_tri_passthrough(self):
        rows = np.array([0, 1, 0])
        cols = np.array([0, 1, 1])
        vals = np.array([1.0, 2.0, 0.5])
        tau  = np.array([0.1, 0.1, 0.2])
        r, c, v, t = normalize_coo(rows, cols, vals, tau)
        assert np.all(r <= c)
        np.testing.assert_array_equal(r, [0, 0, 1])
        np.testing.assert_array_equal(c, [0, 1, 1])
        np.testing.assert_allclose(v, [1.0, 0.5, 2.0])

    def test_lower_tri_swapped(self):
        # (1,0) should become (0,1)
        rows = np.array([1, 0])
        cols = np.array([0, 0])
        vals = np.array([0.3, 1.0])
        tau  = np.array([0.2, 0.1])
        r, c, v, t = normalize_coo(rows, cols, vals, tau)
        assert np.all(r <= c)

    def test_duplicate_combined(self):
        # Two measurements of (0,0): vals 1.0 and 3.0, tau 1.0 and 1.0
        # weights equal -> combined = (1*1.0 + 1*3.0) / 2 = 2.0, tau = 1/sqrt(2)
        rows = np.array([0, 0])
        cols = np.array([0, 0])
        vals = np.array([1.0, 3.0])
        tau  = np.array([1.0, 1.0])
        r, c, v, t = normalize_coo(rows, cols, vals, tau)
        assert len(r) == 1
        np.testing.assert_allclose(v[0], 2.0)
        np.testing.assert_allclose(t[0], 1.0 / np.sqrt(2))

    def test_symmetric_pair_combined(self):
        # (0,1) and (1,0) are same entry — should combine
        rows = np.array([0, 1])
        cols = np.array([1, 0])
        vals = np.array([0.5, 0.5])
        tau  = np.array([0.1, 0.1])
        r, c, v, t = normalize_coo(rows, cols, vals, tau)
        assert len(r) == 1
        assert r[0] == 0 and c[0] == 1
        np.testing.assert_allclose(v[0], 0.5)

    def test_inverse_variance_weighting(self):
        # One precise measurement (tau=0.1) and one imprecise (tau=1.0)
        # w1 = 100, w2 = 1 → combined ≈ (100*2.0 + 1*0.0) / 101 ≈ 1.98
        rows = np.array([0, 0])
        cols = np.array([1, 1])
        vals = np.array([2.0, 0.0])
        tau  = np.array([0.1, 1.0])
        r, c, v, t = normalize_coo(rows, cols, vals, tau)
        assert len(r) == 1
        expected = (100 * 2.0 + 1 * 0.0) / 101
        np.testing.assert_allclose(v[0], expected, rtol=1e-10)

    def test_idempotent(self):
        rows = np.array([1, 0, 0, 2])
        cols = np.array([0, 1, 0, 1])
        vals = np.array([0.5, 0.3, 1.0, 0.8])
        tau  = np.array([0.1, 0.2, 0.15, 0.3])
        r1, c1, v1, t1 = normalize_coo(rows, cols, vals, tau)
        r2, c2, v2, t2 = normalize_coo(r1, c1, v1, t1)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_allclose(v1, v2)
        np.testing.assert_allclose(t1, t2)


# ── dense_to_coo ──────────────────────────────────────────────────────────────

class TestDenseToCoo:
    def test_extracts_values(self):
        A = np.array([[1.0, 0.3], [0.3, 2.0]])
        inds = np.array([[0, 0], [1, 1], [0, 1]])
        tau  = np.array([0.1, 0.1, 0.2])
        rows, cols, vals, tau_out, d = dense_to_coo(A, inds, tau)
        np.testing.assert_array_equal(rows, [0, 1, 0])
        np.testing.assert_array_equal(cols, [0, 1, 1])
        np.testing.assert_allclose(vals, [1.0, 2.0, 0.3])
        assert tau_out is tau
        assert d == 2

    def test_roundtrip_with_solve(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=8)
        A = np.zeros((d, d))
        A[rows, cols] = vals
        A[cols, rows] = vals
        inds = np.stack([rows, cols], axis=1)
        rows2, cols2, vals2, tau2, d2 = dense_to_coo(A, inds, tau)
        S, hist = solve(rows2, cols2, vals2, tau2, d2)
        assert S.shape == (d, d)
        assert hist[-1]['obj'] < 1.0


# ── Solver correctness ────────────────────────────────────────────────────────

def _check_solution(S, Sigma_true, d, obj, frob_tol=3.0, obj_tol=1.0):
    """Common assertions for a solved covariance."""
    assert S.shape == (d, d)
    # Symmetric
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    # PSD
    min_eig = np.linalg.eigvalsh(S).min()
    assert min_eig > -1e-6, f"min eigenvalue {min_eig:.2e}"
    # Low objective on low-noise problems
    assert obj < obj_tol, f"objective {obj:.2e} >= {obj_tol}"
    # Close to ground truth
    frob = np.linalg.norm(S - Sigma_true, 'fro')
    assert frob < frob_tol, f"Frobenius error {frob:.2f} >= {frob_tol}"


class TestIpmChol:
    def test_low_noise(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=12)
        S, hist = ipm_chol(rows, cols, vals, tau, d)
        _check_solution(S, Sigma, d, hist[-1]['obj'])

    def test_hist_fields(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=8)
        _, hist = ipm_chol(rows, cols, vals, tau, d)
        assert len(hist) > 0
        assert all(k in hist[-1] for k in ('time', 'obj', 'entropy', 'mu'))

    def test_accepts_lower_tri(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=8)
        # Swap all off-diagonal entries to lower triangle
        swap = rows != cols
        rows_l = np.where(swap, cols, rows)
        cols_l = np.where(swap, rows, cols)
        S1, _ = ipm_chol(rows, cols, vals, tau, d)
        S2, _ = ipm_chol(rows_l, cols_l, vals, tau, d)
        np.testing.assert_allclose(S1, S2, atol=1e-6)


class TestIpmCholSparse:
    def test_low_noise(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=12)
        S, hist = ipm_chol_sparse(rows, cols, vals, tau, d)
        _check_solution(S, Sigma, d, hist[-1]['obj'])

    def test_matches_dense_at_observed(self):
        # Both solvers minimize the same objective, so their fitted values at
        # observed entries should agree closely.  Unobserved entries can differ
        # because they impose different sparsity on the precision matrix.
        Sigma, rows, cols, vals, tau, d = make_problem(d=12)
        S_dense,  _ = ipm_chol(rows, cols, vals, tau, d)
        S_sparse, _ = ipm_chol_sparse(rows, cols, vals, tau, d)
        np.testing.assert_allclose(S_dense[rows, cols], S_sparse[rows, cols], atol=1e-3)


class TestPgd:
    def test_low_noise(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=12)
        S, hist = pgd(rows, cols, vals, tau, d, max_iters=500)
        assert S.shape == (d, d)
        assert np.linalg.eigvalsh(S).min() > -1e-6

    def test_hist_fields(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=8)
        _, hist = pgd(rows, cols, vals, tau, d, max_iters=10)
        assert all(k in hist[-1] for k in ('time', 'obj', 'entropy'))


class TestPgdMaxent:
    def test_low_noise(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=12)
        S, hist = pgd_maxent(rows, cols, vals, tau, d)
        _check_solution(S, Sigma, d, hist[-1]['obj'])


class TestDrSplitting:
    def test_low_noise(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=12)
        S, hist = dr_splitting(rows, cols, vals, tau, d, max_iters=2000)
        _check_solution(S, Sigma, d, hist[-1]['obj'])

    def test_hist_fields(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=8)
        _, hist = dr_splitting(rows, cols, vals, tau, d, max_iters=50, diag_every=25)
        assert all(k in hist[-1] for k in ('time', 'obj', 'entropy', 'r_primal', 'r_dual'))


# ── solve() dispatch ──────────────────────────────────────────────────────────

class TestSolve:
    @pytest.mark.parametrize('solver', ['ipm_chol', 'ipm_chol_sparse', 'pgd_maxent', 'dr_splitting'])
    def test_solver_names(self, solver):
        Sigma, rows, cols, vals, tau, d = make_problem(d=10)
        S, hist = solve(rows, cols, vals, tau, d, solver=solver)
        assert S.shape == (d, d)
        assert np.linalg.eigvalsh(S).min() > -1e-6

    def test_callable_solver(self):
        Sigma, rows, cols, vals, tau, d = make_problem(d=10)
        S, hist = solve(rows, cols, vals, tau, d, solver=ipm_chol)
        assert S.shape == (d, d)

    def test_decompose_blocks(self):
        # Two disconnected components: nodes 0-3 and 4-7
        d = 8
        rng = np.random.default_rng(1)
        Sigma = np.eye(d) + 0.1 * rng.standard_normal((d, d))
        Sigma = Sigma @ Sigma.T / d

        rows = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 4])
        cols = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 5])
        vals = Sigma[rows, cols]
        tau  = np.full(len(rows), 0.05)

        S, hist = solve(rows, cols, vals, tau, d, decompose='blocks')
        assert S.shape == (d, d)
        # Cross-block entries should be zero (independent components)
        assert S[0, 4] == 0.0

    def test_flexible_input_lower_tri(self):
        # Lower-triangle input should give same result as upper
        Sigma, rows_u, cols_u, vals, tau, d = make_problem(d=10)
        swap = rows_u != cols_u
        rows_l = np.where(swap, cols_u, rows_u)
        cols_l = np.where(swap, rows_u, cols_u)
        S_u, _ = solve(rows_u, cols_u, vals, tau, d)
        S_l, _ = solve(rows_l, cols_l, vals, tau, d)
        np.testing.assert_allclose(S_u, S_l, atol=1e-8)

    def test_duplicate_measurements_combined(self):
        # Two identical measurements of the same entry should give the same
        # result as one measurement with half the noise variance (tau/sqrt(2))
        Sigma, rows, cols, vals, tau, d = make_problem(d=8)
        rows2 = np.concatenate([rows, rows])
        cols2 = np.concatenate([cols, cols])
        vals2 = np.concatenate([vals, vals])
        tau2  = np.concatenate([tau,  tau])

        S1, _ = solve(rows, cols, vals, tau / np.sqrt(2), d)
        S2, _ = solve(rows2, cols2, vals2, tau2, d)
        np.testing.assert_allclose(S1, S2, atol=1e-6)
