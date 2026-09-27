"""Unit tests for least-squares parameter uncertainty (Issue #11021)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.fit_uncertainty import (
    ParameterUncertainty,
    fitted_uncertainty_or_none,
    least_squares_parameter_uncertainty,
)

pytestmark = pytest.mark.unit


def test_linear_regression_covariance_matches_closed_form() -> None:
    """Validate covariance against closed-form OLS formula sigma^2 (X^T X)^(-1)."""
    rng = np.random.default_rng(12345)
    x_data = np.linspace(0.5, 2.5, 50)
    noise = rng.normal(0.0, 0.1, size=len(x_data))
    y_data = 2.0 + 3.0 * x_data + noise

    X = np.column_stack([np.ones_like(x_data), x_data])

    def residual(p: np.ndarray) -> np.ndarray:
        return (X @ p) - y_data

    def jacobian(p: np.ndarray) -> np.ndarray:
        return X

    opt = least_squares(residual, x0=np.zeros(2), jac=jacobian)
    unc = least_squares_parameter_uncertainty(opt)

    assert isinstance(unc, ParameterUncertainty)
    assert unc.status == "estimated"
    assert unc.parameter_indices == (0, 1)
    assert unc.n_residuals == 50
    assert unc.n_free == 2
    assert unc.rank == 2
    assert unc.at_bound == ()
    assert unc.condition_number >= 1.0

    m = len(y_data)
    res = X @ opt.x - y_data
    rss = float(np.sum(res**2))
    sigma2_closed = rss / (m - 2)
    cov_closed = sigma2_closed * np.linalg.inv(X.T @ X)
    se_closed = np.sqrt(np.diag(cov_closed))

    assert unc.covariance is not None
    assert unc.standard_errors is not None
    assert unc.correlation is not None

    np.testing.assert_allclose(unc.residual_variance, sigma2_closed, rtol=1e-10)
    np.testing.assert_allclose(unc.covariance, cov_closed, rtol=1e-10)
    np.testing.assert_allclose(unc.standard_errors, se_closed, rtol=1e-10)

    corr_closed = cov_closed / np.outer(se_closed, se_closed)
    np.testing.assert_allclose(unc.correlation, corr_closed, rtol=1e-10)

    # Read-only check
    assert unc.covariance.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        unc.covariance[0, 0] = 999.0


def test_marginal_vs_conditional_block() -> None:
    """Assert theta block equals Schur-complement marginal and differs from conditional."""
    rng = np.random.default_rng(42)
    m = 30
    J_theta = rng.standard_normal((m, 2))
    # Nuisance columns correlated with theta
    J_nuisance = J_theta @ np.array(
        [[1.0, -0.5], [0.5, 1.2]]
    ) + 0.3 * rng.standard_normal((m, 2))
    J = np.column_stack([J_theta, J_nuisance])

    p_true = np.array([1.5, -2.0, 0.5, 1.0])
    y = J @ p_true + rng.normal(0, 0.05, size=m)

    opt = least_squares(lambda p: J @ p - y, x0=np.zeros(4), jac=lambda p: J)

    unc_theta = least_squares_parameter_uncertainty(opt, parameter_indices=(0, 1))
    assert unc_theta.status == "estimated"
    assert unc_theta.parameter_indices == (0, 1)

    # Full Fisher matrix F = J^T J
    F = J.T @ J
    A = F[0:2, 0:2]  # J_theta^T J_theta
    B = F[0:2, 2:4]  # J_theta^T J_nuisance
    D = F[2:4, 2:4]  # J_nuisance^T J_nuisance

    # Schur complement marginal covariance for theta: (A - B D^-1 B^T)^-1 * sigma^2
    schur_inv = np.linalg.inv(A - B @ np.linalg.inv(D) @ B.T)
    marginal_cov_closed = unc_theta.residual_variance * schur_inv

    # Conditional covariance: (J_theta^T J_theta)^-1 * sigma^2
    conditional_cov = unc_theta.residual_variance * np.linalg.inv(A)

    assert unc_theta.covariance is not None
    # 1. Equals Schur-complement marginal
    np.testing.assert_allclose(unc_theta.covariance, marginal_cov_closed, rtol=1e-10)
    # 2. Differs from conditional covariance
    assert not np.allclose(unc_theta.covariance, conditional_cov, rtol=1e-3)


def test_rank_deficiency_duplicate_column() -> None:
    """Duplicate column gives rank_deficient status, covariance None, and rank 1 fewer."""
    rng = np.random.default_rng(77)
    m = 20
    col0 = rng.standard_normal(m)
    col1 = rng.standard_normal(m)
    col2 = col0.copy()  # Duplicate column!
    J = np.column_stack([col0, col1, col2])
    y = rng.standard_normal(m)

    opt = least_squares(lambda p: J @ p - y, x0=np.zeros(3), jac=lambda p: J)
    unc = least_squares_parameter_uncertainty(opt)

    assert unc.status == "rank_deficient"
    assert unc.covariance is None
    assert unc.standard_errors is None
    assert unc.correlation is None
    assert unc.n_free == 3
    assert unc.rank == 2  # exactly 1 fewer than n_free


def test_active_bound_marks_nan_and_partial_at_bounds() -> None:
    """Bound one parameter so optimum sits on it; assert NaN row/col and partial_at_bounds."""
    rng = np.random.default_rng(88)
    m = 25
    X = rng.standard_normal((m, 3))
    y = X @ np.array([2.0, 5.0, -1.0]) + rng.normal(0, 0.05, size=m)

    # Bound parameter 1 to upper bound 0.0 (true value is 5.0)
    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, 0.0, np.inf])
    opt = least_squares(
        lambda p: X @ p - y, x0=np.zeros(3), bounds=bounds, jac=lambda p: X
    )
    assert opt.active_mask[1] != 0  # actively constrained

    unc = least_squares_parameter_uncertainty(opt)
    assert unc.status == "partial_at_bounds"
    assert unc.at_bound == (1,)
    assert unc.covariance is not None
    assert unc.standard_errors is not None
    assert unc.correlation is not None

    # Parameter 1 row and column in covariance are NaN
    assert np.isnan(unc.covariance[1, :]).all()
    assert np.isnan(unc.covariance[:, 1]).all()
    # Free parameters 0 and 2 have finite variance
    assert np.isfinite(unc.covariance[0, 0])
    assert unc.covariance[0, 0] > 0
    assert np.isfinite(unc.covariance[2, 2])
    assert unc.covariance[2, 2] > 0

    # Standard errors: NaN for bound parameter, finite for free
    assert np.isnan(unc.standard_errors[1])
    assert np.isfinite(unc.standard_errors[0])
    assert np.isfinite(unc.standard_errors[2])

    # Correlation: NaN for bound parameter, 1.0 on diagonal for free
    assert np.isnan(unc.correlation[1, :]).all()
    assert np.isnan(unc.correlation[:, 1]).all()
    assert unc.correlation[0, 0] == pytest.approx(1.0)
    assert unc.correlation[2, 2] == pytest.approx(1.0)


def test_underdetermined_m_less_equal_n() -> None:
    """m <= n_free gives underdetermined status with None covariance."""
    J = np.eye(3)
    opt = SimpleNamespace(
        x=np.zeros(3),
        jac=J,
        cost=0.5,
        active_mask=np.zeros(3, dtype=int),
    )
    unc = least_squares_parameter_uncertainty(opt)  # type: ignore[arg-type]
    assert unc.status == "underdetermined"
    assert unc.covariance is None
    assert unc.standard_errors is None
    assert unc.correlation is None
    assert unc.n_residuals == 3
    assert unc.n_free == 3


def test_contracts_enforce_valid_inputs() -> None:
    """Contract assertions raise on non-finite jac, duplicate indices, or invalid rtol."""
    # Non-finite jac
    opt_nan = SimpleNamespace(
        x=np.zeros(2),
        jac=np.array([[1.0, np.nan], [2.0, 3.0]]),
        cost=1.0,
    )
    with pytest.raises(ContractViolationError):
        least_squares_parameter_uncertainty(opt_nan)  # type: ignore[arg-type]

    # Valid base optimum for index and rtol contract tests
    opt_valid = SimpleNamespace(
        x=np.zeros(2),
        jac=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        cost=1.0,
    )
    # Duplicate indices
    with pytest.raises(ContractViolationError):
        least_squares_parameter_uncertainty(
            opt_valid,
            parameter_indices=(0, 0),  # type: ignore[arg-type]
        )

    # Out of range indices
    with pytest.raises(ContractViolationError):
        least_squares_parameter_uncertainty(
            opt_valid,
            parameter_indices=(0, 2),  # type: ignore[arg-type]
        )

    # rank_rtol <= 0
    with pytest.raises(ContractViolationError):
        least_squares_parameter_uncertainty(opt_valid, rank_rtol=0.0)  # type: ignore[arg-type]


def test_fit_report_gets_none_for_a_non_finite_jacobian() -> None:
    """A diverged fit still reports; its uncertainty is None, not a crash."""
    opt_nan = SimpleNamespace(
        x=np.zeros(2),
        jac=np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]]),
        cost=1.0,
    )
    assert fitted_uncertainty_or_none(opt_nan) is None  # type: ignore[arg-type]
    assert fitted_uncertainty_or_none(SimpleNamespace(x=np.zeros(1))) is None  # type: ignore[arg-type]


def test_fit_report_helper_matches_the_direct_computation() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 2))
    y = X @ np.array([1.0, -1.0]) + rng.normal(0, 0.1, size=20)
    opt = least_squares(lambda p: X @ p - y, x0=np.zeros(2), jac=lambda p: X)

    via_helper = fitted_uncertainty_or_none(opt, parameter_indices=(1,))
    direct = least_squares_parameter_uncertainty(opt, parameter_indices=(1,))

    assert via_helper is not None
    assert via_helper.covariance is not None and direct.covariance is not None
    np.testing.assert_array_equal(via_helper.covariance, direct.covariance)
