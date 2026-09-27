"""Unit tests for sparse residual discovery (STLSQ) and physics-structured surrogate integration (issue #11024)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.neural_motion.surrogates import (
    PhysicsStructuredSurrogate,
)
from src.shared.python.neural_motion.surrogates.sparse_residual import (
    CandidateLibrary,
    SparseResidualFit,
    fit_sparse_residual,
)

pytestmark = pytest.mark.unit


def test_candidate_library_feature_names_and_order_degree2() -> None:
    """CandidateLibrary produces deterministic monomial and trig feature names in order."""
    # 2 states at degree 2, with bias, without trig
    lib = CandidateLibrary(degree=2, include_bias=True, include_trig=False)
    names = lib.feature_names(["x", "y"])
    assert names == ("1", "x", "y", "x^2", "x y", "y^2")

    # Without bias
    lib_no_bias = CandidateLibrary(degree=2, include_bias=False, include_trig=False)
    assert lib_no_bias.feature_names(["x", "y"]) == ("x", "y", "x^2", "x y", "y^2")

    # With trig: bias, degree 1, degree 2, then sin/cos for each state in order
    lib_trig = CandidateLibrary(degree=2, include_bias=True, include_trig=True)
    expected_trig = (
        "1",
        "x",
        "y",
        "x^2",
        "x y",
        "y^2",
        "sin(x)",
        "cos(x)",
        "sin(y)",
        "cos(y)",
    )
    assert lib_trig.feature_names(["x", "y"]) == expected_trig

    # Test transform numerical output
    sample = np.array([[2.0, 3.0]])
    phi = lib_trig.transform(sample)
    expected_values = np.array(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                6.0,
                9.0,
                np.sin(2.0),
                np.cos(2.0),
                np.sin(3.0),
                np.cos(3.0),
            ]
        ]
    )
    assert phi.shape == (1, 10)
    np.testing.assert_allclose(phi, expected_values, rtol=1e-12)


def test_damped_pendulum_sparse_residual() -> None:
    """Damped pendulum: STLSQ recovers x' and (-9.81 sin x - 0.3 x') within 1% tolerance."""

    def pendulum_rhs(_t: float, y: np.ndarray) -> list[float]:
        # y = [x, x_dot]
        return [float(y[1]), float(-9.81 * np.sin(y[0]) - 0.3 * y[1])]

    sol = solve_ivp(
        pendulum_rhs,
        t_span=(0.0, 10.0),
        y0=[1.0, 0.0],
        t_eval=np.linspace(0.0, 10.0, 500),
        rtol=1e-10,
        atol=1e-12,
    )
    states = sol.y.T  # (500, 2)
    x = states[:, 0]
    x_dot = states[:, 1]
    x_ddot = -9.81 * np.sin(x) - 0.3 * x_dot
    targets = np.column_stack([x_dot, x_ddot])  # (500, 2)

    library = CandidateLibrary(degree=1, include_trig=True, include_bias=True)
    fit = fit_sparse_residual(
        states,
        targets,
        library=library,
        threshold=0.05,
        state_names=["x", "x'"],
    )

    assert fit.converged
    assert fit.coefficients.shape == (len(fit.feature_names), 2)
    assert fit.active.shape == (len(fit.feature_names), 2)

    # Active terms: exactly x' for first target
    active_t0 = [
        name
        for name, act in zip(fit.feature_names, fit.active[:, 0], strict=True)
        if act
    ]
    assert active_t0 == ["x'"]

    # Active terms: exactly sin(x) and x' for second target
    active_t1 = [
        name
        for name, act in zip(fit.feature_names, fit.active[:, 1], strict=True)
        if act
    ]
    assert set(active_t1) == {"sin(x)", "x'"}

    # Coefficients within 1% relative of (1, -9.81, -0.3)
    idx_xdot = fit.feature_names.index("x'")
    idx_sinx = fit.feature_names.index("sin(x)")

    c_xdot_t0 = fit.coefficients[idx_xdot, 0]
    c_sinx_t1 = fit.coefficients[idx_sinx, 1]
    c_xdot_t1 = fit.coefficients[idx_xdot, 1]

    assert np.isclose(c_xdot_t0, 1.0, rtol=0.01)
    assert np.isclose(c_sinx_t1, -9.81, rtol=0.01)
    assert np.isclose(c_xdot_t1, -0.3, rtol=0.01)

    # Inactive coefficients must be zero
    assert np.all(fit.coefficients[~fit.active] == 0.0)


def test_lorenz_sparse_residual() -> None:
    """Lorenz system with 1e-3 noise: STLSQ recovers exact 7-term support within 2%."""

    def lorenz_rhs(_t: float, y: np.ndarray) -> list[float]:
        x_val, y_val, z_val = y
        return [
            10.0 * (y_val - x_val),
            x_val * (28.0 - z_val) - y_val,
            x_val * y_val - (8.0 / 3.0) * z_val,
        ]

    sol = solve_ivp(
        lorenz_rhs,
        t_span=(0.0, 20.0),
        y0=[-8.0, 8.0, 27.0],
        t_eval=np.linspace(0.0, 20.0, 2000),
        rtol=1e-10,
        atol=1e-12,
    )
    states = sol.y.T  # (2000, 3)
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    dx = 10.0 * (y - x)
    dy = 28.0 * x - y - x * z
    dz = x * y - (8.0 / 3.0) * z
    true_derivs = np.column_stack([dx, dy, dz])

    rng = np.random.default_rng(42)
    noisy_targets = true_derivs + rng.normal(0.0, 1e-3, size=true_derivs.shape)

    library = CandidateLibrary(degree=2, include_bias=True, include_trig=False)
    fit = fit_sparse_residual(
        states,
        noisy_targets,
        library=library,
        threshold=0.5,
        state_names=["x", "y", "z"],
    )

    assert fit.converged
    # Assert exact 7-term support across all 3 targets
    assert int(np.sum(fit.active)) == 7

    # Target 0: x (-10), y (+10)
    active_t0 = [
        name
        for name, act in zip(fit.feature_names, fit.active[:, 0], strict=True)
        if act
    ]
    assert set(active_t0) == {"x", "y"}
    assert np.isclose(
        fit.coefficients[fit.feature_names.index("x"), 0], -10.0, rtol=0.02
    )
    assert np.isclose(
        fit.coefficients[fit.feature_names.index("y"), 0], 10.0, rtol=0.02
    )

    # Target 1: x (+28), y (-1), x z (-1)
    active_t1 = [
        name
        for name, act in zip(fit.feature_names, fit.active[:, 1], strict=True)
        if act
    ]
    assert set(active_t1) == {"x", "y", "x z"}
    assert np.isclose(
        fit.coefficients[fit.feature_names.index("x"), 1], 28.0, rtol=0.02
    )
    assert np.isclose(
        fit.coefficients[fit.feature_names.index("y"), 1], -1.0, rtol=0.02
    )
    assert np.isclose(
        fit.coefficients[fit.feature_names.index("x z"), 1], -1.0, rtol=0.02
    )

    # Target 2: z (-8/3), x y (+1)
    active_t2 = [
        name
        for name, act in zip(fit.feature_names, fit.active[:, 2], strict=True)
        if act
    ]
    assert set(active_t2) == {"z", "x y"}
    assert np.isclose(
        fit.coefficients[fit.feature_names.index("z"), 2], -8.0 / 3.0, rtol=0.02
    )
    assert np.isclose(
        fit.coefficients[fit.feature_names.index("x y"), 2], 1.0, rtol=0.02
    )


def test_threshold_zero_equals_lstsq_and_huge_threshold_empty() -> None:
    """Threshold 0 equals np.linalg.lstsq; huge threshold zeroes all terms."""
    rng = np.random.default_rng(123)
    states = rng.normal(size=(50, 2))
    targets = rng.normal(size=(50, 2))
    library = CandidateLibrary(degree=2, include_bias=True, include_trig=False)

    # Threshold 0.0 equals ordinary least squares
    fit_zero = fit_sparse_residual(states, targets, library=library, threshold=0.0)
    phi = library.transform(states)
    expected_lstsq, _, _, _ = np.linalg.lstsq(phi, targets, rcond=None)

    np.testing.assert_allclose(fit_zero.coefficients, expected_lstsq, rtol=1e-10)
    assert np.all(fit_zero.active)
    assert fit_zero.converged

    # Huge threshold zeroes everything
    fit_huge = fit_sparse_residual(states, targets, library=library, threshold=1e9)
    assert not np.any(fit_huge.active)
    np.testing.assert_allclose(fit_huge.coefficients, 0.0)
    assert fit_huge.converged


def test_precondition_violations_raise() -> None:
    """DbC violations: NaN, mismatched rows, underdetermined system, bad parameters."""
    library = CandidateLibrary(degree=2, include_bias=True)
    valid_state = np.ones((10, 2))
    valid_target = np.ones((10, 2))

    # NaN in state
    nan_state = valid_state.copy()
    nan_state[0, 0] = np.nan
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(nan_state, valid_target, library=library, threshold=0.1)

    # NaN in target
    nan_target = valid_target.copy()
    nan_target[0, 0] = np.nan
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(valid_state, nan_target, library=library, threshold=0.1)

    # Mismatched rows
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(
            valid_state[:5], valid_target, library=library, threshold=0.1
        )

    # Underdetermined fit (rows < features): degree 2 with 2 states has 6 features
    underdetermined_state = np.ones((4, 2))
    underdetermined_target = np.ones((4, 2))
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(
            underdetermined_state,
            underdetermined_target,
            library=library,
            threshold=0.1,
        )

    # Negative threshold
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(valid_state, valid_target, library=library, threshold=-0.01)

    # Negative ridge
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(
            valid_state, valid_target, library=library, threshold=0.1, ridge=-1.0
        )

    # max_iter < 1
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(
            valid_state, valid_target, library=library, threshold=0.1, max_iter=0
        )

    # 1-D state
    with pytest.raises((PreconditionError, ValueError)):
        fit_sparse_residual(
            valid_state[:, 0], valid_target, library=library, threshold=0.1
        )


def test_physics_structured_surrogate_without_residual_raises() -> None:
    """PhysicsStructuredSurrogate without residual raises NotImplementedError naming #11007."""
    surrogate = PhysicsStructuredSurrogate(n_dof=2, n_coeffs=7)
    coeffs = np.zeros(14)
    timegrid = np.linspace(0.0, 0.3, 30)

    prior = surrogate.analytical_prior(coeffs, timegrid)
    assert prior.shape == (30, 2)
    assert np.all(np.isfinite(prior))

    with pytest.raises(NotImplementedError, match="11007"):
        surrogate.residual_correction(prior)

    with pytest.raises(NotImplementedError, match="11007"):
        surrogate.forward_trajectory(coeffs, timegrid)


def test_physics_structured_surrogate_with_fitted_residual() -> None:
    """PhysicsStructuredSurrogate with fitted residual predicts exact residual correction."""
    library = CandidateLibrary(degree=1, include_bias=True, include_trig=False)
    # Fit simple linear residual for 2 DOF
    x_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    fit = fit_sparse_residual(x_train, y_train, library=library, threshold=0.01)

    surrogate = PhysicsStructuredSurrogate(n_dof=2, n_coeffs=7, residual=fit)
    coeffs = np.zeros(14)
    timegrid = np.linspace(0.0, 0.3, 30)

    prior = surrogate.analytical_prior(coeffs, timegrid)
    correction = surrogate.residual_correction(prior)
    expected_correction = fit.predict(prior)
    np.testing.assert_allclose(correction, expected_correction)

    traj = surrogate.forward_trajectory(coeffs, timegrid)
    np.testing.assert_allclose(traj, prior + expected_correction)


def test_sparse_residual_fit_properties_and_predict() -> None:
    """SparseResidualFit fields are read-only and predict verifies shape and values."""
    library = CandidateLibrary(degree=1, include_bias=True)
    states = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    targets = np.array([[0.5], [1.5], [2.5]])
    fit = fit_sparse_residual(states, targets, library=library, threshold=0.01)

    # Read-only coefficients and active masks
    assert not fit.coefficients.flags.writeable
    assert not fit.active.flags.writeable
    with pytest.raises(ValueError):
        fit.coefficients[0, 0] = 999.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        fit.active[0, 0] = False  # type: ignore[misc]

    # Predict method
    pred = fit.predict(states)
    assert pred.shape == (3, 1)
    expected_rmse = float(np.sqrt(np.mean((targets - pred) ** 2)))
    assert np.isclose(fit.training_rmse, expected_rmse, rtol=1e-10)
