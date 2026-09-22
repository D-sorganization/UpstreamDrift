"""Contracts for shared continuous Bernstein actuator controls."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.bernstein_controls import (
    BernsteinFitResidualPolicy,
    bernstein_curvature_penalty,
    evaluate_bernstein_controls,
    solve_bounded_bernstein_least_squares,
)


pytestmark = pytest.mark.unit


def test_controls_interpolate_endpoints_and_stay_within_control_bounds() -> None:
    """The partition of unity preserves endpoints and componentwise bounds."""
    controls = np.array(
        [
            [-4.0, -1.0, 3.0, 8.0],
            [5.0, 2.0, -2.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        evaluate_bernstein_controls(controls, 0.0), controls[:, 0]
    )
    np.testing.assert_allclose(
        evaluate_bernstein_controls(controls, 1.0), controls[:, -1]
    )

    values = np.vstack(
        [
            evaluate_bernstein_controls(controls, time)
            for time in np.linspace(0.0, 1.0, 101)
        ]
    )
    assert np.all(values >= controls.min(axis=1))
    assert np.all(values <= controls.max(axis=1))


def test_controls_clamp_outside_normalized_horizon() -> None:
    controls = np.array([[1.0, 3.0], [-2.0, 4.0]])

    np.testing.assert_allclose(
        evaluate_bernstein_controls(controls, -0.01), controls[:, 0]
    )
    np.testing.assert_allclose(
        evaluate_bernstein_controls(controls, 2.0), controls[:, -1]
    )


@pytest.mark.parametrize(
    ("controls", "normalized_time", "message"),
    [
        (np.array([1.0, 2.0]), 0.0, "two-dimensional"),
        (np.ones((0, 2)), 0.0, "at least one actuator"),
        (np.ones((1, 1)), 0.0, "at least two control points"),
        (np.array([[np.nan, 1.0]]), 0.0, "finite"),
        (np.ones((1, 2)), np.nan, "normalized_time"),
    ],
)
def test_controls_reject_invalid_inputs(
    controls: np.ndarray, normalized_time: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_bernstein_controls(controls, normalized_time)


def test_curvature_penalty_is_flattened_by_actuator_and_rejects_bad_weight() -> None:
    controls = np.array([[0.0, 1.0, 4.0], [3.0, 3.0, 3.0]])

    np.testing.assert_allclose(
        bernstein_curvature_penalty(controls, weight=4.0), np.array([4.0, 0.0])
    )
    with pytest.raises(ValueError, match="weight"):
        bernstein_curvature_penalty(controls, weight=-0.01)


def test_fit_residual_uses_shared_tracking_curvature_and_effort_order() -> None:
    """All bounded fitters append the same regularization convention."""
    policy = BernsteinFitResidualPolicy(
        curvature_weight=4.0,
        effort_weight=1.0,
        effort_scale=2.0,
    )
    result = policy.assemble(
        np.array([1.0, -2.0]),
        np.array([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]]),
    )
    expected = np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result, expected)


def test_bounded_fit_solver_uses_declared_bounds_and_minimum_budget() -> None:
    """All Bernstein fitters share finite bounds and convergence tolerances."""
    result = solve_bounded_bernstein_least_squares(
        lambda parameters: parameters - np.array([0.5]),
        np.array([0.0]),
        np.array([-1.0]),
        np.array([1.0]),
        max_evaluations=1,
    )
    np.testing.assert_allclose(result.x, np.array([0.5]), atol=1e-5)
