"""Static feasibility must retain closure instead of fitting an open tree."""

import numpy as np
import pytest

from src.shared.python.motion_matching.constrained_marker_pose import (
    MarkerPoseResult,
    fit_marker_pose,
)

pytestmark = pytest.mark.unit


def test_marker_fit_retains_constraint_when_unconstrained_target_is_better() -> None:
    initial = np.array([0.2, 0.8])
    result = fit_marker_pose(
        initial,
        np.zeros(2),
        np.ones(2),
        np.array([[1.0, 1.0, 0.0]]),
        np.array([True]),
        lambda q: np.array([[q[0], q[1], 0]]),
        lambda q: np.array([q.sum() - 1]),
    )
    assert result.optimizer_converged
    assert result.closure_satisfied
    np.testing.assert_allclose(result.coordinates, [0.5, 0.5], atol=1e-5)
    assert result.marker_rms_m == pytest.approx(np.sqrt(0.5), abs=1e-8)
    np.testing.assert_array_equal(initial, [0.2, 0.8])
    assert not result.coordinates.flags.writeable


def test_missing_markers_are_masked_without_weakening_closure() -> None:
    result = fit_marker_pose(
        np.array([0.2]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([[0.5, 0, 0], [np.nan] * 3]),
        np.array([True, False]),
        lambda q: np.array([[q[0], 0, 0], [100, 0, 0]]),
        lambda q: np.array([q[0] - 0.5]),
    )
    assert result.closure_satisfied
    assert result.marker_rms_m < 1e-7


def test_failed_oracle_is_not_a_finite_penalty() -> None:
    with pytest.raises(ValueError):
        fit_marker_pose(
            np.array([0.2]),
            np.array([0.0]),
            np.array([1.0]),
            np.zeros((1, 3)),
            np.array([True]),
            lambda q: np.full((1, 3), np.nan),
            lambda q: np.array([q[0]]),
        )


@pytest.mark.parametrize("invalid", [False, True])
def test_supplied_derivatives_preserve_masked_constrained_solution(
    invalid: bool,
) -> None:
    calls = []

    def marker_jacobian(q: np.ndarray) -> np.ndarray:
        calls.append("marker")
        return (
            np.full((2, 3, 2), np.nan)
            if invalid
            else np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0]] * 3])
        )

    def closure_jacobian(q: np.ndarray) -> np.ndarray:
        calls.append("closure")
        return np.ones((1, 2))

    def solve() -> MarkerPoseResult:
        return fit_marker_pose(
            np.array([0.2, 0.8]),
            np.zeros(2),
            np.ones(2),
            np.array([[1.0, 1.0, 0.0], [np.nan] * 3]),
            np.array([True, False]),
            lambda q: np.array([[q[0], q[1], 0.0], [100.0, 0.0, 0.0]]),
            lambda q: np.array([q.sum() - 1]),
            forward_jacobian=marker_jacobian,
            closure_jacobian=closure_jacobian,
        )

    if invalid:
        with pytest.raises(ValueError, match="Jacobian"):
            solve()
    else:
        result = solve()
        assert result.closure_satisfied and result.optimizer_converged
        np.testing.assert_allclose(result.coordinates, [0.5, 0.5], atol=1e-7)
        assert set(calls) == {"marker", "closure"}
