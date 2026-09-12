"""Explicit equality constraints must survive competing marker objectives."""

import numpy as np
import pytest

from src.shared.python.motion_matching.equality_least_squares import (
    solve_equality_least_squares,
)

pytestmark = pytest.mark.unit


def residual(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] - 2, x[1] - x[0], 2 * (x[1] - x[0]), x[1] - 3])


def jacobian(x: np.ndarray) -> np.ndarray:
    return np.array([[1.0, 0.0], [-1.0, 1.0], [-2.0, 2.0], [0.0, 1.0]])


def solve(**kwargs):
    return solve_equality_least_squares(
        residual,
        jacobian,
        np.zeros(2),
        np.zeros(2),
        np.full(2, 4.0),
        equality_start=1,
        projection=np.array([[0.2, 0.4]]),
        max_iterations=20,
        max_evaluations=kwargs.pop("max_evaluations", 30),
        **kwargs,
    )


def test_competing_objective_preserves_projected_equality() -> None:
    result = solve()
    assert result.success
    np.testing.assert_allclose(result.x, [2.5, 2.5], atol=1e-8)
    assert result.fun == pytest.approx(0.5)
    assert result.nfev <= 30


def test_hard_evaluation_budget_returns_unaccepted_evaluated_point() -> None:
    result = solve(max_evaluations=1)
    assert not result.success
    assert result.nfev == 1
    np.testing.assert_array_equal(result.x, [0.0, 0.0])
    assert "evaluation budget" in result.message


@pytest.mark.parametrize(
    "projection",
    [np.array([[np.nan, 0.0]]), np.zeros((1, 2)), np.array([[1.0, 2.0], [2.0, 4.0]])],
)
def test_invalid_or_dependent_projection_is_rejected(projection: np.ndarray) -> None:
    with pytest.raises(ValueError, match="projection"):
        solve_equality_least_squares(
            residual,
            jacobian,
            np.zeros(2),
            np.zeros(2),
            np.full(2, 4.0),
            equality_start=1,
            projection=projection,
            max_iterations=20,
            max_evaluations=30,
        )


def test_objective_and_constraint_jacobians_match_centered_differences(
    monkeypatch,
) -> None:
    from src.shared.python.motion_matching import equality_least_squares as module

    original = module.minimize

    def checked(fun, x0, **kwargs):
        point = np.array([0.7, 1.2])
        step = 1e-5
        fd = np.array(
            [
                (fun(point + step * d) - fun(point - step * d)) / (2 * step)
                for d in np.eye(2)
            ]
        )
        np.testing.assert_allclose(kwargs["jac"](point), fd, atol=1e-8)
        c = kwargs["constraints"]
        cj = np.column_stack(
            [
                (c["fun"](point + step * d) - c["fun"](point - step * d)) / (2 * step)
                for d in np.eye(2)
            ]
        )
        np.testing.assert_allclose(c["jac"](point), cj, atol=1e-8)
        return original(fun, x0, **kwargs)

    monkeypatch.setattr(module, "minimize", checked)
    assert solve().success
