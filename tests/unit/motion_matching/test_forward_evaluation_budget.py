"""Bound actual forward/linearization calls without returning partial success."""

import numpy as np
import pytest

from src.shared.python.motion_matching.continuous_forward import integrate_forward
from src.shared.python.motion_matching.forward_sensitivity import (
    integrate_sensitivities,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("augmented", [False, True])
def test_budget_stops_before_excess_callback(augmented: bool) -> None:
    calls = 0

    def derivative(
        t: float, state: np.ndarray
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        if augmented:
            return state, np.eye(1), np.ones((1, 1))
        return state

    with pytest.raises(RuntimeError, match="evaluation budget.*7"):
        if augmented:
            integrate_sensitivities(
                np.ones(1), np.array([0.0, 1.0]), derivative, 1, max_evaluations=7
            )
        else:
            integrate_forward(
                np.ones(1), np.array([0.0, 1.0]), derivative, max_evaluations=7
            )
    assert calls == 7


@pytest.mark.parametrize("budget", [0, -1, True, 2.5])
def test_invalid_budget_rejected_before_work(budget: int | float) -> None:
    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        pytest.fail("Invalid budget reached dynamics")

    with pytest.raises(ValueError, match="max_evaluations"):
        integrate_forward(
            np.ones(1), np.array([0.0, 1.0]), derivative, max_evaluations=budget
        )


def test_sufficient_budget_preserves_manufactured_solution() -> None:
    result = integrate_forward(
        np.array([2.0]),
        np.array([0.0, 0.1]),
        lambda t, x: np.array([3.0]),
        max_step=0.1,
        max_evaluations=100,
    )
    np.testing.assert_allclose(result.state[:, 0], [2.0, 2.3], atol=1e-12)
    assert result.evaluations <= 100
