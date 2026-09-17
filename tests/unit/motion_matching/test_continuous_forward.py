"""Continuous integration must preserve absolute time and reject invalid states."""

import numpy as np
import pytest

from src.shared.python.motion_matching.continuous_forward import integrate_forward

pytestmark = pytest.mark.unit


def test_nonconstant_sextic_input_uses_one_initial_state() -> None:
    time = np.array([0.0, 0.3, 0.8])
    initial = np.array([2.0, -0.5])
    result = integrate_forward(
        initial, time, lambda t, x: np.array([x[1], t**6]), max_step=0.02
    )
    expected = np.column_stack((2 - 0.5 * time + time**8 / 56, -0.5 + time**7 / 7))
    np.testing.assert_allclose(result.state, expected, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(initial, [2.0, -0.5])
    assert result.evaluations > 0
    assert not result.state.flags.writeable


@pytest.mark.parametrize("clock", [[0], [0, 0], [0.1, 0.2], [0, np.nan]])
def test_invalid_clock_rejected(clock: list[float]) -> None:
    with pytest.raises(ValueError, match="time"):
        integrate_forward(np.zeros(2), np.array(clock), lambda t, x: x)


@pytest.mark.parametrize("derivative", [np.array([1.0]), np.array([np.nan, 0])])
def test_invalid_derivative_fails(derivative: np.ndarray) -> None:
    with pytest.raises(ValueError, match="derivative"):
        integrate_forward(np.zeros(2), np.array([0.0, 1.0]), lambda t, x: derivative)
