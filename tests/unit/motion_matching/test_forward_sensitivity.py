"""Forward sensitivities must follow the full variational equation."""

import numpy as np
import pytest

from src.shared.python.motion_matching.forward_sensitivity import (
    integrate_sensitivities,
)

pytestmark = pytest.mark.unit


def test_polynomial_acceleration_matches_analytic_sensitivity() -> None:
    def linearize(t: float, state: np.ndarray) -> tuple:
        return (
            np.array([state[1], 2 * t * t]),
            np.array([[0, 1], [0, 0]]),
            np.array([[0], [t * t]]),
        )

    result = integrate_sensitivities(
        np.zeros(2), np.array([0.0, 0.3, 0.8]), linearize, 1, max_step=0.02
    )
    t = result.integration.time
    np.testing.assert_allclose(result.integration.state[:, 0], t**4 / 6, atol=1e-12)
    np.testing.assert_allclose(
        result.state_parameter_jacobian[:, 0, 0], t**4 / 12, atol=1e-12
    )
    np.testing.assert_allclose(
        result.state_parameter_jacobian[:, 1, 0], t**3 / 3, atol=1e-12
    )
    assert not result.state_parameter_jacobian.flags.writeable


def test_state_coupling_and_initial_sensitivity_are_preserved() -> None:
    def linearize(t: float, state: np.ndarray) -> tuple:
        return 2 * state, np.array([[2.0]]), state.reshape(1, 1)

    result = integrate_sensitivities(
        np.ones(1),
        np.array([0.0, 0.5]),
        linearize,
        1,
        initial_sensitivity=np.array([[3.0]]),
        max_step=0.02,
    )
    np.testing.assert_allclose(
        result.state_parameter_jacobian[-1, 0, 0], np.exp(1) * 3.5, rtol=1e-10
    )


def test_wrong_linearization_shape_fails() -> None:
    with pytest.raises(ValueError, match="linearization"):
        integrate_sensitivities(
            np.zeros(2),
            np.array([0.0, 1.0]),
            lambda t, x: (x, np.zeros((1, 1)), np.zeros((2, 1))),
            1,
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_parameter_count_fails(count: int) -> None:
    with pytest.raises(ValueError, match="parameter_count"):
        integrate_sensitivities(
            np.ones(1), np.array([0.0, 1.0]), lambda t, x: None, count
        )
