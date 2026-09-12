"""Analytic constrained acceleration checks independent of a native installation."""

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.native_model import (
    solve_weld_acceleration,
)

pytestmark = pytest.mark.unit


def test_weld_balances_force_and_acceleration_bias() -> None:
    mass = np.diag([2.0, 3.0])
    force = np.array([4.0, 9.0])
    jacobian = np.array([[1.0, -1.0]])
    bias = np.array([0.5])
    acceleration = solve_weld_acceleration(mass, force, jacobian, bias)
    np.testing.assert_allclose(acceleration, [2.3, 2.8], atol=1e-13)
    np.testing.assert_allclose(jacobian @ acceleration + bias, 0, atol=1e-13)
    np.testing.assert_allclose(
        np.array([1.0, 1.0]) @ (mass @ acceleration - force), 0, atol=1e-13
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_inputs_fail(bad: float) -> None:
    with pytest.raises(ValueError):
        solve_weld_acceleration(
            np.eye(2), np.array([bad, 0]), np.ones((1, 2)), np.zeros(1)
        )


def test_redundant_constraint_fails_instead_of_silent_projection() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        solve_weld_acceleration(np.eye(2), np.zeros(2), np.ones((2, 2)), np.zeros(2))


def test_shape_mismatch_fails() -> None:
    with pytest.raises(ValueError):
        solve_weld_acceleration(np.eye(2), np.zeros(3), np.ones((1, 2)), np.zeros(1))
