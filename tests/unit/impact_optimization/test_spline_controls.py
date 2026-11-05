"""SHOW-C3 (#14559): spline torque controls."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.spline_controls import SplineTorqueControl

pytestmark = pytest.mark.unit

KNOTS_T = np.linspace(0.0, 1.0, 5)


def _ctrl(n_joints: int = 3) -> SplineTorqueControl:
    return SplineTorqueControl(knot_times_s=KNOTS_T, n_joints=n_joints)


def test_zero_knots_give_zero_torque() -> None:
    c = _ctrl()
    out = c.torque(np.zeros((5, 3)), np.linspace(0.0, 1.0, 40))
    assert out.shape == (40, 3)
    assert np.all(out == 0.0)


def test_interpolates_knot_values_at_knot_times() -> None:
    c = _ctrl()
    k = np.random.default_rng(1).normal(size=(5, 3))
    np.testing.assert_allclose(c.torque(k, KNOTS_T), k, atol=1e-12)


def test_is_linear_in_the_knots() -> None:
    c = _ctrl()
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(2, 5, 3))
    t = np.linspace(0.0, 1.0, 33)
    np.testing.assert_allclose(
        c.torque(2.0 * a + b, t), 2.0 * c.torque(a, t) + c.torque(b, t), atol=1e-12
    )


def test_holds_end_values_outside_the_knot_range() -> None:
    c = _ctrl(1)
    k = np.arange(5.0).reshape(5, 1)
    out = c.torque(k, np.array([-1.0, 2.0]))
    assert out[0, 0] == pytest.approx(0.0) and out[1, 0] == pytest.approx(4.0)


def test_flat_vector_round_trip() -> None:
    c = _ctrl()
    k = np.arange(15.0).reshape(5, 3)
    assert c.n_params == 15
    np.testing.assert_array_equal(c.unpack(k.ravel()), k)


@pytest.mark.parametrize(
    "times", [np.array([0.0]), np.array([0.0, 0.5, 0.5]), np.array([1.0, 0.0])]
)
def test_invalid_knot_times_rejected(times: np.ndarray) -> None:
    with pytest.raises(ContractViolationError):
        SplineTorqueControl(knot_times_s=times, n_joints=2)


def test_wrong_knot_shape_rejected() -> None:
    with pytest.raises(ContractViolationError):
        _ctrl().torque(np.zeros((4, 3)), np.array([0.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_evaluation_times_rejected(bad: float) -> None:
    with pytest.raises(ContractViolationError):
        _ctrl().torque(np.zeros((5, 3)), np.array([0.0, bad]))


def test_non_one_dimensional_times_rejected() -> None:
    with pytest.raises(ContractViolationError):
        _ctrl().torque(np.zeros((5, 3)), np.zeros((2, 2)))
