"""Tests for the model-free dynamics filter (MM-7b)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import dynamics_filter as module

pytestmark = pytest.mark.unit

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_project_inside_moves_only_outside_points() -> None:
    inside = np.array([0.5, 0.5])
    np.testing.assert_allclose(module.project_inside(inside, SQUARE), inside)
    outside = np.array([1.5, 0.5])
    np.testing.assert_allclose(module.project_inside(outside, SQUARE), [1.0, 0.5])
    corner = np.array([2.0, 2.0])
    np.testing.assert_allclose(
        module.project_inside(corner, SQUARE), [1.0, 1.0], atol=1e-9
    )
    # A margin shrinks the admissible region.
    np.testing.assert_allclose(
        module.project_inside(outside, SQUARE, margin_m=0.1), [0.9, 0.5], atol=1e-9
    )
    # Vertex order does not matter.
    shuffled = SQUARE[[2, 0, 3, 1]]
    np.testing.assert_allclose(module.project_inside(outside, shuffled), [1.0, 0.5])
    # A margin that empties the hull returns the centroid.
    np.testing.assert_allclose(
        module.project_inside(outside, SQUARE, margin_m=0.6), [0.5, 0.5]
    )
    with pytest.raises(ValueError):
        module.project_inside(outside, SQUARE[:2])
    with pytest.raises(ValueError):
        module.project_inside(outside, SQUARE, margin_m=-1.0)


def test_cart_table_shift_closes_the_gap_through_position_and_acceleration() -> None:
    times = np.linspace(0.0, 1.0, 101)
    zmp = np.zeros((101, 2))
    target = zmp.copy()
    target[40:60, 0] = 0.05  # the reference must move 5 cm forward for 0.2 s
    shift = module.cart_table_shift(zmp, target, 0.9, times)
    assert shift.shape == (101, 2)
    np.testing.assert_allclose(shift[:2], 0.0)
    np.testing.assert_allclose(shift[:, 1], 0.0, atol=1e-12)  # nothing asked along y
    # The cart-table effect of the shift matches the gap where it was asked.
    dt = times[1] - times[0]
    acc = np.gradient(np.gradient(shift[:, 0], dt), dt)
    effect = shift[:, 0] - 0.9 / module.GRAVITY_M_S2 * acc
    assert np.abs(effect[45:55] - 0.05).max() < 0.01
    assert np.abs(shift[:, 0]).max() < 0.05  # cheaper than moving the mass 5 cm
    with pytest.raises(ValueError):
        module.cart_table_shift(zmp[:2], target[:2], 0.9, times[:2])
    with pytest.raises(ValueError):
        module.cart_table_shift(zmp, target, -1.0, times)
    with pytest.raises(ValueError):
        module.cart_table_shift(zmp, target, 0.9, times, acceleration_weight=-1.0)


def test_cart_table_shift_is_zero_without_a_gap() -> None:
    times = np.linspace(0.0, 0.5, 20)
    zmp = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(
        module.cart_table_shift(zmp, zmp, 0.9, times), 0.0, atol=1e-12
    )
