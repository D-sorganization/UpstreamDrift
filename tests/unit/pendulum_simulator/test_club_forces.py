"""Tests for src.shared.python.pendulum_simulator.club_forces (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.club_forces import (
    equivalent_couple,
    moment_of_net_force,
    net_force_on_club,
)


class TestNetForceOnClub:
    def test_sum_of_forces(self) -> None:
        f_r = np.array([3.0, 1.0])
        f_l = np.array([2.0, -1.0])
        result = net_force_on_club(f_r, f_l)
        np.testing.assert_allclose(result, [5.0, 0.0])

    def test_club_forces_returns_shape_2(self) -> None:
        result = net_force_on_club(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert result.shape == (2,)

    def test_zero_forces(self) -> None:
        result = net_force_on_club(np.zeros(2), np.zeros(2))
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_opposite_forces_cancel(self) -> None:
        f = np.array([5.0, 3.0])
        result = net_force_on_club(f, -f)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)

    def test_accepts_tuples(self) -> None:
        result = net_force_on_club((1.0, 2.0), (3.0, 4.0))
        np.testing.assert_allclose(result, [4.0, 6.0])


class TestMomentOfNetForce:
    def test_horizontal_force_at_lever(self) -> None:
        # r = force_point - action_point = (1,0) - (0,0) = (1,0)
        # F = (0, 10) → M = 1*10 - 0*0 = 10
        net = np.array([0.0, 10.0])
        force_pt = np.array([1.0, 0.0])
        action_pt = np.array([0.0, 0.0])
        assert moment_of_net_force(net, force_pt, action_pt) == pytest.approx(10.0)

    def test_zero_force_zero_moment(self) -> None:
        net = np.array([0.0, 0.0])
        force_pt = np.array([5.0, 3.0])
        action_pt = np.array([1.0, 1.0])
        assert moment_of_net_force(net, force_pt, action_pt) == pytest.approx(0.0)

    def test_coincident_points_zero_moment(self) -> None:
        net = np.array([1.0, 2.0])
        pt = np.array([3.0, 4.0])
        assert moment_of_net_force(net, pt, pt) == pytest.approx(0.0)

    def test_club_forces_returns_float(self) -> None:
        net = np.array([1.0, 0.0])
        result = moment_of_net_force(net, np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        assert isinstance(result, float)


class TestEquivalentCouple:
    def test_equal_and_opposite_forces_at_equal_distances(self) -> None:
        # Two hands at equal offsets, equal and opposite forces → couple = 2 * r * F
        action_pt = np.array([0.0, 0.0])
        pos_r = np.array([0.5, 0.0])
        pos_l = np.array([-0.5, 0.0])
        f_r = np.array([0.0, 10.0])
        f_l = np.array([0.0, 10.0])
        # r_r = (0.5, 0), F_r = (0, 10): moment = 0.5*10 - 0*0 = 5
        # r_l = (-0.5, 0), F_l = (0, 10): moment = -0.5*10 - 0*0 = -5
        result = equivalent_couple(f_r, pos_r, f_l, pos_l, action_pt)
        assert result == pytest.approx(0.0)

    def test_single_force_at_offset(self) -> None:
        action_pt = np.array([0.0, 0.0])
        pos_r = np.array([1.0, 0.0])
        pos_l = np.array([0.0, 0.0])  # at action point, contributes zero moment
        f_r = np.array([0.0, 5.0])
        f_l = np.array([0.0, 0.0])
        # r_r = (1,0), F_r = (0,5) → moment = 5
        # r_l = (0,0), F_l = (0,0) → moment = 0
        result = equivalent_couple(f_r, pos_r, f_l, pos_l, action_pt)
        assert result == pytest.approx(5.0)

    def test_club_forces_returns_float(self) -> None:
        result = equivalent_couple(
            np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
        )
        assert isinstance(result, float)

    def test_zero_forces_zero_couple(self) -> None:
        result = equivalent_couple(
            np.zeros(2),
            np.array([1.0, 0.0]),
            np.zeros(2),
            np.array([-1.0, 0.0]),
            np.zeros(2),
        )
        assert result == pytest.approx(0.0)
