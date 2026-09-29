"""Tests for ZMPComputer."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.locomotion.zmp_computer import ZMPComputer, ZMPResult


class FakeRC:
    def get_state(self):
        return np.zeros(3), np.zeros(3)

    def set_state(self, q, v):
        return None

    def compute_mass_matrix(self):
        return np.eye(3)

    def compute_bias_forces(self):
        return np.zeros(3)

    def compute_gravity_forces(self):
        return np.zeros(3)

    def compute_jacobian(self, body_name):
        return None

    def get_time(self):
        return 0.0


class FakeHumanoid(FakeRC):
    def __init__(self, com=np.zeros(3), com_v=np.zeros(3), mass=70.0) -> None:
        self._com = np.asarray(com, dtype=float)
        self._com_v = np.asarray(com_v, dtype=float)
        self._mass = mass

    def get_com_position(self):
        return self._com.copy()

    def get_com_velocity(self):
        return self._com_v.copy()

    def get_total_mass(self):
        return self._mass

    def compute_centroidal_momentum(self):
        return np.zeros(6)

    def compute_centroidal_momentum_matrix(self):
        return np.zeros((6, 3))

    def get_foot_position(self, foot):
        return np.zeros(3)

    def get_foot_velocity(self, foot):
        return np.zeros(3)

    def get_foot_jacobian(self, foot):
        return np.zeros((6, 3))


def test_construct_humanoid() -> None:
    z = ZMPComputer(FakeHumanoid())
    assert z.ground_height == 0.0
    z.ground_height = 0.1
    assert z.ground_height == 0.1


def test_compute_zmp_static() -> None:
    eng = FakeHumanoid(com=np.array([0.0, 0.0, 1.0]))
    z = ZMPComputer(eng)
    r = z.compute_zmp(com_acceleration=np.zeros(3))
    assert isinstance(r, ZMPResult)
    assert r.zmp_position[2] == 0.0
    assert np.isfinite(r.support_margin)


def test_compute_zmp_free_fall_invalid() -> None:
    eng = FakeHumanoid(com=np.array([0.0, 0.0, 1.0]))
    z = ZMPComputer(eng)
    # Set vertical acceleration to -g => denom ~ 0
    r = z.compute_zmp(com_acceleration=np.array([0.0, 0.0, -z.GRAVITY]))
    assert not r.is_valid
    assert r.total_normal_force == 0.0


def test_compute_zmp_with_angular_momentum() -> None:
    eng = FakeHumanoid(com=np.array([0.0, 0.0, 1.0]))
    z = ZMPComputer(eng)
    r = z.compute_zmp(
        com_acceleration=np.zeros(3),
        angular_momentum_rate=np.array([0.1, 0.2, 0.0]),
    )
    assert r.zmp_position.shape == (3,)


def test_compute_capture_point() -> None:
    eng = FakeHumanoid(
        com=np.array([0.0, 0.0, 1.0]),
        com_v=np.array([0.5, 0.0, 0.0]),
    )
    z = ZMPComputer(eng)
    cp = z.compute_capture_point()
    assert cp.shape == (3,)
    assert cp[0] > 0  # Should be ahead of CoM


def test_compute_capture_point_low_height() -> None:
    eng = FakeHumanoid(com=np.array([0.0, 0.0, -1.0]))
    z = ZMPComputer(eng, ground_height=0.0)
    cp = z.compute_capture_point()
    assert cp.shape == (3,)


def test_compute_dcm_equals_capture_point() -> None:
    eng = FakeHumanoid(
        com=np.array([0.0, 0.0, 1.0]),
        com_v=np.array([0.5, 0.0, 0.0]),
    )
    z = ZMPComputer(eng)
    cp = z.compute_capture_point()
    dcm = z.compute_dcm()
    assert np.allclose(cp, dcm)


def test_stability_margin_default_polygon() -> None:
    z = ZMPComputer(FakeHumanoid())
    m = z.compute_stability_margin(np.zeros(3))
    assert m > 0  # origin should be inside default 26x15cm footprint


def test_stability_margin_outside() -> None:
    z = ZMPComputer(FakeHumanoid())
    m = z.compute_stability_margin(np.array([1.0, 0.0, 0.0]))
    assert m < 0


def test_stability_margin_custom_polygon_lt3() -> None:
    z = ZMPComputer(FakeHumanoid())
    bad_poly = np.array([[0.0, 0.0], [1.0, 0.0]])
    m = z.compute_stability_margin(np.zeros(3), support_polygon=bad_poly)
    assert m == pytest.approx(-1.0)


def test_get_com_position_non_humanoid_raises() -> None:
    z = ZMPComputer(FakeRC())
    with pytest.raises(ValueError, match="COM position"):
        z._get_com_position()


def test_get_com_velocity_non_humanoid_returns_zeros() -> None:
    z = ZMPComputer(FakeRC())
    v = z._get_com_velocity()
    assert np.allclose(v, 0.0)


def test_estimate_mass_non_humanoid_default() -> None:
    z = ZMPComputer(FakeRC())
    m = z._estimate_mass()
    assert m > 0


def test_check_support_custom_polygon_inside() -> None:
    z = ZMPComputer(FakeHumanoid())
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    inside, margin = z._check_support(np.array([0.5, 0.5]), poly)
    assert inside
    assert margin > 0


def test_check_support_custom_polygon_outside() -> None:
    z = ZMPComputer(FakeHumanoid())
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    inside, margin = z._check_support(np.array([2.0, 0.5]), poly)
    assert not inside
    assert margin < 0


def test_check_support_polygon_too_small() -> None:
    z = ZMPComputer(FakeHumanoid())
    poly = np.array([[0.0, 0.0], [1.0, 0.0]])
    inside, margin = z._check_support(np.zeros(2), poly)
    assert not inside
    assert margin == pytest.approx(-1.0)
