"""Squeeze and couple-consistency checks of the hand internal force (#11739)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact import ClubDynamics
from src.shared.python.grip_contact.couple_check import (
    ClubKinematics,
    couple_consistency,
    peak_squeeze_n,
    required_hand_moment_nm,
)

pytestmark = pytest.mark.unit

N = 41
T = np.linspace(0.0, 0.4, N)
G = np.array([0.0, 0.0, -9.81])
CLUB = ClubDynamics(0.3, (0.0, 0.0, 0.0), np.diag([0.002, 0.08, 0.08]))
P_L = np.tile([0.0, 0.0, 0.0], (N, 1))
P_R = np.tile([0.076, 0.0, 0.0], (N, 1))  # inter-hand line along +x
MID = 0.5 * (P_L + P_R)


def _tile(v) -> np.ndarray:
    return np.tile(np.asarray(v, float), (N, 1))


def _consistent_case():
    """Hand wrenches and the moment they apply about the midpoint."""
    rng = np.random.default_rng(3)
    f_l = rng.normal(0.0, 50.0, (N, 3))
    f_r = rng.normal(0.0, 50.0, (N, 3))
    t_l = rng.normal(0.0, 5.0, (N, 3))
    t_r = rng.normal(0.0, 5.0, (N, 3))
    m_mid = t_l + t_r + np.cross(P_L - MID, f_l) + np.cross(P_R - MID, f_r)
    return (f_l, f_r), (t_l, t_r), m_mid


def test_static_hold_needs_only_the_weight_moment() -> None:
    rot = np.repeat(np.eye(3)[None], N, axis=0)
    com = _tile([0.5, 0.0, 0.0])
    zero = _tile([0, 0, 0])
    club = ClubKinematics(rot, zero, zero, com, zero)
    m = required_hand_moment_nm(club, CLUB, G, MID)
    expected = np.cross(com[0] - MID[0], CLUB.mass_kg * -G)
    np.testing.assert_allclose(m, _tile(expected), atol=1e-12)


def test_angular_acceleration_needs_i_alpha() -> None:
    alpha = 30.0
    rot = np.repeat(np.eye(3)[None], N, axis=0)  # inertia is axisymmetric in z
    omega = np.column_stack([np.zeros(N), np.zeros(N), alpha * T])
    zero = _tile([0, 0, 0])
    acc = _tile([0.0, 0.0, alpha])
    club = ClubKinematics(rot, omega, acc, zero, zero)
    m = required_hand_moment_nm(club, CLUB, np.zeros(3), zero)
    np.testing.assert_allclose(m[:, 2], 0.08 * alpha, rtol=1e-9)
    np.testing.assert_allclose(m[:, :2], 0.0, atol=1e-12)


def test_consistent_wrenches_have_zero_error() -> None:
    forces, torques, m_mid = _consistent_case()
    res = couple_consistency(forces, (P_L, P_R), torques, m_mid, noise_floor_nm=0.0)
    assert res.checked.all()
    assert res.max_relative_error() < 1e-9


def test_squeeze_is_invisible_to_the_couple_but_caught_by_the_squeeze_check() -> None:
    (f_l, f_r), torques, m_mid = _consistent_case()
    squeeze = _tile([60.0, 0.0, 0.0])  # left pushes toward the right hand
    res = couple_consistency(
        (f_l + squeeze, f_r - squeeze), (P_L, P_R), torques, m_mid, 0.0
    )
    assert res.max_relative_error() < 1e-9
    base = peak_squeeze_n(f_l, f_r, P_L, P_R)
    assert peak_squeeze_n(f_l + squeeze, f_r - squeeze, P_L, P_R) > base + 20.0


def test_an_unneeded_transverse_pair_fails_the_consistency_check() -> None:
    (f_l, f_r), torques, m_mid = _consistent_case()
    extra = _tile([0.0, 0.0, 200.0])  # a pair the club motion does not need
    res = couple_consistency(
        (f_l + extra, f_r - extra), (P_L, P_R), torques, m_mid, 0.0
    )
    assert res.max_relative_error() > 0.05


def test_noise_floor_skips_small_couples() -> None:
    forces, torques, m_mid = _consistent_case()
    res = couple_consistency(forces, (P_L, P_R), torques, m_mid, noise_floor_nm=1e6)
    assert not res.checked.any()
    assert res.max_relative_error() == 0.0


def test_preconditions() -> None:
    forces, torques, m_mid = _consistent_case()
    with pytest.raises(ValueError, match="noise_floor"):
        couple_consistency(forces, (P_L, P_R), torques, m_mid, -1.0)
    with pytest.raises(ValueError, match="shape"):
        couple_consistency(forces, (P_L, P_R), torques, m_mid[:5])
    rot = np.repeat(np.eye(3)[None], N, axis=0)
    zero = _tile([0, 0, 0])
    with pytest.raises(ValueError, match="rotation"):
        ClubKinematics(rot[:3], zero, zero, zero, zero)
    with pytest.raises(ValueError, match="alpha"):
        ClubKinematics(rot, zero, zero[:4], zero, zero)
    club = ClubKinematics(rot, zero, zero, zero, zero)
    with pytest.raises(ValueError, match="gravity"):
        required_hand_moment_nm(club, CLUB, [0.0, 1.0], zero)
    assert club.subset(np.arange(N) < 5).com_m.shape == (5, 3)
