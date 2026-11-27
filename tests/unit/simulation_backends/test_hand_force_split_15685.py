"""World-frame drift/control split of the hand-on-club force (#15685, K5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    constrained_acceleration_at,
    drift_and_control_reactions,
)
from src.shared.python.simulation_backends.hand_force_split import (
    HANDS,
    PARTS,
    hand_force_split,
    hand_wrench_split,
)

pytestmark = pytest.mark.unit


class _Provider:
    """Three coordinates, one 6-row closure; constant M, linear bias."""

    is_constrained = True
    nv = 6

    def __init__(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(size=(6, 6))
        self._mass = a @ a.T + 6 * np.eye(6)
        self._jac = rng.normal(size=(6, 6))[:3]

    def mass_matrix(self, q):
        return self._mass

    def bias_forces(self, q, v):
        return 2.0 * np.asarray(v) + np.array([0, 0, 9.8, 0, 0, 0])

    def closure_jacobian(self, q):
        return self._jac

    def closure_drift(self, q, v):
        return np.zeros(3)


class _SixRow(_Provider):
    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(5)
        self._jac = rng.normal(size=(6, 6)) + 3 * np.eye(6)

    def closure_drift(self, q, v):
        return np.zeros(6)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_parts_sum_to_the_total_and_are_the_club_side_world_force() -> None:
    provider = _SixRow()
    rot, origin = _rotation_z(0.4), np.array([0.1, 0.2, 1.0])
    q = np.zeros((2, 6))
    v = np.full((2, 6), 0.3)
    tau = np.ones((2, 6))
    split = hand_force_split(provider, lambda _q: (rot, origin), q, v, tau)
    drift, control = drift_and_control_reactions(provider, q[0], v[0], tau[0])
    np.testing.assert_allclose(split.drift_n[0], -rot @ drift[:3])
    np.testing.assert_allclose(split.control_n[0], -rot @ control[:3])
    np.testing.assert_allclose(split.point_m[1], origin)
    np.testing.assert_allclose(split.total_n, split.drift_n + split.control_n)
    assert split.drift_n.shape == (2, 3)


def test_shapes_are_checked() -> None:
    provider = _SixRow()
    frame = lambda _q: (np.eye(3), np.zeros(3))  # noqa: E731
    with pytest.raises(ValueError, match="rows"):
        hand_force_split(
            provider, frame, np.zeros((2, 6)), np.zeros((3, 6)), np.zeros((2, 6))
        )
    with pytest.raises(ValueError, match="six"):
        hand_force_split(
            _Provider(), frame, np.zeros((1, 6)), np.zeros((1, 6)), np.zeros((1, 6))
        )


# --- Both hands, ZTCF and ZVCF (#15150 Track Z, Z2) -------------------------

_K = np.random.default_rng(11).normal(size=(6, 6))
_GRAVITY = np.array([0.0, 0.0, -9.81])
_CLUB_MASS = 0.31


def _club_load(q, v, a, gravity):
    """Fake bare-club net load: affine in ``a``, quadratic in ``v``, gravity once.

    Returns (force, moment about the world origin) the hands must supply.
    """
    v = np.asarray(v, dtype=float)
    load = _K @ np.asarray(a, dtype=float) + 0.5 * float(v @ v) * np.ones(6)
    if gravity:
        load[:3] -= _CLUB_MASS * _GRAVITY
    return load[:3], load[3:]


def _lead_point(_q):
    return np.array([0.05, -0.1, 1.1])


def _split(q, v, tau):
    rot, origin = _rotation_z(0.4), np.array([0.1, 0.2, 1.0])
    return (
        hand_wrench_split(
            _SixRow(), lambda _q: (rot, origin), _club_load, _lead_point, q, v, tau
        ),
        rot,
        origin,
    )


def test_each_hand_ztcf_plus_zvcf_is_its_total() -> None:
    q, v, tau = np.zeros((2, 6)), np.full((2, 6), 0.3), np.ones((2, 6))
    split, rot, origin = _split(q, v, tau)
    a, lam = constrained_acceleration_at(_SixRow(), q[0], v[0], tau[0])
    trail = split.total("trail")
    np.testing.assert_allclose(trail.force_n[0], -rot @ lam[:3], atol=1e-9)
    np.testing.assert_allclose(trail.torque_nm[0], -rot @ lam[3:], atol=1e-9)
    net_f, net_m = _club_load(q[0], v[0], a, True)
    lead = split.total("lead")
    np.testing.assert_allclose(lead.force_n[0] + trail.force_n[0], net_f, atol=1e-9)
    moment = (
        lead.torque_nm[0]
        + np.cross(lead.point_m[0], lead.force_n[0])
        + trail.torque_nm[0]
        + np.cross(trail.point_m[0], trail.force_n[0])
    )
    np.testing.assert_allclose(moment, net_m, atol=1e-9)
    for hand in HANDS:
        parts = [split.part(hand, p) for p in PARTS]
        np.testing.assert_allclose(
            parts[0].force_n + parts[1].force_n, split.total(hand).force_n
        )


def test_gravity_is_counted_once_in_the_ztcf() -> None:
    """Owner, 2026-11-24: gravity is passive; the ZVCF is the torques alone."""
    q, v = np.zeros((1, 6)), np.zeros((1, 6))
    split, _, _ = _split(q, v, np.zeros((1, 6)))
    for hand in HANDS:
        zvcf = split.part(hand, "zvcf")
        np.testing.assert_allclose(zvcf.force_n, 0.0, atol=1e-12)
        np.testing.assert_allclose(zvcf.torque_nm, 0.0, atol=1e-12)
    supported = split.part("lead", "ztcf").force_n + split.part("trail", "ztcf").force_n
    assert np.linalg.norm(supported) > 0.0


def test_unknown_hand_or_part_is_refused() -> None:
    split, _, _ = _split(np.zeros((1, 6)), np.zeros((1, 6)), np.zeros((1, 6)))
    with pytest.raises(ValueError, match="hand"):
        split.part("left", "ztcf")
    with pytest.raises(ValueError, match="part"):
        split.part("lead", "gravity")
