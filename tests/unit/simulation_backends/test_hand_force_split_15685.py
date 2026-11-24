"""World-frame drift/control split of the hand-on-club force (#15685, K5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    drift_and_control_reactions,
)
from src.shared.python.simulation_backends.hand_force_split import (
    hand_force_split,
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
