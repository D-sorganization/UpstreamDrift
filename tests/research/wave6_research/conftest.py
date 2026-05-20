"""Shared fixtures for wave6 research tests."""

from __future__ import annotations

import numpy as np
import pytest


class FakeEngine:
    """Minimal physics engine stub for MPC/diff tests."""

    def __init__(self, n_q: int = 2, n_v: int | None = None) -> None:
        self.n_q = n_q
        self.n_v = n_v if n_v is not None else n_q
        self._q = np.zeros(self.n_q)
        self._v = np.zeros(self.n_v)
        self._tau = np.zeros(self.n_v)

    def set_joint_positions(self, q):  # noqa: ANN001
        self._q = np.asarray(q, dtype=float).copy()

    def set_joint_velocities(self, v):  # noqa: ANN001
        self._v = np.asarray(v, dtype=float).copy()

    def set_joint_torques(self, tau):  # noqa: ANN001
        self._tau = np.asarray(tau, dtype=float).copy()

    def get_joint_positions(self):
        return self._q.copy()

    def get_joint_velocities(self):
        return self._v.copy()

    def step(self, dt: float) -> None:
        # Linear: v += tau*dt (unit mass), q += v*dt
        self._v = self._v + self._tau * dt
        self._q = self._q + self._v * dt


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine(n_q=2, n_v=2)


@pytest.fixture
def small_engine() -> FakeEngine:
    return FakeEngine(n_q=1, n_v=1)
