"""Unit tests for the engine-free parts of the Crocoddyl action layer (MS-31).

A one-dof fake plant (mass-spring-damper with an actuated effort) stands in
for the Pinocchio plant so the linearly implicit rollout and the tracking
warm start can be checked without any engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.crocoddyl_action import (
    implicit_euler_rollout,
    tracking_rollout,
)

pytestmark = pytest.mark.unit


@dataclass
class _Derivatives:
    dq: np.ndarray
    dv: np.ndarray
    deffort: np.ndarray


class FakePlant:
    """q'' = -k q - c v + u with one actuated coordinate and one unactuated."""

    def __init__(self, k: float = 100.0, c: float = 8000.0) -> None:
        self.n = 2
        self.actuated = np.array([False, True])
        self.lower = np.array([-np.inf, -np.inf])
        self.upper = np.array([np.inf, np.inf])
        self.k = k
        self.c = c

    def acceleration(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return -self.k * q - self.c * v + tau

    def derivatives(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray
    ) -> _Derivatives:
        return _Derivatives(-self.k * np.eye(2), -self.c * np.eye(2), np.eye(2))

    def effort_vector(self, u: np.ndarray) -> np.ndarray:
        tau = np.zeros(2)
        tau[self.actuated] = u
        return tau


def test_implicit_rollout_is_stable_where_explicit_euler_is_not() -> None:
    plant = FakePlant()
    dt = 1.0 / 360.0
    us = np.zeros((200, 1))
    q, v = implicit_euler_rollout(plant, np.array([0.1, 0.1]), np.zeros(2), us, dt)
    assert np.isfinite(q).all()
    assert np.abs(q[-1]).max() < 0.1  # heavily damped, decays
    # explicit Euler with c*dt = 22 diverges
    q_e, v_e = np.array([0.1, 0.1]), np.zeros(2)
    for _ in range(10):
        v_e = v_e + dt * plant.acceleration(q_e, v_e, np.zeros(2))
        q_e = q_e + dt * v_e
    assert np.abs(q_e).max() > 1.0


def test_tracking_rollout_follows_reference_within_bounds() -> None:
    plant = FakePlant(k=0.0, c=0.0)
    dt = 1.0 / 360.0
    t = np.arange(60) * dt
    q_ref = np.stack([np.zeros_like(t), 0.05 * np.sin(2 * np.pi * 2 * t)], axis=1)
    v_ref = np.gradient(q_ref, dt, axis=0)
    a_ref = np.gradient(v_ref, dt, axis=0)
    q, v, us = tracking_rollout(
        plant,
        q_ref,
        v_ref,
        a_ref,
        dt,
        kp=400.0,
        kd=40.0,
        effort_bounds=np.array([50.0]),
        ridge=1e-8,
    )
    assert us.shape == (59, 1)
    assert np.abs(us).max() <= 50.0
    assert np.abs(q[:, 1] - q_ref[:, 1]).max() < 2e-3
    assert np.allclose(q[:, 0], 0.0)  # unactuated coordinate untouched
