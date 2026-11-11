"""Regression: protocol defaults evaluate counterfactuals at explicit q (#14943)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)
from src.shared.python.engine_core.counterfactual_defaults import (
    default_control_acceleration_at_state,
    default_gravity_only_acceleration,
)

pytestmark = pytest.mark.unit


def test_default_control_acceleration_at_state_uses_explicit_q() -> None:
    engine = PendulumPhysicsEngine()
    engine.set_state(np.array([0.35, -0.2]), np.array([0.5, -0.3]))
    q_other = np.array([1.05, -0.75])
    tau = np.array([0.4, 0.2])
    via_default = default_control_acceleration_at_state(engine, q_other, tau)
    native = engine.compute_control_acceleration_at_state(q_other, tau)
    np.testing.assert_allclose(via_default, native, rtol=1e-9, atol=1e-9)
    # M(q) at the live configuration differs from M(q_other) for this tau.
    at_current = engine.compute_control_acceleration(tau)
    assert not np.allclose(via_default, at_current, rtol=1e-6, atol=1e-6)


def test_default_gravity_only_honors_explicit_q() -> None:
    engine = PendulumPhysicsEngine()
    engine.set_state(np.array([0.1, 0.1]), np.array([0.0, 0.0]))
    q_other = np.array([1.2, -0.4])
    via_default = default_gravity_only_acceleration(engine, q_other)
    native = engine.compute_gravity_only_acceleration(q_other)
    np.testing.assert_allclose(via_default, native, rtol=1e-9, atol=1e-9)
    at_current = engine.compute_gravity_only_acceleration(engine.get_state()[0])
    assert not np.allclose(via_default, at_current, rtol=1e-6, atol=1e-6)


class _StatelessControlEngine:
    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(2)

    def compute_gravity_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        return np.asarray(tau, dtype=float).reshape(-1)


def test_default_control_acceleration_requires_configurable_state() -> None:
    q = np.array([0.2, -0.1])
    tau = np.array([1.0, 0.5])
    with pytest.raises(NotImplementedError, match="readable joint/state"):
        default_control_acceleration_at_state(_StatelessControlEngine(), q, tau)
