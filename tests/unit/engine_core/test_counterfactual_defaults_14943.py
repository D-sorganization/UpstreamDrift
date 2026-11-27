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


class _PositionsOnlyJointEngine:
    """Readable q/v and set_joint_positions only (no set_state / set_joint_velocities)."""

    def __init__(self) -> None:
        self._q = np.array([0.25, -0.1])
        self._v = np.array([1.5, -2.0])

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._v.copy()

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).copy()

    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(2)

    def compute_gravity_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        return np.asarray(tau, dtype=float).reshape(-1)


def test_joint_positions_without_velocities_writer_raises() -> None:
    engine = _PositionsOnlyJointEngine()
    q_cf = np.array([0.9, 0.3])
    tau = np.array([0.1, 0.2])
    with pytest.raises(NotImplementedError, match="set_joint_velocities"):
        default_control_acceleration_at_state(engine, q_cf, tau)
    np.testing.assert_array_equal(engine.get_joint_positions(), [0.25, -0.1])
    np.testing.assert_array_equal(engine.get_joint_velocities(), [1.5, -2.0])


class _SplitJointWriterEngine(_PositionsOnlyJointEngine):
    def set_joint_velocities(self, v: np.ndarray) -> None:
        self._v = np.asarray(v, dtype=float).copy()

    def __init__(self) -> None:
        super().__init__()
        self.v_at_eval: np.ndarray | None = None

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        self.v_at_eval = self.get_joint_velocities().copy()
        return np.asarray(tau, dtype=float).reshape(-1)


def test_split_joint_writer_zeros_velocity_and_restores_state() -> None:
    engine = _SplitJointWriterEngine()
    q_orig, v_orig = engine.get_joint_positions(), engine.get_joint_velocities()
    q_cf = np.array([1.1, -0.4])
    tau = np.array([0.3, 0.7])
    out = default_control_acceleration_at_state(engine, q_cf, tau)
    np.testing.assert_allclose(out, tau)
    assert engine.v_at_eval is not None
    np.testing.assert_allclose(engine.v_at_eval, 0.0, atol=0.0)
    np.testing.assert_array_equal(engine.get_joint_positions(), q_orig)
    np.testing.assert_array_equal(engine.get_joint_velocities(), v_orig)


class _SplitJointWriterRaisesEngine(_SplitJointWriterEngine):
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:  # noqa: ARG002
        self.v_at_eval = self.get_joint_velocities().copy()
        raise RuntimeError("counterfactual eval failed")


def test_split_joint_writer_restores_state_when_evaluate_raises() -> None:
    engine = _SplitJointWriterRaisesEngine()
    q_orig, v_orig = engine.get_joint_positions(), engine.get_joint_velocities()
    q_cf = np.array([0.6, 0.2])
    tau = np.array([1.0, 0.0])
    with pytest.raises(RuntimeError, match="counterfactual eval failed"):
        default_control_acceleration_at_state(engine, q_cf, tau)
    np.testing.assert_allclose(engine.v_at_eval, 0.0, atol=0.0)
    np.testing.assert_array_equal(engine.get_joint_positions(), q_orig)
    np.testing.assert_array_equal(engine.get_joint_velocities(), v_orig)
