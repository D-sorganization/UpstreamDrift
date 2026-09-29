"""Comprehensive tests for PendulumPhysicsEngine.

The double pendulum engine is a pure-Python engine with no external
dependencies, so we test it end-to-end without mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint


@pytest.fixture
def engine() -> PendulumPhysicsEngine:
    return PendulumPhysicsEngine()


class TestInitAndProperties:
    def test_engine_type(self, engine: PendulumPhysicsEngine) -> None:
        assert engine.engine_type == "pendulum"

    def test_is_initialized_on_construction(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        assert engine.is_initialized is True

    def test_model_name(self, engine: PendulumPhysicsEngine) -> None:
        assert engine.model_name_str == "DoublePendulum"

    def test_initial_state_is_zero(self, engine: PendulumPhysicsEngine) -> None:
        q, v = engine.get_state()
        assert np.allclose(q, [0.0, 0.0])
        assert np.allclose(v, [0.0, 0.0])

    def test_initial_time_is_zero(self, engine: PendulumPhysicsEngine) -> None:
        assert engine.get_time() == 0.0


class TestLoad:
    def test_load_from_path_is_noop(self, engine: PendulumPhysicsEngine) -> None:
        engine.load_from_path("/any/fake/path.urdf")
        assert engine.is_initialized is True

    def test_load_from_string_is_noop(self, engine: PendulumPhysicsEngine) -> None:
        engine.load_from_string("<urdf/>", extension="urdf")
        assert engine.is_initialized is True


class TestStateAndControl:
    def test_set_state_writes_q_and_v(self, engine: PendulumPhysicsEngine) -> None:
        engine.set_state(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        q, v = engine.get_state()
        assert np.allclose(q, [0.1, 0.2])
        assert np.allclose(v, [0.3, 0.4])

    def test_set_state_ignores_short_arrays(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        engine.set_state(np.array([0.1]), np.array([0.3]))
        q, _ = engine.get_state()
        assert np.allclose(q, [0.0, 0.0])

    def test_set_control_copies_vector(self, engine: PendulumPhysicsEngine) -> None:
        u = np.array([1.0, 2.0])
        engine.set_control(u)
        u[0] = 99.0
        assert engine.control[0] == 1.0

    def test_set_control_ignores_short_vector(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        engine.set_control(np.array([5.0]))
        assert np.allclose(engine.control, [0.0, 0.0])

    def test_forcing_callbacks_reflect_control(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        engine.set_control(np.array([1.5, -2.5]))
        s = engine._pendulum_state
        assert engine._get_shoulder_torque(0.0, s) == pytest.approx(1.5)
        assert engine._get_wrist_torque(0.0, s) == pytest.approx(-2.5)


class TestStepReset:
    def test_step_advances_time(self, engine: PendulumPhysicsEngine) -> None:
        engine.step(0.01)
        assert engine.get_time() == pytest.approx(0.01)

    def test_step_default_dt(self, engine: PendulumPhysicsEngine) -> None:
        engine.step()
        assert engine.get_time() == pytest.approx(0.01)

    def test_reset_restores_initial_state(self, engine: PendulumPhysicsEngine) -> None:
        engine.set_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        engine.set_control(np.array([5.0, 6.0]))
        engine.step(0.01)
        engine.reset()
        q, v = engine.get_state()
        assert np.allclose(q, [0.0, 0.0])
        assert np.allclose(v, [0.0, 0.0])
        assert engine.get_time() == 0.0
        assert np.allclose(engine.control, [0.0, 0.0])

    def test_forward_is_noop(self, engine: PendulumPhysicsEngine) -> None:
        engine.forward()  # should not raise


class TestDynamics:
    def test_mass_matrix_shape_and_finite(self, engine: PendulumPhysicsEngine) -> None:
        M = engine.compute_mass_matrix()
        assert M.shape == (2, 2)
        assert np.all(np.isfinite(M))

    def test_bias_forces_shape(self, engine: PendulumPhysicsEngine) -> None:
        b = engine.compute_bias_forces()
        assert b.shape == (2,)
        assert np.all(np.isfinite(b))

    def test_gravity_forces_shape(self, engine: PendulumPhysicsEngine) -> None:
        g = engine.compute_gravity_forces()
        assert g.shape == (2,)

    def test_inverse_dynamics_short_returns_empty(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        out = engine.compute_inverse_dynamics(np.array([1.0]))
        assert out.size == 0

    def test_inverse_dynamics_finite(self, engine: PendulumPhysicsEngine) -> None:
        tau = engine.compute_inverse_dynamics(np.array([0.1, 0.2]))
        assert tau.shape == (2,)
        assert np.all(np.isfinite(tau))

    def test_drift_acceleration_finite(self, engine: PendulumPhysicsEngine) -> None:
        a = engine.compute_drift_acceleration()
        assert a.shape == (2,)
        assert np.all(np.isfinite(a))

    def test_control_acceleration_short_returns_empty(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        assert engine.compute_control_acceleration(np.array([1.0])).size == 0

    def test_control_acceleration_value(self, engine: PendulumPhysicsEngine) -> None:
        a = engine.compute_control_acceleration(np.array([1.0, 2.0]))
        assert a.shape == (2,)
        assert np.all(np.isfinite(a))

    def test_jacobian_placeholder_returns_none(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        assert engine.compute_jacobian("anything") is None


class TestCounterfactuals:
    def test_ztcf_preserves_original_state(self, engine: PendulumPhysicsEngine) -> None:
        engine.set_state(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        out = engine.compute_ztcf(np.array([1.0, 1.5]), np.array([0.5, 0.6]))
        assert out.shape == (2,)
        q, v = engine.get_state()
        assert np.allclose(q, [0.1, 0.2])
        assert np.allclose(v, [0.3, 0.4])

    def test_ztcf_short_args_empty(self, engine: PendulumPhysicsEngine) -> None:
        out = engine.compute_ztcf(np.array([1.0]), np.array([0.5]))
        assert out.size == 0

    def test_zvcf_preserves_original_state(self, engine: PendulumPhysicsEngine) -> None:
        engine.set_state(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        engine.set_control(np.array([1.0, 1.0]))
        out = engine.compute_zvcf(np.array([1.5, 1.6]))
        assert out.shape == (2,)
        q, v = engine.get_state()
        assert np.allclose(q, [0.1, 0.2])
        assert np.allclose(v, [0.3, 0.4])

    def test_zvcf_short_args_empty(self, engine: PendulumPhysicsEngine) -> None:
        assert engine.compute_zvcf(np.array([1.0])).size == 0


class TestCheckpointHooks:
    def test_extra_checkpoint_state_contains_phi(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        engine._pendulum_state.phi = 0.5
        engine._pendulum_state.omega_phi = 1.5
        extra = engine._get_extra_checkpoint_state()
        assert extra == {"phi": 0.5, "omega_phi": 1.5}

    def test_restore_extra_checkpoint_state(
        self, engine: PendulumPhysicsEngine
    ) -> None:
        cp = StateCheckpoint(
            id="cp",
            timestamp=2.5,
            wall_time=0.0,
            engine_type="pendulum",
            engine_state={"phi": 0.7, "omega_phi": 0.9},
            q=(0.0, 0.0),
            v=(0.0, 0.0),
        )
        engine._restore_extra_checkpoint_state(cp)
        assert engine.time == 2.5
        assert engine._pendulum_state.phi == 0.7
        assert engine._pendulum_state.omega_phi == 0.9
