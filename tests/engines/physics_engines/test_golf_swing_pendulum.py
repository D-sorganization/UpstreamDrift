"""Tests for GolfSwingPendulumEngine.

The engine wraps a Tools-vendored ``double_pendulum_golf`` package. When
the vendor submodule is not initialised the engine reports
``is_initialized = False`` and most methods return safe defaults. We test
the wrapper-layer behaviour exhaustively.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.golf_swing_physics_engine import (
    GolfSwingPendulumEngine,
)


@pytest.fixture
def engine() -> GolfSwingPendulumEngine:
    return GolfSwingPendulumEngine()


class TestConstruction:
    def test_engine_type(self, engine: GolfSwingPendulumEngine) -> None:
        assert engine.engine_type == "golf_swing_pendulum"

    def test_engine_name(self) -> None:
        assert GolfSwingPendulumEngine.ENGINE_NAME == "GolfSwingPendulum"

    def test_initial_state_zero(self, engine: GolfSwingPendulumEngine) -> None:
        q, v = engine.get_state()
        assert np.allclose(q, [0.0, 0.0])
        assert np.allclose(v, [0.0, 0.0])

    def test_time_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert engine.get_time() == 0.0

    def test_params_override_stored(self) -> None:
        eng = GolfSwingPendulumEngine(params={"m1": 7.0})
        assert eng._params_override == {"m1": 7.0}


class TestLoadAndNoop:
    def test_load_from_path_noop(self, engine: GolfSwingPendulumEngine) -> None:
        engine.load_from_path("ignored.urdf")  # should not raise

    def test_load_from_string_noop(self, engine: GolfSwingPendulumEngine) -> None:
        engine.load_from_string("<xml/>", extension="urdf")

    def test_forward_noop(self, engine: GolfSwingPendulumEngine) -> None:
        engine.forward()


class TestStateAndControl:
    def test_set_state(self, engine: GolfSwingPendulumEngine) -> None:
        engine.set_state(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        q, v = engine.get_state()
        assert np.allclose(q, [0.1, 0.2])
        assert np.allclose(v, [0.3, 0.4])

    def test_set_state_short_ignored(self, engine: GolfSwingPendulumEngine) -> None:
        engine.set_state(np.array([0.1]), np.array([0.3]))
        q, _ = engine.get_state()
        assert np.allclose(q, [0.0, 0.0])

    def test_set_control_writes_tau(self, engine: GolfSwingPendulumEngine) -> None:
        engine.set_control(np.array([1.0, -2.0]))
        assert np.allclose(engine._tau, [1.0, -2.0])
        assert engine._torque_profile is None

    def test_set_control_profile_then_set_control_clears_profile(
        self, engine: GolfSwingPendulumEngine
    ) -> None:
        engine.set_control_profile(lambda t: (1.0, 2.0))
        assert engine._torque_profile is not None
        engine.set_control(np.array([0.0, 0.0]))
        assert engine._torque_profile is None

    def test_reset(self, engine: GolfSwingPendulumEngine) -> None:
        engine.set_state(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
        engine.set_control(np.array([3.0, 3.0]))
        engine.time = 5.0
        engine.reset()
        q, v = engine.get_state()
        assert np.allclose(q, [0.0, 0.0])
        assert np.allclose(v, [0.0, 0.0])
        assert engine.time == 0.0
        assert np.allclose(engine._tau, [0.0, 0.0])


class TestUninitDefaults:
    """When Tools vendor isn't available, engine returns defaults."""

    def test_mass_matrix_identity(self, engine: GolfSwingPendulumEngine) -> None:
        assert np.allclose(engine.compute_mass_matrix(), np.eye(2))

    def test_bias_forces_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert np.allclose(engine.compute_bias_forces(), [0.0, 0.0])

    def test_gravity_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert np.allclose(engine.compute_gravity_forces(), [0.0, 0.0])

    def test_inverse_dynamics_zero(self, engine: GolfSwingPendulumEngine) -> None:
        out = engine.compute_inverse_dynamics(np.array([1.0, 2.0]))
        assert np.allclose(out, [0.0, 0.0])

    def test_drift_acceleration_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert np.allclose(engine.compute_drift_acceleration(), [0.0, 0.0])

    def test_control_acceleration_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert np.allclose(
            engine.compute_control_acceleration(np.array([1.0, 2.0])), [0.0, 0.0]
        )

    def test_compute_jacobian_uninit_returns_none(
        self, engine: GolfSwingPendulumEngine
    ) -> None:
        assert engine.compute_jacobian("tip") is None

    def test_ztcf_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert np.allclose(
            engine.compute_ztcf(np.array([0.1, 0.2]), np.array([0.0, 0.0])),
            [0.0, 0.0],
        )

    def test_zvcf_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert np.allclose(engine.compute_zvcf(np.array([0.1, 0.2])), [0.0, 0.0])

    def test_forward_kinematics_default(self, engine: GolfSwingPendulumEngine) -> None:
        out = engine.forward_kinematics()
        assert out == {
            "shoulder": (0.0, 0.0),
            "wrist": (0.0, 0.0),
            "tip": (0.0, 0.0),
        }

    def test_clubhead_speed_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert engine.clubhead_speed() == 0.0

    def test_total_energy_zero(self, engine: GolfSwingPendulumEngine) -> None:
        assert engine.total_energy() == 0.0

    def test_step_uninit_warns_no_raise(self, engine: GolfSwingPendulumEngine) -> None:
        engine.step(0.01)  # should warn but not raise


class TestArgValidation:
    def test_inverse_dynamics_none_raises(
        self, engine: GolfSwingPendulumEngine
    ) -> None:
        with pytest.raises(ValueError):
            engine.compute_inverse_dynamics(None)  # type: ignore[arg-type]

    def test_control_acceleration_none_raises(
        self, engine: GolfSwingPendulumEngine
    ) -> None:
        with pytest.raises(ValueError):
            engine.compute_control_acceleration(None)  # type: ignore[arg-type]

    def test_jacobian_none_raises(self, engine: GolfSwingPendulumEngine) -> None:
        with pytest.raises(ValueError):
            engine.compute_jacobian(None)  # type: ignore[arg-type]

    def test_ztcf_none_raises(self, engine: GolfSwingPendulumEngine) -> None:
        with pytest.raises(ValueError):
            engine.compute_ztcf(None, np.zeros(2))  # type: ignore[arg-type]

    def test_zvcf_none_raises(self, engine: GolfSwingPendulumEngine) -> None:
        with pytest.raises(ValueError):
            engine.compute_zvcf(None)  # type: ignore[arg-type]


class TestCheckpoint:
    def test_extra_checkpoint_round_trip(self, engine: GolfSwingPendulumEngine) -> None:
        engine._state = np.array([0.1, 0.2, 0.3, 0.4])
        engine._tau = np.array([1.0, 2.0])
        extra = engine._get_extra_checkpoint_state()
        assert extra["state"] == [0.1, 0.2, 0.3, 0.4]
        assert extra["tau"] == [1.0, 2.0]


class TestAvailability:
    def test_is_available_returns_bool(self) -> None:
        assert isinstance(GolfSwingPendulumEngine.is_available(), bool)
