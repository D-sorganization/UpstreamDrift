"""Tests for MyoSuitePhysicsEngine and its mixins.

MyoSuite wraps a Gym environment around a MuJoCo sim. We mock both the
gym ``env`` and the underlying ``sim`` so the mixin glue (init, simulation
core, dynamics, drift control, muscle interface) is exercised without
needing MyoSuite/MuJoCo installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.engines.physics_engines.myosuite.python._engine_init import EngineInitMixin
from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)
from src.shared.python.core.contracts.exceptions import PreconditionError


def _make_sim(nv: int = 2) -> MagicMock:
    sim = MagicMock()
    sim.model.opt.timestep = 0.002
    sim.model.nv = nv
    sim.data.qpos = np.zeros(nv)
    sim.data.qvel = np.zeros(nv)
    sim.data.qacc = np.zeros(nv)
    sim.data.qfrc_bias = np.zeros(nv)
    sim.data.qfrc_inverse = np.zeros(nv)
    sim.data.qM = np.zeros(nv)
    sim.data.ctrl = np.zeros(nv)
    sim.data.time = 0.0
    return sim


@pytest.fixture
def engine() -> MyoSuitePhysicsEngine:
    return MyoSuitePhysicsEngine()


@pytest.fixture
def loaded_engine() -> MyoSuitePhysicsEngine:
    eng = MyoSuitePhysicsEngine()
    sim = _make_sim()
    env = MagicMock()
    env.sim = sim
    env.action_space.sample = MagicMock(return_value=np.zeros(2))
    env.step = MagicMock(return_value=(None, 0.0, False, False, {}))
    eng.env = env
    eng.sim = sim
    eng.env_id = "myoTest-v0"
    eng._dt = 0.002
    return eng


class TestEngineInitMixin:
    def test_default_state(self, engine: MyoSuitePhysicsEngine) -> None:
        assert engine.env is None
        assert engine.sim is None
        assert engine.is_initialized is False
        assert engine.model_name == "MyoSuite_NoModel"
        assert engine.model is None

    def test_reset_loaded_state(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine._reset_loaded_state()
        assert loaded_engine.env is None
        assert loaded_engine.sim is None
        assert loaded_engine.env_id == ""

    def test_model_returns_sim_model(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        assert loaded_engine.model is loaded_engine.sim.model

    def test_model_name_uses_env_id(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        assert loaded_engine.model_name == "myoTest-v0"

    def test_extract_sim_from_env_direct(self) -> None:
        env = MagicMock()
        env.sim = "mysim"
        assert EngineInitMixin._extract_sim_from_env(env) == "mysim"

    def test_extract_sim_from_env_unwrapped(self) -> None:
        env = MagicMock()
        env.sim = None
        env.unwrapped.sim = "deep"
        assert EngineInitMixin._extract_sim_from_env(env) == "deep"

    def test_extract_sim_from_env_missing_raises(self) -> None:
        env = MagicMock()
        env.sim = None
        env.unwrapped.sim = None
        with pytest.raises(RuntimeError, match="Could not extract"):
            EngineInitMixin._extract_sim_from_env(env)

    def test_load_from_path_without_myosuite_raises(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(ImportError):
            engine.load_from_path("anything")

    def test_load_from_string_unsupported(self, engine: MyoSuitePhysicsEngine) -> None:
        with pytest.raises(RuntimeError, match="does not support"):
            engine.load_from_string("xml", extension="xml")

    def test_load_from_path_success(self, engine: MyoSuitePhysicsEngine) -> None:
        """When myosuite is available, load_from_path uses gym.make."""
        # Patch the module-level MYOSUITE_AVAILABLE and gym shim.
        sim = _make_sim()
        env = MagicMock()
        env.reset = MagicMock()
        env.sim = sim

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = MagicMock(return_value=env)

        from src.engines.physics_engines.myosuite.python import (
            _engine_init as init_mod,
        )

        sys.modules["gymnasium"] = fake_gym
        init_mod.gym = fake_gym
        init_mod.MYOSUITE_AVAILABLE = True
        try:
            engine.load_from_path("myoTest-v0")
            assert engine.is_initialized
            assert engine.env_id == "myoTest-v0"
        finally:
            init_mod.MYOSUITE_AVAILABLE = False
            sys.modules.pop("gymnasium", None)

    def test_load_from_path_failure_resets(self, engine: MyoSuitePhysicsEngine) -> None:
        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = MagicMock(side_effect=ValueError("nope"))
        from src.engines.physics_engines.myosuite.python import (
            _engine_init as init_mod,
        )

        sys.modules["gymnasium"] = fake_gym
        init_mod.gym = fake_gym
        init_mod.MYOSUITE_AVAILABLE = True
        try:
            with pytest.raises(ValueError):
                engine.load_from_path("bad-env")
            assert engine.env is None
        finally:
            init_mod.MYOSUITE_AVAILABLE = False
            sys.modules.pop("gymnasium", None)


class TestSimulationCoreMixin:
    def test_reset_calls_env_reset(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine._terminated = True
        loaded_engine.reset()
        loaded_engine.env.reset.assert_called_once()
        assert loaded_engine._terminated is False

    def test_step_calls_env_step(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine.step()
        loaded_engine.env.step.assert_called_once()

    def test_step_with_terminated_warns_and_returns(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine._terminated = True
        loaded_engine.step()
        loaded_engine.env.step.assert_not_called()

    def test_step_sets_terminated(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine.env.step.return_value = (None, 0.0, True, False, {})
        loaded_engine.step()
        assert loaded_engine._terminated is True

    def test_step_dt_mismatch_warns_but_proceeds(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine.step(dt=0.5)
        loaded_engine.env.step.assert_called_once()

    def test_forward_calls_sim(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine.forward()
        loaded_engine.sim.forward.assert_called()

    def test_get_state_empty_uninit(self, engine: MyoSuitePhysicsEngine) -> None:
        q, v = engine.get_state()
        assert q.size == 0 and v.size == 0

    def test_get_state_returns_arrays(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine.sim.data.qpos = np.array([0.1, 0.2])
        loaded_engine.sim.data.qvel = np.array([0.3, 0.4])
        q, v = loaded_engine.get_state()
        assert np.allclose(q, [0.1, 0.2])
        assert np.allclose(v, [0.3, 0.4])

    def test_set_state_none_raises(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        with pytest.raises(ValueError):
            loaded_engine.set_state(None, np.zeros(2))  # type: ignore[arg-type]

    def test_set_state_writes_qpos_qvel(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine.set_state(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        assert np.allclose(loaded_engine.sim.data.qpos, [0.1, 0.2])
        assert np.allclose(loaded_engine.sim.data.qvel, [0.3, 0.4])

    def test_set_control_none_raises(self, engine: MyoSuitePhysicsEngine) -> None:
        with pytest.raises(ValueError):
            engine.set_control(None)  # type: ignore[arg-type]

    def test_set_control_caches_last_action(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        engine.set_control(np.array([0.5, 0.6]))
        assert np.allclose(engine._last_action, [0.5, 0.6])

    def test_set_control_writes_to_sim(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine.set_control(np.array([0.7, 0.8]))
        assert np.allclose(loaded_engine.sim.data.ctrl, [0.7, 0.8])

    def test_get_time_uses_sim(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine.sim.data.time = 1.25
        assert loaded_engine.get_time() == 1.25

    def test_get_time_zero_uninit(self, engine: MyoSuitePhysicsEngine) -> None:
        assert engine.get_time() == 0.0


class TestDynamicsMixin:
    def test_compute_mass_matrix_uninit_raises(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(PreconditionError):
            engine.compute_mass_matrix()

    def test_compute_mass_matrix_with_mujoco(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        mj = types.ModuleType("mujoco")
        mj.mj_fullM = MagicMock(side_effect=TypeError("fallback"))
        sys.modules["mujoco"] = mj
        try:
            out = loaded_engine.compute_mass_matrix()
            assert out.shape == (2, 2)
            assert np.allclose(out, np.eye(2))
        finally:
            sys.modules.pop("mujoco", None)

    def test_compute_bias_forces(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine.sim.data.qfrc_bias = np.array([1.0, 2.0])
        out = loaded_engine.compute_bias_forces()
        assert np.allclose(out, [1.0, 2.0])

    def test_compute_gravity_forces_returns_empty(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        assert loaded_engine.compute_gravity_forces().size == 0

    def test_compute_inverse_dynamics_uninit_raises(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(PreconditionError):
            engine.compute_inverse_dynamics(np.zeros(2))

    def test_compute_inverse_dynamics_none_raises(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(ValueError):
            loaded_engine.compute_inverse_dynamics(None)  # type: ignore[arg-type]

    def test_compute_jacobian_none_body_raises(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(ValueError):
            loaded_engine.compute_jacobian(None)  # type: ignore[arg-type]

    def test_compute_jacobian_unknown_body_returns_none(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        mj = types.ModuleType("mujoco")
        mj.mjtObj = types.SimpleNamespace(mjOBJ_BODY=1)
        mj.mj_name2id = MagicMock(return_value=-1)
        sys.modules["mujoco"] = mj
        try:
            out = loaded_engine.compute_jacobian("missing")
            assert out is None
        finally:
            sys.modules.pop("mujoco", None)

    def test_compute_jacobian_valid(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        mj = types.ModuleType("mujoco")
        mj.mjtObj = types.SimpleNamespace(mjOBJ_BODY=1)
        mj.mj_name2id = MagicMock(return_value=2)
        mj.mj_jacBody = MagicMock()
        sys.modules["mujoco"] = mj
        try:
            out = loaded_engine.compute_jacobian("link")
            assert set(out.keys()) == {"linear", "angular", "spatial"}
            assert out["spatial"].shape == (6, 2)
        finally:
            sys.modules.pop("mujoco", None)


class TestDriftControlMixin:
    def test_compute_drift_acceleration(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        # ctrl has .copy() method on real arrays; set to array
        loaded_engine.sim.data.ctrl = np.array([0.5, 0.5])
        loaded_engine.sim.data.qacc = np.array([1.0, 2.0])
        out = loaded_engine.compute_drift_acceleration()
        assert np.allclose(out, [1.0, 2.0])

    def test_compute_control_acceleration_none_raises(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(ValueError):
            loaded_engine.compute_control_acceleration(None)  # type: ignore[arg-type]

    def test_compute_control_acceleration_with_mocked_mass_matrix(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine.compute_mass_matrix = MagicMock(return_value=np.eye(2))
        out = loaded_engine.compute_control_acceleration(np.array([1.0, 2.0]))
        assert np.allclose(out, [1.0, 2.0])

    def test_compute_control_acceleration_empty_mass(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine.compute_mass_matrix = MagicMock(return_value=np.array([]))
        out = loaded_engine.compute_control_acceleration(np.array([1.0, 2.0]))
        assert out.size == 0

    def test_compute_ztcf_restores_state(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        loaded_engine.sim.data.ctrl = np.array([0.5, 0.5])
        loaded_engine.sim.data.qpos = np.array([0.0, 0.0])
        loaded_engine.sim.data.qvel = np.array([0.0, 0.0])
        loaded_engine.sim.data.qacc = np.array([3.0, 4.0])
        out = loaded_engine.compute_ztcf(np.array([1.0, 1.0]), np.array([0.5, 0.5]))
        assert out.shape == (2,)

    def test_compute_ztcf_none_q_raises(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(ValueError):
            loaded_engine.compute_ztcf(None, np.zeros(2))  # type: ignore[arg-type]

    def test_compute_zvcf(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine.sim.data.qpos = np.array([0.0, 0.0])
        loaded_engine.sim.data.qvel = np.array([0.0, 0.0])
        loaded_engine.sim.data.qacc = np.array([1.0, 2.0])
        out = loaded_engine.compute_zvcf(np.array([0.5, 0.5]))
        assert out.shape == (2,)

    def test_compute_zvcf_none_raises(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(ValueError):
            loaded_engine.compute_zvcf(None)  # type: ignore[arg-type]

    def test_get_acceleration(self, loaded_engine: MyoSuitePhysicsEngine) -> None:
        loaded_engine.sim.data.qacc = np.array([1.0, 2.0])
        assert np.allclose(loaded_engine.get_acceleration(), [1.0, 2.0])

    def test_get_acceleration_uninit_empty(self, engine: MyoSuitePhysicsEngine) -> None:
        assert engine.get_acceleration().size == 0


class TestMuscleInterfaceMixin:
    def test_get_muscle_analyzer_uninit_returns_none(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        assert engine.get_muscle_analyzer() is None

    def test_create_grip_model_uninit(self, engine: MyoSuitePhysicsEngine) -> None:
        assert engine.create_grip_model() is None

    def test_set_muscle_activations_none_raises(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        with pytest.raises(ValueError):
            engine.set_muscle_activations(None)  # type: ignore[arg-type]

    def test_set_muscle_activations_no_analyzer_is_noop(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        engine.set_muscle_activations({"a": 0.5})  # no-op without sim

    def test_compute_muscle_induced_accelerations_uninit(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        assert engine.compute_muscle_induced_accelerations() == {}

    def test_analyze_muscle_contributions_uninit(
        self, engine: MyoSuitePhysicsEngine
    ) -> None:
        assert engine.analyze_muscle_contributions() is None

    def test_get_muscle_state_uninit(self, engine: MyoSuitePhysicsEngine) -> None:
        assert engine.get_muscle_state() is None

    def test_get_muscle_names_uninit(self, engine: MyoSuitePhysicsEngine) -> None:
        assert engine.get_muscle_names() == []

    def test_set_muscle_activations_clamps_and_writes(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        analyzer = MagicMock()
        analyzer.muscle_names = ["m1"]
        analyzer.muscle_actuator_ids = [0]
        loaded_engine.get_muscle_analyzer = MagicMock(return_value=analyzer)
        loaded_engine.sim.data.ctrl = np.array([0.0, 0.0])
        loaded_engine.set_muscle_activations({"m1": 2.0})  # over-clamped to 1.0
        assert loaded_engine.sim.data.ctrl[0] == 1.0

    def test_set_muscle_activations_unknown_muscle(
        self, loaded_engine: MyoSuitePhysicsEngine
    ) -> None:
        analyzer = MagicMock()
        analyzer.muscle_names = ["m1"]
        analyzer.muscle_actuator_ids = [0]
        loaded_engine.get_muscle_analyzer = MagicMock(return_value=analyzer)
        # Should not raise — just warn.
        loaded_engine.set_muscle_activations({"missing": 0.5})
