"""Tests for the MuJoCo humanoid_golf module."""

import json
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest


def test_pd_controller():
    """Test PDController."""
    physics_mock = MagicMock()
    physics_mock.model.nu = 5
    physics_mock.named.data.qpos = {"joint1": 0.5, "joint2": -0.2}
    physics_mock.named.data.qvel = {"joint1": 0.1, "joint2": 0.0}

    actuators = {"joint1": 1, "joint2": 3}
    target_pose = {"joint1": 1.0, "joint2": -0.5}

    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf.sim import (
            PDController,
        )

        controller = PDController(actuators, target_pose, kp=10.0, kd=2.0)
    action = controller.get_action(physics_mock)

    assert action.shape == (5,)
    # error = 1.0 - 0.5 = 0.5
    # torque = (10.0 * 0.5) - (2.0 * 0.1) = 5.0 - 0.2 = 4.8
    assert np.isclose(action[1], 4.8)

    # error = -0.5 - (-0.2) = -0.3
    # torque = (10.0 * -0.3) - (2.0 * 0.0) = -3.0
    assert np.isclose(action[3], -3.0)


def test_polynomial_controller():
    """Test PolynomialController."""
    physics_mock = MagicMock()
    physics_mock.model.nu = 5
    physics_mock.model.id2name.side_effect = lambda i, x: (
        "rhumerusrx" if i == 2 else f"j{i}"
    )
    physics_mock.data.time = 2.0

    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf.sim import (
            PolynomialController,
        )

        controller = PolynomialController(physics_mock)
    assert controller.coeffs[2, 1] == 60.0
    assert controller.coeffs[2, 3] == -20.0

    action = controller.get_action(physics_mock)
    assert action.shape == (5,)
    # At t=2.0: 60*2 - 20*(2^3) = 120 - 160 = -40
    assert np.isclose(action[2], -40.0)


def test_lqr_controller():
    """Test LQRController."""
    physics_mock = MagicMock()
    physics_mock.model.nu = 2
    physics_mock.model.nq = 3
    physics_mock.model.nv = 3
    physics_mock.data.qpos = np.array([1.0, 2.0, 3.0])
    physics_mock.data.qvel = np.array([0.1, 0.2, 0.3])
    physics_mock.model.actuator_trnid = np.array([[0, 0], [1, 0]])
    physics_mock.model.jnt_qposadr = {0: 0, 1: 1}
    physics_mock.model.jnt_dofadr = {0: 0, 1: 1}

    actuators = {"j1": 0, "j2": 1}
    target_pose = {"j1": 0.0}

    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf.sim import (
            LQRController,
        )

        controller = LQRController(physics_mock, target_pose, actuators)
    assert controller.K.shape == (2, 6)

    action = controller.get_action(physics_mock)
    assert action.shape == (2,)


def test_timestep():
    """Test TimeStep."""
    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf.sim import (
            TimeStep,
        )

        ts_first = TimeStep(0)
    assert ts_first.first() is True
    assert ts_first.mid() is False
    assert ts_first.last() is False

    ts_mid = TimeStep(1)
    assert ts_mid.first() is False
    assert ts_mid.mid() is True
    assert ts_mid.last() is False


def test_physics_env_wrapper():
    """Test PhysicsEnvWrapper."""
    physics_mock = MagicMock()
    physics_mock.model.nu = 5

    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf.sim import (
            PhysicsEnvWrapper,
        )

        wrapper = PhysicsEnvWrapper(physics_mock)
    assert wrapper.physics == physics_mock

    spec = wrapper.action_spec()
    assert spec.shape == (5,)

    wrapper.step(np.zeros(5))
    physics_mock.set_control.assert_called_once()
    physics_mock.step.assert_called_once()

    wrapper.reset()
    physics_mock.reset.assert_called_once()


def test_save_load_state():
    """Test save and load state."""
    physics_mock = MagicMock()
    physics_mock.get_state.return_value = np.array([1.0, 2.0])

    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf.sim import (
            save_state,
            load_state,
        )

        with patch("builtins.open", mock_open()) as mock_file:
            save_state(physics_mock, "test.json")
        mock_file.assert_called_once_with("test.json", "w")

    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="[1.0, 2.0]")),
    ):
        load_state(physics_mock, "test.json")
        physics_mock.set_state.assert_called_once()


def test_extract_simulation_params():
    """Test _extract_simulation_params."""
    config = {
        "control_mode": "lqr",
        "live_view": True,
        "club_length": 1.2,
    }

    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf.sim import (
            _extract_simulation_params,
        )

        with patch.dict("os.environ", {"MUJOCO_GL": ""}):
            params = _extract_simulation_params(config, 5.0)
        assert params["control_mode"] == "lqr"
        assert params["use_viewer"] is True
        assert params["duration"] == 5.0
        assert params["club_params"]["length"] == 1.2


def test_iaa_helper():
    """Test iaa_helper."""
    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf import (
            iaa_helper,
        )

        physics_mock = MagicMock()

    # Just verify that when mjlib is not present it returns empty dict gracefully
    # If mjlib is missing, it returns empty dict
    with patch.dict("sys.modules", {"dm_control.mujoco.wrapper.mjbindings": None}):
        res = iaa_helper.compute_induced_accelerations(physics_mock)
        assert res == {}

        res = iaa_helper.compute_counterfactuals(physics_mock)
        assert res == {}

        res = iaa_helper.get_mass_matrix(physics_mock)
        assert res is None


def test_visualization():
    """Test visualization tools."""
    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf import (
            visualization,
        )

        tracer = visualization.TrajectoryTracer(max_points=5)
    tracer.add_point("hand", np.array([1, 2, 3]))
    tracer.add_point("hand", np.array([4, 5, 6]))

    trace = tracer.get_trace("hand")
    assert len(trace) == 2
    assert np.allclose(trace[0], [1, 2, 3])

    tracer.set_desired_trajectory("hand", [np.array([1, 1, 1])])
    assert len(tracer.get_desired_trace("hand")) == 1

    tracer.clear("hand")
    assert len(tracer.get_trace("hand")) == 0

    # Test arrow factory
    arrow = visualization.create_force_arrow_geom(
        np.array([0, 0, 0]), np.array([1, 0, 0]), 10.0, [1, 0, 0, 1]
    )
    assert arrow["type"] == "arrow"

    # Test line factory
    lines = visualization.create_trace_line_geom(
        [np.array([0, 0, 0]), np.array([1, 1, 1])], [1, 0, 0, 1]
    )
    assert len(lines) == 1
    assert lines[0]["type"] == "line"


def test_utils():
    """Test utils module."""
    with patch.dict(
        sys.modules,
        {
            "dm_control": MagicMock(),
            "dm_control.suite": MagicMock(),
            "dm_control.mjcf": MagicMock(),
            "mujoco": MagicMock(),
        },
    ):
        from src.engines.physics_engines.mujoco.docker.src.humanoid_golf import utils

        physics_mock = MagicMock()
    physics_mock.model.nu = 2
    physics_mock.model.id2name.side_effect = lambda i, type: f"act{i}"

    mapping = utils.get_actuator_indices(physics_mock)
    assert mapping == {"act0": 0, "act1": 1}

    # Test customize_visuals
    physics_mock.model.ngeom = 2
    physics_mock.model.id2name.side_effect = lambda i, t: (
        "left_eye" if i == 0 else "torso"
    )
    utils.customize_visuals(physics_mock, {"colors": {"eyes": [0.1, 0.2, 0.3, 1.0]}})
    # No assert needed, just ensuring it doesn't crash on mocked setup
