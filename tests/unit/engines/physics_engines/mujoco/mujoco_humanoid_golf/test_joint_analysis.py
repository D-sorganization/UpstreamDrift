"""Unit tests for joint_analysis.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.joint_analysis import (
    GimbalJointAnalyzer,
    UniversalJointAnalyzer,
    analyze_constraint_forces_over_time,
)


@pytest.fixture
def mock_mujoco():
    with patch(
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.joint_analysis.mj"
    ) as mock_mj:
        mock_mj.mj_name2id.side_effect = lambda model, type, name: {
            "jx": 0,
            "jy": 1,
            "jz": 2,
            "joint_1": 1,
            "joint_2": 2,
        }.get(name, -1)
        mock_mj.mjtObj.mjOBJ_JOINT = 1

        mock_mj.mjtJoint.mjJNT_HINGE = 0
        mock_mj.mjtJoint.mjJNT_SLIDE = 1
        mock_mj.mjtJoint.mjJNT_BALL = 2
        mock_mj.mjtJoint.mjJNT_FREE = 3

        yield mock_mj


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.jnt_dofadr = [0, 5, 10]
    model.jnt_type = [0, 2, 3]  # hinge, ball, free
    model.jnt_qposadr = [0, 7, 14]
    model.opt.timestep = 0.01
    return model


@pytest.fixture
def mock_data():
    data = MagicMock()
    data.qfrc_constraint = np.arange(20, dtype=np.float64)
    data.qpos = np.arange(20, dtype=np.float64) * 0.1
    data.time = 0.5
    return data


def test_universal_joint_forces(mock_mujoco, mock_model, mock_data):
    """Test getting joint constraint forces."""
    analyzer = UniversalJointAnalyzer(mock_model, mock_data)

    # Hinge (1 DOF)
    mock_model.jnt_type[1] = mock_mujoco.mjtJoint.mjJNT_HINGE
    forces = analyzer.get_joint_forces("joint_1")
    assert forces.shape == (1,)
    assert forces[0] == 5.0

    # Ball (3 DOF)
    mock_model.jnt_type[1] = mock_mujoco.mjtJoint.mjJNT_BALL
    forces = analyzer.get_joint_forces("joint_1")
    assert forces.shape == (3,)
    np.testing.assert_array_equal(forces, [5.0, 6.0, 7.0])

    # Free (6 DOF)
    mock_model.jnt_type[1] = mock_mujoco.mjtJoint.mjJNT_FREE
    forces = analyzer.get_joint_forces("joint_1")
    assert forces.shape == (6,)
    np.testing.assert_array_equal(forces, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


def test_universal_joint_forces_missing(mock_mujoco, mock_model, mock_data):
    """Test missing joint."""
    analyzer = UniversalJointAnalyzer(mock_model, mock_data)
    with pytest.raises(ValueError, match="not found"):
        analyzer.get_joint_forces("missing")


def test_universal_joint_angles(mock_mujoco, mock_model, mock_data):
    """Test getting universal joint angles."""
    analyzer = UniversalJointAnalyzer(mock_model, mock_data)

    angle1, angle2 = analyzer.get_universal_joint_angles("joint_1", "joint_2")
    assert np.isclose(angle1, 0.7)
    assert np.isclose(angle2, 1.4)


def test_torque_wobble(mock_mujoco, mock_model, mock_data):
    """Test torque wobble calculation."""
    analyzer = UniversalJointAnalyzer(mock_model, mock_data)

    # No bend
    ratio = analyzer.calculate_torque_wobble(0.5, 0.0)
    assert ratio == 1.0

    # Bend, theta=0
    ratio = analyzer.calculate_torque_wobble(0.0, np.pi / 4)
    assert np.isclose(ratio, np.cos(np.pi / 4))

    # Bend, theta=pi/2
    ratio = analyzer.calculate_torque_wobble(np.pi / 2, np.pi / 4)
    assert np.isclose(ratio, 1.0 / np.cos(np.pi / 4))


def test_analyze_torque_transmission(mock_mujoco, mock_model, mock_data):
    """Test full torque transmission analysis."""
    analyzer = UniversalJointAnalyzer(mock_model, mock_data)

    # Mock the angles so wobble changes
    analyzer.get_universal_joint_angles = MagicMock(return_value=(0.1, 0.1))

    result = analyzer.analyze_torque_transmission("joint_1", "joint_2", num_cycles=1)

    assert "angles" in result
    assert "velocity_ratios" in result
    assert "torque_ratios" in result
    assert "wobble_amplitude" in result
    assert "mean_velocity_ratio" in result
    assert len(result["angles"]) == 360


def test_gimbal_joint_angles(mock_mujoco, mock_model, mock_data):
    """Test gimbal joint angles."""
    analyzer = GimbalJointAnalyzer(mock_model, mock_data)

    x, y, z = analyzer.get_gimbal_angles("jx", "jy", "jz")
    assert np.isclose(x, 0.0)
    assert np.isclose(y, 0.7)
    assert np.isclose(z, 1.4)


def test_gimbal_lock(mock_mujoco, mock_model, mock_data):
    """Test gimbal lock detection."""
    analyzer = GimbalJointAnalyzer(mock_model, mock_data)

    # Set y to near pi/2
    mock_data.qpos[7] = np.pi / 2 - 0.01

    is_lock, dist = analyzer.check_gimbal_lock("jx", "jy", "jz", threshold=0.05)
    assert is_lock
    assert np.isclose(dist, 0.01)


def test_analyze_constraint_forces_over_time(mock_mujoco, mock_model, mock_data):
    """Test recording constraint forces over time."""
    mock_mujoco.mjtJoint.mjJNT_HINGE = 0
    mock_model.jnt_type[1] = 0

    result = analyze_constraint_forces_over_time(
        mock_model, mock_data, ["joint_1"], duration=0.03, timestep=0.01
    )

    assert "time" in result
    assert "joint_1" in result
    assert len(result["time"]) == 3
    assert len(result["joint_1"]) == 3
