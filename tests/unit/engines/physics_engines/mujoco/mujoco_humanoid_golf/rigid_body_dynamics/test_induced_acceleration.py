"""Unit tests for induced_acceleration.py."""

from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.induced_acceleration import (
    MuJoCoInducedAccelerationAnalyzer,
)


@pytest.fixture
def mock_model():
    """Mock MjModel."""
    model = MagicMock(spec=mujoco.MjModel)
    model.nv = 3
    model.nu = 2
    return model


@pytest.fixture
def mock_data():
    """Mock MjData."""
    data = MagicMock(spec=mujoco.MjData)
    data.qvel = np.array([1.0, 2.0, 3.0])
    data.cvel = np.ones((10, 6))
    data.qfrc_bias = np.array([10.0, 20.0, 30.0])
    data.qfrc_actuator = np.array([5.0, 6.0, 7.0])
    data.qfrc_constraint = np.array([1.0, 1.0, 1.0])
    data.qM = np.ones(10)
    data.cacc = np.zeros((10, 6))
    data.cacc[5] = np.array([0, 0, 0, 10, 20, 30])  # Local acc for body 5

    # 3x3 identity for body 5 xmat
    xmat = np.eye(3).flatten()
    data.xmat = np.zeros((10, 9))
    data.xmat[5] = xmat

    return data


@patch("mujoco.mj_fullM")
@patch("mujoco.mj_rne")
@patch("mujoco.mj_forward")
@patch("numpy.linalg.solve")
def test_compute_components(
    mock_solve, mock_forward, mock_rne, mock_fullM, mock_model, mock_data
):
    """Test compute_components."""
    analyzer = MuJoCoInducedAccelerationAnalyzer(mock_model, mock_data)

    # Setup mocks
    mock_solve.return_value = np.zeros((3, 4))

    def side_effect_rne(m, d, flg, term_G):
        term_G[:] = np.array([1.0, 2.0, 3.0])

    mock_rne.side_effect = side_effect_rne

    result = analyzer.compute_components()

    # Assertions
    mock_fullM.assert_called_once()
    mock_rne.assert_called_once()
    mock_forward.assert_called_once()
    mock_solve.assert_called_once()

    assert "gravity" in result
    assert "velocity" in result
    assert "control" in result
    assert "constraint" in result
    assert "total" in result

    assert result["total"].shape == (3,)


@patch("mujoco.mj_name2id")
@patch("mujoco.mj_jacBody")
def test_compute_task_space_components(
    mock_jacBody, mock_name2id, mock_model, mock_data
):
    """Test compute_task_space_components."""
    analyzer = MuJoCoInducedAccelerationAnalyzer(mock_model, mock_data)

    mock_name2id.return_value = 5

    def side_effect_jac(m, d, jacp, jacr, body_id):
        jacp[:] = np.eye(3)

    mock_jacBody.side_effect = side_effect_jac

    qdd_comps = {
        "gravity": np.array([0.1, 0.2, 0.3]),
        "velocity": np.array([1.0, 1.0, 1.0]),
        "control": np.array([2.0, 2.0, 2.0]),
        "constraint": np.array([0.0, 0.0, 0.0]),
        "total": np.array([3.1, 3.2, 3.3]),
    }

    result = analyzer.compute_task_space_components("test_body", qdd_comps=qdd_comps)

    mock_name2id.assert_called_once_with(
        mock_model, mujoco.mjtObj.mjOBJ_BODY, "test_body"
    )
    mock_jacBody.assert_called_once()

    assert result is not None
    assert "gravity" in result
    assert "total" in result
    # Total actual world should be [10, 20, 30] from cacc[5][3:6]
    np.testing.assert_array_equal(result["total"], np.array([10.0, 20.0, 30.0]))


@patch("mujoco.mj_name2id")
def test_compute_task_space_invalid_body(mock_name2id, mock_model, mock_data):
    """Test compute_task_space_components with invalid body."""
    analyzer = MuJoCoInducedAccelerationAnalyzer(mock_model, mock_data)

    mock_name2id.return_value = -1
    result = analyzer.compute_task_space_components("invalid_body")

    assert result is None
