"""Unit tests for kinematic_forces.py."""

from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
    KinematicForceAnalyzer,
    KinematicForceData,
    MjDataContext,
)


@pytest.fixture
def mock_model():
    model = MagicMock(spec=mujoco.MjModel)
    model.nv = 10
    model.nq = 11
    model.nbody = 5
    model.opt.timestep = 0.01
    return model


@pytest.fixture
def mock_data():
    data = MagicMock(spec=mujoco.MjData)
    data.qpos = np.zeros(11)
    data.qvel = np.zeros(10)
    data.qacc = np.zeros(10)
    return data


def test_kfa_init(mock_model, mock_data):
    """Test analyzer initialization."""
    with (
        patch("mujoco.mj_jacBody") as mock_jacBody,
        patch("mujoco.mj_id2name", return_value="body"),
        patch("mujoco.MjData") as mock_mjdata,
    ):
        mock_mjdata.return_value = MagicMock()
        analyzer = KinematicForceAnalyzer(mock_model, mock_data)
        assert analyzer.model == mock_model
        assert analyzer.data == mock_data
        assert analyzer.nv == 10
        mock_jacBody.assert_called_once()


def test_mj_data_context():
    """Test the data context block."""
    model = MagicMock()
    data = MagicMock()
    context = MjDataContext(model, data)
    assert context.model == model
    assert context.data == data


def test_kinematic_force_data():
    """Test data class initialization."""
    data = KinematicForceData(
        time=1.0,
        coriolis_forces=np.ones(10),
        gravity_forces=np.ones(10) * 2,
    )

    assert data.time == 1.0
    assert np.allclose(data.gravity_forces, 2.0)
