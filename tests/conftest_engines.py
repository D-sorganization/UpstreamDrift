"""Shared fixtures for mocking physics engines.

Consolidates mock engine setup patterns used across unit, integration, and parity tests.
"""

from unittest.mock import MagicMock, patch

import pytest


# Mock classes that need to be defined before importing the engine
class MockPhysicsEngine:
    pass


@pytest.fixture
def mock_drake_dependencies():
    """Fixture to mock pydrake and interfaces safely.

    This fixture mocks pydrake modules to allow testing Drake integration
    without having Drake installed.
    """
    mock_pydrake = MagicMock()
    mock_interfaces = MagicMock()
    mock_interfaces.PhysicsEngine = MockPhysicsEngine

    with patch.dict(
        "sys.modules",
        {
            "pydrake": mock_pydrake,
            "pydrake.math": MagicMock(),
            "pydrake.multibody": MagicMock(),
            "pydrake.multibody.plant": MagicMock(),
            "pydrake.multibody.parsing": MagicMock(),
            "pydrake.systems": MagicMock(),
            "pydrake.systems.framework": MagicMock(),
            "pydrake.systems.analysis": MagicMock(),
            "pydrake.all": MagicMock(),
            "shared.python.interfaces": mock_interfaces,
        },
    ):
        yield mock_pydrake, mock_interfaces


@pytest.fixture
def mock_mujoco_dependencies():
    """Fixture to mock mujoco and interfaces safely.

    This fixture mocks mujoco modules to allow testing MuJoCo integration
    without having MuJoCo installed.
    """
    mock_mujoco = MagicMock()
    mock_interfaces = MagicMock()
    mock_interfaces.PhysicsEngine = MockPhysicsEngine

    # Create common MuJoCo structure mocks
    # These are needed for attribute access in many tests
    mock_model = MagicMock()
    mock_model.nv = 2
    mock_model.nu = 2
    mock_model.nq = 2
    mock_model.nbody = 2

    mock_data = MagicMock()
    mock_data.qpos = MagicMock()
    mock_data.qvel = MagicMock()
    mock_data.qacc = MagicMock()
    mock_data.ctrl = MagicMock()

    mock_mujoco.MjModel.return_value = mock_model
    mock_mujoco.MjData.return_value = mock_data

    with patch.dict(
        "sys.modules",
        {
            "mujoco": mock_mujoco,
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.interfaces": mock_interfaces,
        },
    ):
        yield mock_mujoco, mock_interfaces
