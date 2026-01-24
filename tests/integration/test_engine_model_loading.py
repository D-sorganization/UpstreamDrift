from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
)


@pytest.fixture
def mock_engine_manager():
    """Fixture to provide EngineManager with mocked root."""
    return EngineManager(Path("/mock/suite/root"))


@patch("src.shared.python.engine_probes.MuJoCoProbe.probe")
@patch(
    "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.MuJoCoPhysicsEngine"
)
def test_mujoco_loads_default_model(
    mock_mujoco_engine_cls, mock_probe, mock_engine_manager
):
    """Test that MuJoCo engine loads the default model if present."""
    # Setup probe
    mock_probe.return_value.is_available.return_value = True

    # Setup Engine Mock
    mock_engine_instance = mock_mujoco_engine_cls.return_value

    # Force engine availability
    mock_engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

    # Mock file existence for the model
    with patch("pathlib.Path.exists", return_value=True):
        with patch.dict("sys.modules", {"mujoco": MagicMock()}):
            mock_engine_manager.switch_engine(EngineType.MUJOCO)

            # Verify load_from_path was called
            mock_engine_instance.load_from_path.assert_called()
            # Verify it was called with something ending in simple_pendulum.xml
            args, _ = mock_engine_instance.load_from_path.call_args
            assert str(args[0]).endswith("simple_pendulum.xml")


@patch("src.shared.python.engine_probes.PinocchioProbe.probe")
@patch(
    "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine.PinocchioPhysicsEngine"
)
def test_pinocchio_loads_default_model(
    mock_pin_engine_cls, mock_probe, mock_engine_manager
):
    """Test that Pinocchio engine loads the default model if present."""
    mock_probe.return_value.is_available.return_value = True
    mock_engine_instance = mock_pin_engine_cls.return_value
    mock_engine_manager.engine_status[EngineType.PINOCCHIO] = EngineStatus.AVAILABLE

    with patch("pathlib.Path.exists", return_value=True):
        with patch.dict("sys.modules", {"pinocchio": MagicMock()}):
            mock_engine_manager.switch_engine(EngineType.PINOCCHIO)

            mock_engine_instance.load_from_path.assert_called()
            args, _ = mock_engine_instance.load_from_path.call_args
            assert str(args[0]).endswith("golfer.urdf")


@patch("src.shared.python.engine_probes.DrakeProbe.probe")
@patch("src.engines.physics_engines.drake.python.drake_physics_engine.DrakePhysicsEngine")
def test_drake_loads_default_model(
    mock_drake_engine_cls, mock_probe, mock_engine_manager
):
    """Test that Drake engine attempts to load the shared URDF."""
    mock_probe.return_value.is_available.return_value = True
    mock_engine_instance = mock_drake_engine_cls.return_value
    mock_engine_manager.engine_status[EngineType.DRAKE] = EngineStatus.AVAILABLE

    with patch("pathlib.Path.exists", return_value=True):
        with patch.dict("sys.modules", {"pydrake": MagicMock()}):
            mock_engine_manager.switch_engine(EngineType.DRAKE)

            mock_engine_instance.load_from_path.assert_called()
            args, _ = mock_engine_instance.load_from_path.call_args
            assert str(args[0]).endswith("golfer.urdf")
