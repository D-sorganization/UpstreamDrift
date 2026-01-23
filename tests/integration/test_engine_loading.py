import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from shared.python.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
)


@pytest.fixture
def mock_engine_manager():
    """Fixture to provide EngineManager with mocked root."""
    return EngineManager(Path("/mock/suite/root"))


def test_engine_initialization(mock_engine_manager):
    """Test that EngineManager initializes correctly."""
    assert mock_engine_manager.current_engine is None
    # engine_status might be all UNAVAILABLE if paths don't exist


@patch("shared.python.engine_probes.MuJoCoProbe.probe")
def test_mujoco_loading_success(mock_probe, mock_engine_manager):
    """Test successful MuJoCo loading."""
    # Mock probe result
    mock_probe.return_value.is_available.return_value = True

    # Force engine availability (bypass discovery)
    mock_engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

    # Mock file system checks and mujoco module using sys.modules
    mock_mujoco_pkg = MagicMock()
    mock_mujoco_pkg.__version__ = "3.2.3"
    mock_mujoco_pkg.MjModel.from_xml_path.return_value = MagicMock()

    with patch.dict("sys.modules", {"mujoco": mock_mujoco_pkg}):
        # We also need to mock the PhysicsEngine import inside switch_engine
        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.MuJoCoPhysicsEngine"
        ):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.glob", return_value=[Path("model.xml")]):
                    result = mock_engine_manager.switch_engine(EngineType.MUJOCO)

                    assert result is True
                    assert mock_engine_manager.get_current_engine() == EngineType.MUJOCO
                    # Verify that the engine was loaded successfully
                    assert (
                        mock_engine_manager.engine_status[EngineType.MUJOCO]
                        == EngineStatus.LOADED
                    )
                    assert mock_engine_manager.active_physics_engine is not None


@patch("shared.python.engine_probes.MuJoCoProbe.probe")
def test_mujoco_loading_failure_missing_dependency(mock_probe, mock_engine_manager):
    """Test MuJoCo loading failure when dependency is missing."""
    # Mock probe result failure
    mock_probe.return_value.is_available.return_value = False
    mock_probe.return_value.diagnostic_message = "MuJoCo not installed"

    # Force engine availability
    mock_engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

    result = mock_engine_manager.switch_engine(EngineType.MUJOCO)

    assert result is False
    assert mock_engine_manager.get_current_engine() is None


@patch.dict(
    "sys.modules",
    {
        "pydrake": MagicMock(),
        "pydrake.systems.framework": MagicMock(),
        "pydrake.geometry": MagicMock(),
    },
)
@patch("engines.physics_engines.drake.python.drake_physics_engine.DrakePhysicsEngine")
@patch("shared.python.engine_probes.DrakeProbe.probe")
def test_drake_loading_success(mock_probe, mock_drake_class, mock_engine_manager):
    """Test successful Drake loading."""
    mock_probe.return_value.is_available.return_value = True

    # Force engine availability
    mock_engine_manager.engine_status[EngineType.DRAKE] = EngineStatus.AVAILABLE

    mock_drake = sys.modules["pydrake"]
    mock_drake.__version__ = "1.22.0"  # type: ignore[attr-defined]

    result = mock_engine_manager.switch_engine(EngineType.DRAKE)

    assert result is True
    assert mock_engine_manager.get_current_engine() == EngineType.DRAKE
    # Verify that the engine was loaded successfully
    assert mock_engine_manager.engine_status[EngineType.DRAKE] == EngineStatus.LOADED
    assert mock_engine_manager.active_physics_engine is not None


@patch(
    "engines.physics_engines.pinocchio.python.pinocchio_physics_engine.PinocchioPhysicsEngine"
)
@patch("shared.python.engine_probes.PinocchioProbe.probe")
def test_pinocchio_loading_success(
    mock_probe, mock_pin_engine_class, mock_engine_manager
):
    """Test successful Pinocchio loading."""
    mock_probe.return_value.is_available.return_value = True

    # Force engine availability
    mock_engine_manager.engine_status[EngineType.PINOCCHIO] = EngineStatus.AVAILABLE

    mock_pin = MagicMock()
    mock_pin.__version__ = "2.6.0"

    with patch.dict("sys.modules", {"pinocchio": mock_pin}):
        with patch("pathlib.Path.exists", return_value=True):
            result = mock_engine_manager.switch_engine(EngineType.PINOCCHIO)

            assert result is True
            assert mock_engine_manager.get_current_engine() == EngineType.PINOCCHIO
            # Verify that the engine was loaded successfully
            assert (
                mock_engine_manager.engine_status[EngineType.PINOCCHIO]
                == EngineStatus.LOADED
            )
            assert mock_engine_manager.active_physics_engine is not None


def test_cleanup_releases_resources(mock_engine_manager):
    """Test that cleanup releases resources."""
    # Mock some loaded resources
    mock_matlab = MagicMock()
    mock_engine_manager._matlab_engine = mock_matlab

    mock_engine_manager.cleanup()

    mock_matlab.quit.assert_called_once()
    assert mock_engine_manager._matlab_engine is None
    # Verify cleanup completed successfully
    assert mock_engine_manager.active_physics_engine is None
    assert mock_engine_manager.current_engine is None


def test_cleanup_handles_exceptions(mock_engine_manager):
    """Test that cleanup handles exceptions during shutdown."""
    mock_matlab = MagicMock()
    mock_matlab.quit.side_effect = Exception("Shutdown error")
    mock_engine_manager._matlab_engine = mock_matlab

    # Should not raise
    mock_engine_manager.cleanup()

    assert mock_engine_manager._matlab_engine is None
