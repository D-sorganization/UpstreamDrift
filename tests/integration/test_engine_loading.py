from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.data_io.path_utils import get_src_root
from src.shared.python.engine_core.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
)
from src.shared.python.engine_core.engine_registry import EngineRegistry
from src.shared.python.engine_core.interfaces import PhysicsEngine

_REGISTRATION_SPEC_ATTRS = [
    "engine_type",
    "factory",
    "registration_path",
    "requires_binary",
    "probe_class",
]


@pytest.fixture
def mock_engine_manager():
    """Fixture to provide EngineManager with actual repo root to pass security validation."""
    # Use actual src root so paths pass security validation checks
    return EngineManager(get_src_root())


def test_engine_initialization(mock_engine_manager):
    """Test that EngineManager initializes correctly."""
    assert mock_engine_manager.current_engine is None
    # engine_status might be all UNAVAILABLE if paths don't exist


@pytest.mark.parametrize(
    "engine_type",
    [EngineType.MUJOCO, EngineType.DRAKE, EngineType.PINOCCHIO],
    ids=["mujoco", "drake", "pinocchio"],
)
def test_engine_loading_success(mock_engine_manager, engine_type):
    """Test successful engine loading via registry factory mock."""
    # Force engine availability (bypass discovery)
    mock_engine_manager.engine_status[engine_type] = EngineStatus.AVAILABLE

    # Mock the registry to return a registration with a mock factory
    mock_engine_instance = MagicMock(spec=PhysicsEngine)
    mock_registration = MagicMock(spec=_REGISTRATION_SPEC_ATTRS)
    mock_registration.factory.return_value = mock_engine_instance

    with patch(
        "src.shared.python.engine_core.engine_manager.get_registry"
    ) as mock_get_reg:
        mock_registry = MagicMock(spec=EngineRegistry)
        mock_registry.get.return_value = mock_registration
        mock_get_reg.return_value = mock_registry

        result = mock_engine_manager.switch_engine(engine_type)

        assert result is True
        assert mock_engine_manager.get_current_engine() == engine_type
        assert mock_engine_manager.engine_status[engine_type] == EngineStatus.LOADED
        assert mock_engine_manager.active_physics_engine is not None


def test_mujoco_loading_failure_no_registration(mock_engine_manager):
    """Test MuJoCo loading failure when no registration found."""
    # Force engine availability
    mock_engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

    # Mock registry returning no registration
    with patch(
        "src.shared.python.engine_core.engine_manager.get_registry"
    ) as mock_get_reg:
        mock_registry = MagicMock(spec=EngineRegistry)
        mock_registry.get.return_value = None
        mock_get_reg.return_value = mock_registry

        result = mock_engine_manager.switch_engine(EngineType.MUJOCO)

        assert result is False
        assert mock_engine_manager.get_current_engine() is None


def test_cleanup_releases_resources(mock_engine_manager):
    """Test that cleanup releases resources."""
    # Mock some loaded resources
    mock_matlab = MagicMock(spec=["quit", "exit"])
    mock_engine_manager._matlab_engine = mock_matlab

    mock_engine_manager.cleanup()

    mock_matlab.quit.assert_called_once()
    assert mock_engine_manager._matlab_engine is None
    # Verify cleanup completed successfully
    assert mock_engine_manager.active_physics_engine is None
    assert mock_engine_manager.current_engine is None


def test_cleanup_handles_exceptions(mock_engine_manager):
    """Test that cleanup handles exceptions during shutdown."""
    mock_matlab = MagicMock(spec=["quit", "exit"])
    mock_matlab.quit.side_effect = Exception("Shutdown error")
    mock_engine_manager._matlab_engine = mock_matlab

    # Should not raise
    mock_engine_manager.cleanup()

    assert mock_engine_manager._matlab_engine is None
