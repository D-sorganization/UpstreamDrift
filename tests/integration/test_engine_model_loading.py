from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.data_io.path_utils import get_src_root
from src.shared.python.engine_core.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
)


@pytest.fixture
def mock_engine_manager():
    """Fixture to provide EngineManager with actual repo root to pass security validation."""
    # Use actual src root so paths pass security validation checks
    return EngineManager(get_src_root())


def test_mujoco_loads_default_model(mock_engine_manager):
    """Test that MuJoCo engine loads via registry factory.

    The engine manager uses registry-based loading (get_registry().get().factory()),
    so we mock the registry factory to verify switch_engine succeeds.
    """
    mock_engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

    mock_engine_instance = MagicMock()
    mock_registration = MagicMock()
    mock_registration.factory.return_value = mock_engine_instance

    with patch(
        "src.shared.python.engine_core.engine_manager.get_registry"
    ) as mock_get_reg:
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_registration
        mock_get_reg.return_value = mock_registry

        result = mock_engine_manager.switch_engine(EngineType.MUJOCO)

        assert result is True
        mock_registration.factory.assert_called_once()
        assert mock_engine_manager.active_physics_engine is mock_engine_instance


def test_pinocchio_loads_default_model(mock_engine_manager):
    """Test that Pinocchio engine loads via registry factory."""
    mock_engine_manager.engine_status[EngineType.PINOCCHIO] = EngineStatus.AVAILABLE

    mock_engine_instance = MagicMock()
    mock_registration = MagicMock()
    mock_registration.factory.return_value = mock_engine_instance

    with patch(
        "src.shared.python.engine_core.engine_manager.get_registry"
    ) as mock_get_reg:
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_registration
        mock_get_reg.return_value = mock_registry

        result = mock_engine_manager.switch_engine(EngineType.PINOCCHIO)

        assert result is True
        mock_registration.factory.assert_called_once()
        assert mock_engine_manager.active_physics_engine is mock_engine_instance


def test_drake_loads_default_model(mock_engine_manager):
    """Test that Drake engine loads via registry factory."""
    mock_engine_manager.engine_status[EngineType.DRAKE] = EngineStatus.AVAILABLE

    mock_engine_instance = MagicMock()
    mock_registration = MagicMock()
    mock_registration.factory.return_value = mock_engine_instance

    with patch(
        "src.shared.python.engine_core.engine_manager.get_registry"
    ) as mock_get_reg:
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_registration
        mock_get_reg.return_value = mock_registry

        result = mock_engine_manager.switch_engine(EngineType.DRAKE)

        assert result is True
        mock_registration.factory.assert_called_once()
        assert mock_engine_manager.active_physics_engine is mock_engine_instance
