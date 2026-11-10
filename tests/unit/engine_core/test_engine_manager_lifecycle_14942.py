"""EngineManager lifecycle: reload loaded engines and cleanup (#14942)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_core.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
    GolfModelingError,
)
from src.shared.python.engine_core.workflow_adapter import EngineWorkflowAdapter

pytestmark = pytest.mark.unit


@pytest.fixture
def manager() -> EngineManager:
    with patch("src.shared.python.data_io.common_utils.setup_structured_logging"):
        with patch.object(EngineManager, "_discover_engines"):
            return EngineManager(Path("/mock/root"))


def test_switch_engine_reactivates_loaded_engine_without_reload(
    manager: EngineManager,
) -> None:
    engine_a = MagicMock(name="engine_a")
    engine_b = MagicMock(name="engine_b")
    manager._loaded_engines[EngineType.MUJOCO] = engine_a
    manager._loaded_engines[EngineType.DRAKE] = engine_b
    manager.engine_status[EngineType.MUJOCO] = EngineStatus.LOADED
    manager.engine_status[EngineType.DRAKE] = EngineStatus.LOADED
    manager.current_engine = EngineType.DRAKE
    manager.active_physics_engine = engine_b

    with patch.object(manager, "_load_engine") as mock_load:
        assert manager.switch_engine(EngineType.MUJOCO) is True
        mock_load.assert_not_called()

    assert manager.current_engine == EngineType.MUJOCO
    assert manager.active_physics_engine is engine_a


def test_switch_engine_retries_from_error_and_keeps_last_error(
    manager: EngineManager,
) -> None:
    manager.engine_status[EngineType.MUJOCO] = EngineStatus.ERROR
    manager._last_engine_errors[EngineType.MUJOCO] = "first failure"

    with patch.object(
        manager,
        "_load_engine",
        side_effect=[GolfModelingError("still broken")],
    ):
        assert manager.switch_engine(EngineType.MUJOCO) is False
    assert manager.engine_status[EngineType.MUJOCO] == EngineStatus.ERROR
    assert manager._last_engine_errors[EngineType.MUJOCO] == "still broken"

    with patch.object(manager, "_load_engine", return_value=MagicMock()) as mock_load:
        assert manager.switch_engine(EngineType.MUJOCO) is True
        mock_load.assert_called_once_with(EngineType.MUJOCO)
    assert EngineType.MUJOCO not in manager._last_engine_errors


def test_load_sequence_a_b_a_succeeds(manager: EngineManager) -> None:
    engine_a = MagicMock(name="engine_a")
    engine_b = MagicMock(name="engine_b")
    manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE
    manager.engine_status[EngineType.DRAKE] = EngineStatus.AVAILABLE

    with patch.object(
        manager,
        "_load_engine",
        side_effect=[engine_a, engine_b],
    ) as mock_load:
        assert manager.switch_engine(EngineType.MUJOCO) is True
        assert manager.switch_engine(EngineType.DRAKE) is True
        manager.engine_status[EngineType.MUJOCO] = EngineStatus.LOADED
        manager._loaded_engines[EngineType.MUJOCO] = engine_a
        manager.engine_status[EngineType.DRAKE] = EngineStatus.LOADED
        manager._loaded_engines[EngineType.DRAKE] = engine_b
        assert manager.switch_engine(EngineType.MUJOCO) is True

    assert mock_load.call_count == 2


def test_cleanup_returns_loaded_engine_to_available(manager: EngineManager) -> None:
    active = MagicMock(name="active_engine")
    active.close = MagicMock()
    manager.active_physics_engine = active
    manager.current_engine = EngineType.MUJOCO
    manager.engine_status[EngineType.MUJOCO] = EngineStatus.LOADED
    manager._loaded_engines[EngineType.MUJOCO] = active

    manager.cleanup()

    assert manager.engine_status[EngineType.MUJOCO] == EngineStatus.AVAILABLE
    assert manager.current_engine is None
    assert EngineType.MUJOCO not in manager._loaded_engines


def test_probe_reports_loaded_engine_as_available() -> None:
    manager = MagicMock()
    manager.get_engine_status.return_value = EngineStatus.LOADED
    adapter = EngineWorkflowAdapter(manager)

    result = adapter.probe("mujoco")

    assert result.ok is True
    assert result.payload["available"] is True
