"""Tests for shared engine workflow adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.engine_core.workflow_adapter import EngineWorkflowAdapter


def test_probe_returns_unknown_for_invalid_engine_name() -> None:
    manager = MagicMock()
    adapter = EngineWorkflowAdapter(manager)

    result = adapter.probe("invalid")

    assert result.ok is False
    assert result.payload["available"] is False
    assert "Unknown engine" in result.payload["error"]


def test_load_success_uses_engine_manager_switch_engine() -> None:
    manager = MagicMock()
    manager.switch_engine.return_value = True
    adapter = EngineWorkflowAdapter(manager)

    result = adapter.load("mujoco")

    manager.switch_engine.assert_called_once_with(EngineType.MUJOCO)
    assert result.ok is True
    assert result.payload["status"] == "loaded"


def test_unload_cleans_up_when_target_is_current_engine() -> None:
    manager = MagicMock()
    manager.get_current_engine.return_value = EngineType.MUJOCO
    adapter = EngineWorkflowAdapter(manager)

    result = adapter.unload("mujoco")

    manager.cleanup.assert_called_once()
    assert result.ok is True
    assert result.payload["status"] == "unloaded"
