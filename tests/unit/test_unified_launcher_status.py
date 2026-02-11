"""Tests for unified launcher status output."""

from __future__ import annotations

import logging

from src.launchers.unified_launcher import UnifiedLauncher


class _FakeEngineType:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeEngineManager:
    def get_available_engines(self) -> list[_FakeEngineType]:
        return [_FakeEngineType("mujoco"), _FakeEngineType("drake")]


def test_show_status_logs_available_engines(monkeypatch, caplog) -> None:
    """show_status should log available engines and paths."""
    monkeypatch.setattr(
        "src.shared.python.engine_core.engine_manager.EngineManager",
        _FakeEngineManager,
    )

    caplog.set_level(logging.INFO)

    launcher = UnifiedLauncher.__new__(UnifiedLauncher)
    launcher.show_status()

    messages = [record.message for record in caplog.records]
    assert any("Available engines" in message for message in messages)
    assert any("mujoco" in message for message in messages)
    assert any("drake" in message for message in messages)
