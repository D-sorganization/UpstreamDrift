"""Tests for the MemoryPanel widget (user_memory.json viewer/editor)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.launchers.chat_history.memory_panel import MemoryPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def adapter(tmp_path) -> MagicMock:
    memory_file = tmp_path / "user_memory.json"
    memory_file.write_text(
        json.dumps(
            {
                "identity": {"name": "Alice", "role": "researcher"},
                "preferences": {"theme": "dark"},
                "projects": {"current": "UpstreamDrift"},
                "knowledge": {"facts": ["uses Pinocchio"]},
            }
        ),
        encoding="utf-8",
    )
    a = MagicMock()
    a.load_memory.return_value = {
        "identity": {"name": "Alice", "role": "researcher"},
        "preferences": {"theme": "dark"},
        "projects": {"current": "UpstreamDrift"},
        "knowledge": {"facts": ["uses Pinocchio"]},
    }
    a.memory_path = memory_file
    return a


def test_panel_renders_four_structured_sections(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    assert panel.has_section("identity")
    assert panel.has_section("preferences")
    assert panel.has_section("projects")
    assert panel.has_section("knowledge")


def test_panel_loads_existing_values_into_fields(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    assert panel.section_text("identity") != ""
    assert "Alice" in panel.section_text("identity")


def test_panel_save_round_trip_persists(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    panel.set_section_text("preferences", json.dumps({"theme": "light", "lang": "en"}))
    panel.save()
    adapter.save_memory.assert_called_once()
    saved = adapter.save_memory.call_args[0][0]
    assert saved["preferences"]["theme"] == "light"
    assert saved["preferences"]["lang"] == "en"


def test_save_rejects_invalid_json(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    panel.set_section_text("identity", "{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        panel.save()


def test_save_rejects_non_object_section(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    panel.set_section_text("identity", "[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        panel.save()


def test_reset_calls_adapter(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    panel.reset()
    adapter.reset_memory.assert_called_once()


def test_panel_handles_missing_memory_gracefully(qapp) -> None:
    a = MagicMock()
    a.load_memory.return_value = {
        "identity": {},
        "preferences": {},
        "projects": {},
        "knowledge": {},
    }
    panel = MemoryPanel(a)
    panel.refresh()
    for key in ("identity", "preferences", "projects", "knowledge"):
        # Empty dict renders as "{}".
        assert panel.section_text(key).strip() in ("{}", "{\n}")
