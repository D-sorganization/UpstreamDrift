"""Tests that history pane per-conversation hover actions fire the right
adapter calls (Restore, Archive, Delete, Export, Load as context)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.launchers.chat_history.history_pane import HistorySidebarPane  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def fake_adapter() -> MagicMock:
    a = MagicMock()
    a.list_active.return_value = [
        {
            "id": "c1",
            "title": "Active chat",
            "timestamp": "2026-05-01T12:00:00",
            "snippet": "S",
        },
    ]
    a.list_archived.return_value = [
        {
            "id": "c2",
            "title": "Archived chat",
            "timestamp": "2026-04-01T12:00:00",
            "snippet": "T",
        },
    ]
    a.load_as_context.return_value = {"messages": [], "session_id": "c1"}
    return a


def test_restore_action_calls_adapter(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    pane.trigger_action("c2", "restore")
    fake_adapter.restore.assert_called_once_with("c2")


def test_archive_action_calls_adapter(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    pane.trigger_action("c1", "archive")
    fake_adapter.archive.assert_called_once_with("c1")


def test_delete_action_calls_adapter(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    pane.trigger_action("c1", "delete")
    fake_adapter.delete.assert_called_once_with("c1")


def test_export_action_calls_adapter(qapp, fake_adapter, tmp_path) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    target = tmp_path / "export.json"
    pane.trigger_action("c1", "export", target=str(target))
    fake_adapter.export.assert_called_once_with("c1", str(target))


def test_load_as_context_action_emits_signal(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()

    received: list[dict] = []
    pane.context_loaded.connect(received.append)

    pane.trigger_action("c1", "load_as_context")
    fake_adapter.load_as_context.assert_called_once_with("c1")
    assert len(received) == 1
    assert received[0]["session_id"] == "c1"


def test_unknown_action_raises_value_error(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    with pytest.raises(ValueError, match="unknown action"):
        pane.trigger_action("c1", "explode")


def test_action_on_unknown_conversation_raises(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    fake_adapter.load_as_context.return_value = None
    fake_adapter.load_as_context.side_effect = KeyError("not found")
    with pytest.raises(KeyError):
        pane.trigger_action("missing", "load_as_context")
