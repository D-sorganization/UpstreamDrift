"""Tests for the memory archive-digest workflow."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.launchers.chat_history.memory_panel import MemoryPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def adapter() -> MagicMock:
    a = MagicMock()
    a.load_memory.return_value = {
        "identity": {},
        "preferences": {},
        "projects": {},
        "knowledge": {},
    }
    a.list_archived.return_value = [
        {"id": "c1", "title": "Old", "timestamp": "2026-01-01T00:00:00"},
        {"id": "c2", "title": "Older", "timestamp": "2025-12-01T00:00:00"},
    ]
    a.condense_to_memory.return_value = {
        "status": "ok",
        "summary": "User likes Pinocchio.",
        "merged_into": ["knowledge"],
    }
    return a


def test_digest_uses_archived_conversation_ids(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    panel.archive_digest()
    adapter.condense_to_memory.assert_called_once_with(["c1", "c2"])


def test_digest_reloads_memory_after_call(qapp, adapter) -> None:
    panel = MemoryPanel(adapter)
    panel.refresh()
    adapter.load_memory.reset_mock()
    panel.archive_digest()
    # After digest, the panel reloads to display merged entries.
    assert adapter.load_memory.call_count >= 1


def test_digest_with_no_archived_is_noop(qapp) -> None:
    a = MagicMock()
    a.load_memory.return_value = {
        "identity": {},
        "preferences": {},
        "projects": {},
        "knowledge": {},
    }
    a.list_archived.return_value = []
    panel = MemoryPanel(a)
    panel.refresh()
    panel.archive_digest()
    a.condense_to_memory.assert_not_called()


def test_digest_stub_fallback_does_not_crash(qapp) -> None:
    a = MagicMock()
    a.load_memory.return_value = {
        "identity": {},
        "preferences": {},
        "projects": {},
        "knowledge": {},
    }
    a.list_archived.return_value = [
        {"id": "c1", "title": "Old", "timestamp": "2026-01-01T00:00:00"},
    ]
    a.condense_to_memory.return_value = {
        "status": "stub",
        "message": "Condenser API not available — try again later.",
    }
    panel = MemoryPanel(a)
    panel.refresh()
    panel.archive_digest()  # must not raise
    assert panel.last_digest_status() == "stub"


def test_digest_is_gated_when_condenser_is_unavailable(qapp) -> None:
    a = MagicMock(
        spec=[
            "load_memory",
            "list_archived",
            "condense_to_memory",
            "has_condensation_api",
        ]
    )
    a.load_memory.return_value = {
        "identity": {},
        "preferences": {},
        "projects": {},
        "knowledge": {},
    }
    a.has_condensation_api.return_value = False
    panel = MemoryPanel(a)

    panel.archive_digest()

    assert panel.last_digest_status() == "unavailable"
    a.list_archived.assert_not_called()
    a.condense_to_memory.assert_not_called()
