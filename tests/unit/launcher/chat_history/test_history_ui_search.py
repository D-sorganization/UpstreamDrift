"""Tests for HistorySidebarPane search field + debounce + results rendering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.launchers.chat_history.history_pane import HistorySidebarPane  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def fake_adapter() -> MagicMock:
    a = MagicMock()
    a.list_active.return_value = [
        {
            "id": "c1",
            "title": "First chat",
            "timestamp": "2026-05-01T12:00:00",
            "snippet": "Hello there",
        },
        {
            "id": "c2",
            "title": "Second chat",
            "timestamp": "2026-05-02T12:00:00",
            "snippet": "Another",
        },
    ]
    a.list_archived.return_value = [
        {
            "id": "c3",
            "title": "Old chat",
            "timestamp": "2026-04-01T12:00:00",
            "snippet": "Old",
        },
    ]
    a.search.return_value = [
        {
            "id": "c1",
            "title": "First chat",
            "timestamp": "2026-05-01T12:00:00",
            "snippet": "Hello there",
        },
    ]
    return a


def test_pane_creates_search_field(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    assert pane.search_field is not None
    assert pane.search_field.placeholderText() != ""


def test_pane_lists_recent_grouped_by_date(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    # Should have rendered headers for the two different dates.
    text = pane.recent_text()
    assert "2026-05-01" in text
    assert "2026-05-02" in text


def test_pane_renders_archived_section(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter)
    pane.refresh()
    arch = pane.archived_text()
    assert "Old chat" in arch


def test_search_debounce_does_not_call_immediately(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter, debounce_ms=200)
    pane.refresh()
    fake_adapter.search.reset_mock()
    pane.search_field.setText("hello")
    # Without firing the debounce timer, no search call yet.
    assert fake_adapter.search.call_count == 0


def test_search_fires_after_debounce(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter, debounce_ms=0)
    pane.refresh()
    fake_adapter.search.reset_mock()
    pane.search_field.setText("hello")
    pane.flush_pending_search()  # test hook: force the debounce timer to fire
    assert fake_adapter.search.call_count == 1
    fake_adapter.search.assert_called_with("hello")


def test_empty_search_restores_recent_listing(qapp, fake_adapter) -> None:
    pane = HistorySidebarPane(fake_adapter, debounce_ms=0)
    pane.refresh()
    pane.search_field.setText("hello")
    pane.flush_pending_search()
    pane.search_field.setText("")
    pane.flush_pending_search()
    # Empty query is treated as "show all recent" and skips the adapter.
    # The last call was for non-empty 'hello'.
    assert "First chat" in pane.recent_text()


def test_search_results_replace_recent_view(qapp, fake_adapter) -> None:
    fake_adapter.search.return_value = [
        {
            "id": "c2",
            "title": "Second chat",
            "timestamp": "2026-05-02T12:00:00",
            "snippet": "Another",
        },
    ]
    pane = HistorySidebarPane(fake_adapter, debounce_ms=0)
    pane.refresh()
    pane.search_field.setText("another")
    pane.flush_pending_search()
    text = pane.recent_text()
    assert "Second chat" in text
    assert "First chat" not in text
