"""Headless smoke tests for :class:`EmbeddedHostWidget`.

Exercised under ``QT_QPA_PLATFORM=offscreen``; the tests skip cleanly
when PyQt6 is not available.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QLabel, QMessageBox, QWidget  # noqa: E402

from src.launchers.embedded_host import EmbeddedHostWidget  # noqa: E402
from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    register_embeddable_tool,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit]


class _FixtureTool:
    """Minimal :class:`EmbeddableTool` implementation for testing."""

    def __init__(
        self,
        tool_id: str = "fixture_tool",
        *,
        supports_embedded: bool = True,
        prefers_dock: bool = False,
    ) -> None:
        self.tool_id = tool_id
        self.cleanup_called = False
        self.create_calls = 0
        self._caps = EmbedCapabilities(
            supports_embedded=supports_embedded, prefers_dock=prefers_dock
        )

    def embed_capabilities(self) -> EmbedCapabilities:
        return self._caps

    def create_main_widget(self, parent: Any) -> QWidget:
        self.create_calls += 1
        label = QLabel("hello", parent)
        label.setObjectName(f"fixture::{self.tool_id}")
        return label

    def cleanup(self) -> None:
        self.cleanup_called = True

    def is_dirty(self) -> bool:
        return False


class _FixtureToolDirty(_FixtureTool):
    """Variant that always reports a dirty buffer."""

    def is_dirty(self) -> bool:
        return True


@pytest.fixture
def host(qapp):  # noqa: ANN001
    widget = EmbeddedHostWidget()
    yield widget
    widget.close()
    widget.deleteLater()


def test_open_tab_returns_index_zero(host) -> None:  # noqa: ANN001
    tool = _FixtureTool()
    register_embeddable_tool(tool)

    index = host.open_tab(tool.tool_id)

    assert index == 0
    assert host.tab_widget.count() == 1
    assert tool.create_calls == 1
    assert host.tab_widget.tabText(0) == tool.tool_id
    assert host.tab_widget.widget(0).isVisibleTo(host.tab_widget)


def test_open_tab_is_idempotent(host) -> None:  # noqa: ANN001
    tool = _FixtureTool()
    register_embeddable_tool(tool)

    first = host.open_tab(tool.tool_id)
    second = host.open_tab(tool.tool_id)

    assert first == second == 0
    assert host.tab_widget.count() == 1
    assert tool.create_calls == 1


def test_close_tab_calls_cleanup(host) -> None:  # noqa: ANN001
    tool = _FixtureTool()
    register_embeddable_tool(tool)
    host.open_tab(tool.tool_id)

    closed = host.close_tab(tool.tool_id)

    assert closed is True
    assert tool.cleanup_called is True
    assert host.tab_widget.count() == 0
    assert tool.tool_id not in host.active_tool_ids()


def test_close_tab_by_index(host) -> None:  # noqa: ANN001
    tool = _FixtureTool()
    register_embeddable_tool(tool)
    host.open_tab(tool.tool_id)

    assert host.close_tab(0) is True
    assert tool.cleanup_called is True


def test_close_nonexistent_tab_returns_false(host) -> None:  # noqa: ANN001
    assert host.close_tab(99) is False
    assert host.close_tab("nope") is False


def test_open_unknown_tool_raises(host) -> None:  # noqa: ANN001
    with pytest.raises(ValueError):
        host.open_tab("not_registered")


def test_open_non_embeddable_tool_raises(host) -> None:  # noqa: ANN001
    tool = _FixtureTool("non_embed", supports_embedded=False)
    register_embeddable_tool(tool)

    with pytest.raises(ValueError):
        host.open_tab(tool.tool_id)


def test_open_dock_marks_tool_active(host) -> None:  # noqa: ANN001
    tool = _FixtureTool("dock_tool")
    register_embeddable_tool(tool)

    host.open_dock(tool.tool_id)

    assert tool.tool_id in host.active_tool_ids()
    assert tool.create_calls == 1


def test_open_dock_idempotent(host) -> None:  # noqa: ANN001
    tool = _FixtureTool("dock_tool")
    register_embeddable_tool(tool)

    host.open_dock(tool.tool_id)
    host.open_dock(tool.tool_id)

    assert tool.create_calls == 1


def test_close_dock_calls_cleanup(host) -> None:  # noqa: ANN001
    tool = _FixtureTool("dock_tool")
    register_embeddable_tool(tool)
    host.open_dock(tool.tool_id)

    assert host.close_dock(tool.tool_id) is True
    assert tool.cleanup_called is True
    assert tool.tool_id not in host.active_tool_ids()


def test_close_missing_dock_returns_false(host) -> None:  # noqa: ANN001
    assert host.close_dock("nope") is False


def test_dirty_close_cancel_keeps_tab(host) -> None:  # noqa: ANN001
    tool = _FixtureToolDirty("dirty_tool")
    register_embeddable_tool(tool)
    host.open_tab(tool.tool_id)

    with patch.object(
        QMessageBox,
        "question",
        return_value=QMessageBox.StandardButton.Cancel,
    ) as mocked:
        result = host.close_tab(tool.tool_id)

    assert mocked.called
    assert result is False
    assert tool.cleanup_called is False
    assert tool.tool_id in host.active_tool_ids()


def test_dirty_close_yes_closes_tab(host) -> None:  # noqa: ANN001
    tool = _FixtureToolDirty("dirty_tool")
    register_embeddable_tool(tool)
    host.open_tab(tool.tool_id)

    with patch.object(
        QMessageBox,
        "question",
        return_value=QMessageBox.StandardButton.Yes,
    ):
        result = host.close_tab(tool.tool_id)

    assert result is True
    assert tool.cleanup_called is True


def test_state_snapshot_round_trip(host) -> None:  # noqa: ANN001
    tab_tool = _FixtureTool("tab_tool")
    dock_tool = _FixtureTool("dock_tool")
    register_embeddable_tool(tab_tool)
    register_embeddable_tool(dock_tool)

    host.open_tab(tab_tool.tool_id)
    host.open_dock(dock_tool.tool_id)

    snapshot = host.state_snapshot()
    assert snapshot["tabs"] == [tab_tool.tool_id]
    assert dock_tool.tool_id in snapshot["docks"]
    assert isinstance(snapshot["active_tab"], int)

    # Tear everything down and restore from snapshot.
    host.close_tab(tab_tool.tool_id)
    host.close_dock(dock_tool.tool_id)
    assert host.active_tool_ids() == set()

    host.restore_state(snapshot)

    assert tab_tool.tool_id in host.active_tool_ids()
    assert dock_tool.tool_id in host.active_tool_ids()


def test_restore_state_skips_missing_tools(host, caplog) -> None:  # noqa: ANN001
    # Nothing registered; restore should warn and continue without raising.
    host.restore_state(
        {
            "tabs": ["missing_tab"],
            "docks": {"missing_dock": int(Qt.DockWidgetArea.RightDockWidgetArea.value)},
            "active_tab": 0,
        }
    )
    assert host.active_tool_ids() == set()


def test_set_focus_mode_toggles_tab_bar(host) -> None:  # noqa: ANN001
    tool = _FixtureTool()
    register_embeddable_tool(tool)
    host.open_tab(tool.tool_id)

    tab_bar = host.tab_widget.tabBar()
    assert tab_bar is not None

    host.set_focus_mode(True)
    assert host.focus_mode is True
    assert tab_bar.isVisible() is False

    host.set_focus_mode(False)
    assert host.focus_mode is False
    # Visibility may depend on parent visibility; check the property.
    assert tab_bar.isHidden() is False


def test_close_event_cleans_up_all_tools(host) -> None:  # noqa: ANN001
    tab_tool = _FixtureTool("tab_tool")
    dock_tool = _FixtureTool("dock_tool")
    register_embeddable_tool(tab_tool)
    register_embeddable_tool(dock_tool)
    host.open_tab(tab_tool.tool_id)
    host.open_dock(dock_tool.tool_id)

    host.close()

    assert tab_tool.cleanup_called is True
    assert dock_tool.cleanup_called is True


def test_double_click_tab_bar_toggles_focus(host) -> None:  # noqa: ANN001
    tool = _FixtureTool()
    register_embeddable_tool(tool)
    host.open_tab(tool.tool_id)

    assert host.focus_mode is False
    host._on_tab_bar_double_clicked(0)  # type: ignore[attr-defined]
    assert host.focus_mode is True
    host._on_tab_bar_double_clicked(0)  # type: ignore[attr-defined]
    assert host.focus_mode is False
