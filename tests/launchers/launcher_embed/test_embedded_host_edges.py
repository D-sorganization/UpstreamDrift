"""Edge-case tests for :class:`EmbeddedHostWidget`.

Complements ``tests/ui/launcher_embed/test_embedded_host_smoke.py`` by
covering error branches, the dock-area coercion helper, restore_state
input validation, and the mouse double-click handler.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from src.launchers.embedded_host import (  # noqa: E402
    EmbeddedHostWidget,
    _resolve_tool,
    _safe_cleanup,
    _safe_is_dirty,
)
from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    register_embeddable_tool,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _Tool:
    """Fixture EmbeddableTool that builds a QLabel widget."""

    def __init__(
        self,
        tool_id: str = "fx",
        supports: bool = True,
        dirty: bool = False,
        widget_factory=None,
    ) -> None:
        self.tool_id = tool_id
        self._supports = supports
        self._dirty = dirty
        self._widget_factory = widget_factory
        self.cleanup_called = False
        self.create_calls = 0

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(supports_embedded=self._supports)

    def create_main_widget(self, parent: Any) -> Any:
        self.create_calls += 1
        if self._widget_factory is not None:
            return self._widget_factory(parent)
        w = QLabel(self.tool_id, parent)
        w.setObjectName(f"fixture::{self.tool_id}")
        return w

    def cleanup(self) -> None:
        self.cleanup_called = True

    def is_dirty(self) -> bool:
        return self._dirty


@pytest.fixture
def host(_qapp):  # noqa: ANN001
    w = EmbeddedHostWidget()
    yield w
    w.close()
    w.deleteLater()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestResolveTool:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _resolve_tool("")

    def test_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _resolve_tool("   ")

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _resolve_tool(123)  # type: ignore[arg-type]

    def test_unregistered_raises(self) -> None:
        with pytest.raises(ValueError, match="not registered"):
            _resolve_tool("nope")

    def test_non_embeddable_raises(self) -> None:
        register_embeddable_tool(_Tool("ne", supports=False))
        with pytest.raises(ValueError, match="does not support embedding"):
            _resolve_tool("ne")


class TestSafeIsDirty:
    def test_returns_false_when_attribute_missing(self) -> None:
        class T:
            tool_id = "x"

        # Pass an object lacking is_dirty entirely.
        assert _safe_is_dirty(T()) is False  # type: ignore[arg-type]

    def test_returns_false_when_explicit_none(self) -> None:
        class T:
            tool_id = "x"
            is_dirty = None  # type: ignore[assignment]

        assert _safe_is_dirty(T()) is False  # type: ignore[arg-type]

    def test_returns_false_when_raises(self) -> None:
        class T:
            tool_id = "x"

            def is_dirty(self) -> bool:
                raise RuntimeError("boom")

        assert _safe_is_dirty(T()) is False  # type: ignore[arg-type]

    def test_returns_true(self) -> None:
        class T:
            tool_id = "x"

            def is_dirty(self) -> bool:
                return True

        assert _safe_is_dirty(T()) is True  # type: ignore[arg-type]


class TestSafeCleanup:
    def test_swallows_exceptions(self) -> None:
        class T:
            tool_id = "x"

            def cleanup(self) -> None:
                raise RuntimeError("boom")

        # Should not raise.
        _safe_cleanup(T())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Host construction & properties
# ---------------------------------------------------------------------------


class TestHostConstruction:
    def test_host_window_is_qmainwindow(self, host) -> None:  # noqa: ANN001
        assert isinstance(host.host_window, QMainWindow)

    def test_tab_widget_is_qtabwidget(self, host) -> None:  # noqa: ANN001
        assert isinstance(host.tab_widget, QTabWidget)

    def test_initial_state(self, host) -> None:  # noqa: ANN001
        assert host.tab_widget.count() == 0
        assert host.focus_mode is False
        assert host.active_tool_ids() == set()


# ---------------------------------------------------------------------------
# open_tab / open_dock error branches
# ---------------------------------------------------------------------------


class TestOpenTabErrors:
    def test_create_main_widget_returning_none_raises(self, host) -> None:  # noqa: ANN001
        tool = _Tool("nonewidget", widget_factory=lambda p: None)
        register_embeddable_tool(tool)
        with pytest.raises(ValueError, match="returned None"):
            host.open_tab(tool.tool_id)


class TestOpenDockErrors:
    def test_open_dock_unknown_raises(self, host) -> None:  # noqa: ANN001
        with pytest.raises(ValueError, match="not registered"):
            host.open_dock("ghost")

    def test_open_dock_non_embeddable_raises(self, host) -> None:  # noqa: ANN001
        register_embeddable_tool(_Tool("ne", supports=False))
        with pytest.raises(ValueError, match="does not support embedding"):
            host.open_dock("ne")

    def test_open_dock_create_returns_none(self, host) -> None:  # noqa: ANN001
        tool = _Tool("dock_none", widget_factory=lambda p: None)
        register_embeddable_tool(tool)
        with pytest.raises(ValueError, match="returned None"):
            host.open_dock(tool.tool_id)

    def test_open_dock_idempotent_raises_dock(self, host) -> None:  # noqa: ANN001
        # Already-mounted dock just re-shows.
        tool = _Tool("re_open_dock")
        register_embeddable_tool(tool)
        host.open_dock(tool.tool_id)
        host.open_dock(tool.tool_id)
        assert tool.create_calls == 1


# ---------------------------------------------------------------------------
# close_tab targeting
# ---------------------------------------------------------------------------


class TestCloseTabTargeting:
    def test_bool_target_returns_false(self, host) -> None:  # noqa: ANN001
        # Open tab at index 0 then close(True) should NOT match (bool is rejected).
        tool = _Tool("bool_t")
        register_embeddable_tool(tool)
        host.open_tab(tool.tool_id)
        assert host.close_tab(True) is False  # type: ignore[arg-type]
        assert tool.cleanup_called is False

    def test_unknown_type_returns_false(self, host) -> None:  # noqa: ANN001
        assert host.close_tab(3.14) is False  # type: ignore[arg-type]
        assert host.close_tab(None) is False  # type: ignore[arg-type]

    def test_close_tab_via_signal(self, host) -> None:  # noqa: ANN001
        tool = _Tool("signal_t")
        register_embeddable_tool(tool)
        host.open_tab(tool.tool_id)
        # Emit signal directly to exercise _on_tab_close_requested.
        host.tab_widget.tabCloseRequested.emit(0)
        assert tool.cleanup_called is True


class TestDockDirty:
    def test_dock_dirty_cancel_keeps_open(self, host) -> None:  # noqa: ANN001
        tool = _Tool("dock_dirty", dirty=True)
        register_embeddable_tool(tool)
        host.open_dock(tool.tool_id)

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Cancel,
        ):
            assert host.close_dock(tool.tool_id) is False
        assert tool.cleanup_called is False
        assert tool.tool_id in host.active_tool_ids()

    def test_dock_dirty_yes_closes(self, host) -> None:  # noqa: ANN001
        tool = _Tool("dock_dirty2", dirty=True)
        register_embeddable_tool(tool)
        host.open_dock(tool.tool_id)
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            assert host.close_dock(tool.tool_id) is True
        assert tool.cleanup_called is True


# ---------------------------------------------------------------------------
# restore_state edge cases
# ---------------------------------------------------------------------------


class TestRestoreState:
    def test_non_dict_raises(self, host) -> None:  # noqa: ANN001
        with pytest.raises(ValueError, match="must be a dict"):
            host.restore_state("not a dict")  # type: ignore[arg-type]

    def test_skips_non_string_tab_ids(self, host) -> None:  # noqa: ANN001
        # Non-string entries are silently skipped.
        host.restore_state({"tabs": [123, None, {"x": 1}], "docks": {}})
        assert host.active_tool_ids() == set()

    def test_skips_non_string_dock_ids(self, host) -> None:  # noqa: ANN001
        host.restore_state({"tabs": [], "docks": {123: 1, None: 2}})
        assert host.active_tool_ids() == set()

    def test_active_tab_out_of_range_ignored(self, host) -> None:  # noqa: ANN001
        tool = _Tool("at_oor")
        register_embeddable_tool(tool)
        host.restore_state({"tabs": [tool.tool_id], "docks": {}, "active_tab": 99})
        # Restored OK; out-of-range active_tab is silently skipped.
        assert tool.tool_id in host.active_tool_ids()

    def test_active_tab_negative_ignored(self, host) -> None:  # noqa: ANN001
        tool = _Tool("at_neg")
        register_embeddable_tool(tool)
        host.restore_state({"tabs": [tool.tool_id], "active_tab": -1})
        assert tool.tool_id in host.active_tool_ids()

    def test_active_tab_non_int_ignored(self, host) -> None:  # noqa: ANN001
        tool = _Tool("at_str")
        register_embeddable_tool(tool)
        host.restore_state({"tabs": [tool.tool_id], "active_tab": "0"})
        assert tool.tool_id in host.active_tool_ids()

    def test_none_tabs_and_docks_handled(self, host) -> None:  # noqa: ANN001
        # Explicit ``None`` values should still result in a no-op.
        host.restore_state({"tabs": None, "docks": None})
        assert host.active_tool_ids() == set()


# ---------------------------------------------------------------------------
# _coerce_dock_area
# ---------------------------------------------------------------------------


class TestCoerceDockArea:
    def test_passes_through_enum(self) -> None:
        area = Qt.DockWidgetArea.LeftDockWidgetArea
        assert EmbeddedHostWidget._coerce_dock_area(area) is area

    def test_coerces_valid_int(self) -> None:
        v = int(Qt.DockWidgetArea.LeftDockWidgetArea.value)
        result = EmbeddedHostWidget._coerce_dock_area(v)
        assert result == Qt.DockWidgetArea.LeftDockWidgetArea

    def test_invalid_int_falls_back_to_right(self) -> None:
        result = EmbeddedHostWidget._coerce_dock_area(99999)
        assert result == Qt.DockWidgetArea.RightDockWidgetArea

    def test_bool_int_rejected_returns_right(self) -> None:
        # bool is intentionally not accepted as an int dock area value.
        result = EmbeddedHostWidget._coerce_dock_area(True)
        assert result == Qt.DockWidgetArea.RightDockWidgetArea

    def test_string_falls_back_to_right(self) -> None:
        result = EmbeddedHostWidget._coerce_dock_area("left")
        assert result == Qt.DockWidgetArea.RightDockWidgetArea

    def test_none_falls_back_to_right(self) -> None:
        result = EmbeddedHostWidget._coerce_dock_area(None)
        assert result == Qt.DockWidgetArea.RightDockWidgetArea


# ---------------------------------------------------------------------------
# state_snapshot dock area persistence
# ---------------------------------------------------------------------------


class TestStateSnapshotDockArea:
    def test_snapshot_records_dock_area(self, host) -> None:  # noqa: ANN001
        tool = _Tool("dock_left")
        register_embeddable_tool(tool)
        host.open_dock(tool.tool_id, area=Qt.DockWidgetArea.LeftDockWidgetArea)
        snap = host.state_snapshot()
        assert snap["docks"][tool.tool_id] == int(
            Qt.DockWidgetArea.LeftDockWidgetArea.value
        )

    def test_restore_uses_recorded_area(self, host) -> None:  # noqa: ANN001
        tool = _Tool("dock_left2")
        register_embeddable_tool(tool)
        host.open_dock(tool.tool_id, area=Qt.DockWidgetArea.LeftDockWidgetArea)
        snap = host.state_snapshot()
        host.close_dock(tool.tool_id)
        host.restore_state(snap)
        # Reopened in the same area.
        assert tool.tool_id in host.active_tool_ids()


# ---------------------------------------------------------------------------
# mouseDoubleClickEvent passthrough
# ---------------------------------------------------------------------------


class TestMouseDoubleClick:
    def test_double_click_does_not_raise(self, host) -> None:  # noqa: ANN001
        ev = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            QPointF(1.0, 1.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        # Should pass through without raising.
        host.mouseDoubleClickEvent(ev)


# ---------------------------------------------------------------------------
# Tab reorder / reindexing
# ---------------------------------------------------------------------------


class TestOpenTabIdempotent:
    def test_existing_tab_returned_and_made_current(self, host) -> None:  # noqa: ANN001
        a = _Tool("aa")
        b = _Tool("bb")
        register_embeddable_tool(a)
        register_embeddable_tool(b)
        host.open_tab(a.tool_id)
        host.open_tab(b.tool_id)
        # b is current at index 1; reopening a should set current to its index.
        idx = host.open_tab(a.tool_id)
        assert idx == 0
        assert host.tab_widget.currentIndex() == 0
        assert a.create_calls == 1


class TestCloseTabDirty:
    def test_tab_dirty_cancel(self, host) -> None:  # noqa: ANN001
        tool = _Tool("dt", dirty=True)
        register_embeddable_tool(tool)
        host.open_tab(tool.tool_id)
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Cancel,
        ):
            assert host.close_tab(tool.tool_id) is False
        assert tool.cleanup_called is False


class TestCloseDockMissing:
    def test_close_dock_missing(self, host) -> None:  # noqa: ANN001
        assert host.close_dock("ghost") is False


class TestFocusModeNoTabBar:
    def test_set_focus_mode_when_tab_bar_none(self, host, monkeypatch) -> None:  # noqa: ANN001
        # Force tabBar() to return None to cover the early-return branch.
        monkeypatch.setattr(host.tab_widget, "tabBar", lambda: None)
        host.set_focus_mode(True)
        assert host.focus_mode is True


class TestDoubleClickTabBarMinusOne:
    def test_minus_one_index_still_toggles(self, host) -> None:  # noqa: ANN001
        # -1 fires when user double-clicks empty tab-bar real estate.
        host._on_tab_bar_double_clicked(-1)  # type: ignore[attr-defined]
        assert host.focus_mode is True


class TestRestoreStateSkipsUnknown:
    def test_unknown_tab_logged_and_skipped(self, host, caplog) -> None:  # noqa: ANN001
        with caplog.at_level("WARNING"):
            host.restore_state({"tabs": ["missing_tab"], "docks": {}})
        assert host.active_tool_ids() == set()

    def test_unknown_dock_logged_and_skipped(self, host, caplog) -> None:  # noqa: ANN001
        with caplog.at_level("WARNING"):
            host.restore_state(
                {
                    "tabs": [],
                    "docks": {
                        "missing_dock": int(Qt.DockWidgetArea.RightDockWidgetArea.value)
                    },
                }
            )
        assert host.active_tool_ids() == set()


class TestLookupTabIntMiss:
    def test_int_target_miss_returns_false(self, host) -> None:  # noqa: ANN001
        # Open one tab; close a non-existent index.
        tool = _Tool("hit")
        register_embeddable_tool(tool)
        host.open_tab(tool.tool_id)
        assert host.close_tab(42) is False
        assert tool.cleanup_called is False


class TestRestoreStateActiveTab:
    def test_active_tab_set_when_in_range(self, host) -> None:  # noqa: ANN001
        a = _Tool("rs_a")
        b = _Tool("rs_b")
        register_embeddable_tool(a)
        register_embeddable_tool(b)
        host.restore_state(
            {"tabs": [a.tool_id, b.tool_id], "docks": {}, "active_tab": 1}
        )
        assert host.tab_widget.currentIndex() == 1


class TestTabReindex:
    def test_close_first_of_two_reindexes(self, host) -> None:  # noqa: ANN001
        a = _Tool("a")
        b = _Tool("b")
        register_embeddable_tool(a)
        register_embeddable_tool(b)
        host.open_tab(a.tool_id)
        host.open_tab(b.tool_id)
        assert host.tab_widget.count() == 2

        # Close first; second tab's stored index should be refreshed.
        assert host.close_tab(a.tool_id) is True
        # ``b`` is now at index 0 in the widget; ensure the record reflects it.
        b_rec = host._active_tabs[b.tool_id]  # type: ignore[attr-defined]
        assert b_rec.index == 0
        assert host.tab_widget.indexOf(b_rec.widget) == 0

    def test_close_tab_by_index_after_reorder(self, host) -> None:  # noqa: ANN001
        a = _Tool("a2")
        b = _Tool("b2")
        register_embeddable_tool(a)
        register_embeddable_tool(b)
        host.open_tab(a.tool_id)
        host.open_tab(b.tool_id)
        # Move tab 1 to position 0.
        host.tab_widget.tabBar().moveTab(1, 0)
        # close_tab(0) should now close ``b`` (since indexOf is consulted).
        assert host.close_tab(0) is True
        assert b.cleanup_called is True
        assert a.cleanup_called is False
