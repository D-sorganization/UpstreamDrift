"""Tab/dock host widget for embedded launcher tools.

This module implements :class:`EmbeddedHostWidget` -- a ``QWidget`` that
hosts :class:`~src.shared.python.launcher_embed.contract.EmbeddableTool`
instances in tabs and dock widgets. It is the runtime substrate that the
main launcher window uses to display tools without spawning separate
top-level windows.

PyQt6 is imported at module level. Consumers that may run in
environments without PyQt6 should import this module lazily and guard
the import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.launcher_embed import (
    EmbeddableTool,
    get_embeddable_tool,
    is_embeddable,
)
from src.shared.python.logging_pkg.logging_config import (
    configure_gui_logging,
    get_logger,
)

if TYPE_CHECKING:
    from PyQt6.QtGui import QCloseEvent

configure_gui_logging()
logger = get_logger(__name__)

__all__ = ["EmbeddedHostWidget"]


@dataclass(slots=True)
class _OpenTab:
    """Bookkeeping record for a tool currently mounted as a tab."""

    tool: EmbeddableTool
    widget: QWidget
    index: int


@dataclass(slots=True)
class _OpenDock:
    """Bookkeeping record for a tool currently mounted as a dock widget."""

    tool: EmbeddableTool
    widget: QWidget
    dock: QDockWidget
    area: Qt.DockWidgetArea


def _resolve_tool(tool_id: str) -> EmbeddableTool:
    """Return the registered embeddable tool or raise ``ValueError``.

    Args:
        tool_id: Registry key for the tool to resolve.

    Raises:
        ValueError: If ``tool_id`` is not registered or is registered but
            does not advertise embedding support.
    """
    if not isinstance(tool_id, str) or not tool_id.strip():
        raise ValueError("tool_id must be a non-empty string")
    tool = get_embeddable_tool(tool_id)
    if tool is None:
        raise ValueError(f"tool_id {tool_id!r} is not registered")
    if not is_embeddable(tool_id):
        raise ValueError(
            f"tool_id {tool_id!r} is registered but does not support embedding"
        )
    return tool


def _safe_is_dirty(tool: EmbeddableTool) -> bool:
    """Return ``tool.is_dirty()`` defensively, defaulting to ``False``.

    Tools that omit :meth:`EmbeddableTool.is_dirty` (despite the
    Protocol) or whose implementation raises are treated as clean.
    """
    is_dirty = getattr(tool, "is_dirty", None)
    if is_dirty is None:
        return False
    try:
        return bool(is_dirty())
    except Exception:  # pragma: no cover - defensive
        logger.exception("is_dirty raised for tool %s", tool.tool_id)
        return False


def _safe_cleanup(tool: EmbeddableTool) -> None:
    """Call ``tool.cleanup()`` swallowing exceptions for shutdown safety."""
    try:
        tool.cleanup()
    except Exception:  # pragma: no cover - defensive
        logger.exception("cleanup raised for tool %s", tool.tool_id)


class EmbeddedHostWidget(QWidget):
    """Widget that hosts embeddable tools as tabs and dock panels.

    A :class:`QTabWidget` is the central area; an internal
    :class:`QMainWindow` provides the dock surface so that callers can
    add :class:`QDockWidget` instances without needing a top-level
    window. The :class:`QMainWindow` is exposed via :attr:`host_window`
    for parents that want to add docks elsewhere.

    Public methods raise :class:`ValueError` for contract violations
    (unknown tool ids, non-embeddable tools); they return ``False`` for
    benign no-op cases (closing a tab that does not exist, prompting on
    a dirty close that the user cancels).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active_tabs: dict[str, _OpenTab] = {}
        self._active_docks: dict[str, _OpenDock] = {}

        # Internal QMainWindow gives us a dock area without forcing the
        # host widget to be a top-level window itself.
        self._host_window = QMainWindow(self)
        self._tab_widget = QTabWidget(self._host_window)
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.setMovable(True)
        self._tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tab_widget.tabBarDoubleClicked.connect(self._on_tab_bar_double_clicked)
        self._host_window.setCentralWidget(self._tab_widget)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._host_window)

        self._focus_mode = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def host_window(self) -> QMainWindow:
        """The internal :class:`QMainWindow` that owns the dock area."""
        return self._host_window

    @property
    def tab_widget(self) -> QTabWidget:
        """The central :class:`QTabWidget` (read-only attribute)."""
        return self._tab_widget

    # ------------------------------------------------------------------
    # Tab API
    # ------------------------------------------------------------------

    def open_tab(self, tool_id: str) -> int:
        """Open ``tool_id`` as a tab and return the tab index.

        Idempotent: if the tool is already open as a tab, the existing
        tab is surfaced (made current) and its index is returned.

        Args:
            tool_id: Registry key for the tool to open.

        Returns:
            The integer tab index of the newly opened or surfaced tab.

        Raises:
            ValueError: If ``tool_id`` is not registered or is not
                embeddable.
        """
        existing = self._active_tabs.get(tool_id)
        if existing is not None:
            self._tab_widget.setCurrentIndex(existing.index)
            return existing.index

        tool = _resolve_tool(tool_id)
        widget = tool.create_main_widget(self)
        if widget is None:
            raise ValueError(f"tool {tool_id!r} create_main_widget returned None")

        index = self._tab_widget.addTab(widget, tool.tool_id)
        self._tab_widget.setCurrentIndex(index)
        self._active_tabs[tool_id] = _OpenTab(tool=tool, widget=widget, index=index)
        return index

    def close_tab(self, target: int | str) -> bool:
        """Close a tab by index or by ``tool_id``.

        If the tool reports :meth:`EmbeddableTool.is_dirty`, the user is
        prompted with a :class:`QMessageBox`; cancelling the prompt
        returns ``False`` and leaves the tab open.

        Args:
            target: Tab index (int) or tool id (str).

        Returns:
            ``True`` if the tab was closed; ``False`` if the tab does
            not exist or the user cancelled a dirty-close prompt.
        """
        record = self._lookup_tab(target)
        if record is None:
            return False

        if _safe_is_dirty(record.tool) and not self._confirm_dirty_close(record.tool):
            return False

        _safe_cleanup(record.tool)
        self._remove_tab_widget(record)
        return True

    def _lookup_tab(self, target: int | str) -> _OpenTab | None:
        """Return the tab record for ``target`` or ``None`` if missing."""
        if isinstance(target, bool):
            # ``bool`` is a subclass of ``int``; reject explicitly so
            # ``close_tab(True)`` does not silently match index 1.
            return None
        if isinstance(target, int):
            # Use indexOf() at lookup time to handle movable tabs correctly.
            # With setMovable(True), tab positions can change via drag-reorder,
            # so cached record.index values may be stale. By computing the
            # current index from the tab widget, we ensure the correct tab
            # is matched even after reordering.
            for record in self._active_tabs.values():
                if self._tab_widget.indexOf(record.widget) == target:
                    return record
            return None
        if isinstance(target, str):
            return self._active_tabs.get(target)
        return None

    def _remove_tab_widget(self, record: _OpenTab) -> None:
        """Remove ``record`` from the tab widget and active-tabs map."""
        index = self._tab_widget.indexOf(record.widget)
        if index != -1:
            self._tab_widget.removeTab(index)
        record.widget.setParent(None)
        record.widget.deleteLater()
        self._active_tabs.pop(record.tool.tool_id, None)
        self._reindex_tabs()

    def _reindex_tabs(self) -> None:
        """Refresh stored indices after a tab has been removed."""
        for record in self._active_tabs.values():
            record.index = self._tab_widget.indexOf(record.widget)

    def _on_tab_close_requested(self, index: int) -> None:
        """Slot connected to ``QTabWidget.tabCloseRequested``."""
        self.close_tab(index)

    def _on_tab_bar_double_clicked(self, index: int) -> None:
        """Slot: double-click on the tab bar toggles focus mode."""
        # The signal fires with -1 when the user double-clicks empty
        # tab-bar real estate; toggle anyway for ergonomics.
        del index
        self.set_focus_mode(not self._focus_mode)

    def _confirm_dirty_close(self, tool: EmbeddableTool) -> bool:
        """Prompt the user to confirm closing a dirty tool.

        Returns ``True`` if the user chose to close anyway, ``False`` to
        cancel.
        """
        result = QMessageBox.question(
            self,
            "Unsaved changes",
            (f"The tool {tool.tool_id!r} has unsaved changes. Close anyway?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return result == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------
    # Dock API
    # ------------------------------------------------------------------

    def open_dock(
        self,
        tool_id: str,
        area: Qt.DockWidgetArea = Qt.DockWidgetArea.RightDockWidgetArea,
    ) -> None:
        """Open ``tool_id`` as a :class:`QDockWidget` in ``area``.

        Idempotent: if the tool is already mounted as a dock, the
        existing dock is raised and shown.

        Args:
            tool_id: Registry key for the tool to open.
            area: Dock area to mount the dock in.

        Raises:
            ValueError: If ``tool_id`` is not registered or is not
                embeddable.
        """
        existing = self._active_docks.get(tool_id)
        if existing is not None:
            existing.dock.show()
            existing.dock.raise_()
            return

        tool = _resolve_tool(tool_id)
        widget = tool.create_main_widget(self)
        if widget is None:
            raise ValueError(f"tool {tool_id!r} create_main_widget returned None")

        dock = QDockWidget(tool.tool_id, self._host_window)
        dock.setObjectName(f"embedded_dock::{tool.tool_id}")
        dock.setWidget(widget)
        self._host_window.addDockWidget(area, dock)
        self._active_docks[tool_id] = _OpenDock(
            tool=tool, widget=widget, dock=dock, area=area
        )

    def close_dock(self, tool_id: str) -> bool:
        """Close the dock for ``tool_id``.

        Returns:
            ``True`` if the dock was closed; ``False`` if no dock for
            ``tool_id`` is open or if the user cancelled a dirty-close
            prompt.
        """
        record = self._active_docks.get(tool_id)
        if record is None:
            return False

        if _safe_is_dirty(record.tool) and not self._confirm_dirty_close(record.tool):
            return False

        _safe_cleanup(record.tool)
        self._host_window.removeDockWidget(record.dock)
        record.dock.setParent(None)
        record.dock.deleteLater()
        self._active_docks.pop(tool_id, None)
        return True

    # ------------------------------------------------------------------
    # Focus mode
    # ------------------------------------------------------------------

    def set_focus_mode(self, enabled: bool) -> None:
        """Toggle focus mode.

        When enabled, the tab bar is hidden so the active tab fills the
        host. When disabled, the tab bar is restored.
        """
        self._focus_mode = bool(enabled)
        tab_bar = self._tab_widget.tabBar()
        if tab_bar is not None:
            tab_bar.setVisible(not self._focus_mode)

    @property
    def focus_mode(self) -> bool:
        """Whether focus mode is currently enabled."""
        return self._focus_mode

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def active_tool_ids(self) -> set[str]:
        """Return the set of currently mounted tool ids (tabs + docks)."""
        return set(self._active_tabs.keys()) | set(self._active_docks.keys())

    def state_snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of currently mounted tools.

        The shape is:

        ``{"tabs": [tool_id, ...], "docks": {tool_id: area_int},
        "active_tab": int}``

        ``area_int`` is the integer value of the corresponding
        :class:`Qt.DockWidgetArea` enum, suitable for JSON
        serialisation.
        """
        # Preserve the visual ordering of tabs.
        ordered: list[tuple[int, str]] = sorted(
            (
                (self._tab_widget.indexOf(rec.widget), tool_id)
                for tool_id, rec in self._active_tabs.items()
            ),
            key=lambda pair: pair[0],
        )
        return {
            "tabs": [tool_id for _, tool_id in ordered],
            "docks": {
                tool_id: int(rec.area.value)
                for tool_id, rec in self._active_docks.items()
            },
            "active_tab": int(self._tab_widget.currentIndex()),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Re-open tabs and docks listed in ``state``.

        Best-effort: tools that are not registered or fail to embed are
        logged and skipped without raising.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")

        for tool_id in state.get("tabs", []) or []:
            if not isinstance(tool_id, str):
                continue
            try:
                self.open_tab(tool_id)
            except ValueError as exc:
                logger.warning("restore_state: skipping tab %r (%s)", tool_id, exc)

        for tool_id, area_value in (state.get("docks", {}) or {}).items():
            if not isinstance(tool_id, str):
                continue
            area = self._coerce_dock_area(area_value)
            try:
                self.open_dock(tool_id, area=area)
            except ValueError as exc:
                logger.warning("restore_state: skipping dock %r (%s)", tool_id, exc)

        active_tab = state.get("active_tab")
        if isinstance(active_tab, int) and 0 <= active_tab < (self._tab_widget.count()):
            self._tab_widget.setCurrentIndex(active_tab)

    @staticmethod
    def _coerce_dock_area(area_value: Any) -> Qt.DockWidgetArea:
        """Best-effort coercion of a serialised area value to an enum."""
        if isinstance(area_value, Qt.DockWidgetArea):
            return area_value
        if isinstance(area_value, int) and not isinstance(area_value, bool):
            try:
                return Qt.DockWidgetArea(area_value)
            except ValueError:
                logger.warning(
                    "restore_state: unknown dock area %r; falling back to right",
                    area_value,
                )
        return Qt.DockWidgetArea.RightDockWidgetArea

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Run :meth:`EmbeddableTool.cleanup` on every active tool."""
        for record in list(self._active_tabs.values()):
            _safe_cleanup(record.tool)
        for record in list(self._active_docks.values()):
            _safe_cleanup(record.tool)
        self._active_tabs.clear()
        self._active_docks.clear()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Convenience: double-click on tab content also toggles focus mode.
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Forward double-click on host chrome to focus-mode toggle."""
        # We intentionally only react to double-clicks that bubble up to
        # the host widget itself; tab-bar double-clicks are handled via
        # ``tabBarDoubleClicked`` so widget content keeps its own
        # double-click semantics.
        super().mouseDoubleClickEvent(event)
