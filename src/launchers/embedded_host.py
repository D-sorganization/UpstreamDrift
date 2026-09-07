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

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ui.tile_help import attach_tile_help
from src.shared.python.launcher_embed import (
    EmbeddableTool,
    InMemoryLauncherContext,
    LauncherContext,
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


@dataclass(slots=True)
class _Backgrounded:
    """Bookkeeping record for a tool that is paused and stashed (hidden).

    The widget is kept alive (not cleaned up) so it can be re-surfaced
    later via :meth:`EmbeddedHostWidget.open_tab` with its state intact.
    """

    tool: EmbeddableTool
    widget: QWidget


@dataclass(slots=True)
class _PoppedOut:
    """Bookkeeping record for a tool re-parented into its own window."""

    tool: EmbeddableTool
    widget: QWidget
    window: QMainWindow


class _PopOutWindow(QMainWindow):
    """Top-level window that re-docks its tool when closed.

    Created by :meth:`EmbeddedHostWidget.pop_out_tab`. On
    :meth:`closeEvent` it asks the host to dock the tool back into a
    tab so closing the window never destroys a live tool.
    """

    def __init__(self, host: EmbeddedHostWidget, tool_id: str) -> None:
        super().__init__()
        self._host = host
        self._tool_id = tool_id
        # Closing the window should re-dock, not delete the tool widget.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Re-dock the tool back into the host on window close."""
        # Re-parent the widget out before the window tears down so it is
        # not destroyed with the window. ``dock_back`` is idempotent: if
        # the host already docked back (programmatic call), this no-ops.
        self._host.dock_back(self._tool_id)
        super().closeEvent(event)


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


def _safe_can_background(tool: EmbeddableTool) -> bool:
    """Return ``tool.can_background()`` defensively, defaulting to ``True``.

    Tools that omit the optional :meth:`EmbeddableTool.can_background`
    hook (the common case for the ~17 pre-existing adapters) default to
    backgroundable. A raising implementation is treated as ``True`` so a
    buggy tool still gets the non-destructive close path.
    """
    can_background = getattr(tool, "can_background", None)
    if can_background is None:
        return True
    try:
        return bool(can_background())
    except Exception:  # pragma: no cover - defensive
        logger.exception("can_background raised for tool %s", tool.tool_id)
        return True


def _safe_detach_to_window(tool: EmbeddableTool) -> bool:
    """Return ``tool.detach_to_window()`` defensively, defaulting to ``True``.

    Tools that omit the optional :meth:`EmbeddableTool.detach_to_window`
    hook default to pop-out-able. A raising implementation is treated as
    pin-only (``False``) so a buggy tool is not yanked out of the host.
    """
    detach = getattr(tool, "detach_to_window", None)
    if detach is None:
        return True
    try:
        return bool(detach())
    except Exception:  # pragma: no cover - defensive
        logger.exception("detach_to_window raised for tool %s", tool.tool_id)
        return False


def _safe_pause(tool: EmbeddableTool, widget: QWidget | None = None) -> None:
    """Call the optional pause hook, preferring per-widget pause when present."""
    if widget is not None:
        pause_widget = getattr(tool, "pause_widget", None)
        if pause_widget is not None:
            try:
                pause_widget(widget)
            except Exception:  # pragma: no cover - defensive
                logger.exception("pause_widget raised for tool %s", tool.tool_id)
            return
    pause = getattr(tool, "pause", None)
    if pause is None:
        return
    try:
        pause()
    except Exception:  # pragma: no cover - defensive
        logger.exception("pause raised for tool %s", tool.tool_id)


def _safe_resume(tool: EmbeddableTool, widget: QWidget | None = None) -> None:
    """Call the optional resume hook, preferring per-widget resume when present."""
    if widget is not None:
        resume_widget = getattr(tool, "resume_widget", None)
        if resume_widget is not None:
            try:
                resume_widget(widget)
            except Exception:  # pragma: no cover - defensive
                logger.exception("resume_widget raised for tool %s", tool.tool_id)
            return
    resume = getattr(tool, "resume", None)
    if resume is None:
        return
    try:
        resume()
    except Exception:  # pragma: no cover - defensive
        logger.exception("resume raised for tool %s", tool.tool_id)


def _safe_set_launcher_context(
    tool: EmbeddableTool,
    context: LauncherContext,
) -> None:
    """Inject the optional launcher context hook when a tool opts in."""
    set_context = getattr(tool, "set_launcher_context", None)
    if set_context is None:
        return
    try:
        set_context(context)
    except Exception:  # pragma: no cover - defensive
        logger.exception("set_launcher_context raised for tool %s", tool.tool_id)


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
        self._launcher_context = InMemoryLauncherContext()
        self._active_tabs: dict[str, _OpenTab] = {}
        self._active_docks: dict[str, _OpenDock] = {}
        # Tools the user closed with "keep running": paused + hidden but
        # not cleaned up, keyed by tool_id (#6013).
        self._backgrounded: dict[str, _Backgrounded] = {}
        # Tools re-parented into their own top-level window (#6013).
        self._popped_out: dict[str, _PoppedOut] = {}

        # Internal QMainWindow gives us a dock area without forcing the
        # host widget to be a top-level window itself.
        self._host_window = QMainWindow(self)
        self._tab_widget = QTabWidget(self._host_window)
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.setMovable(True)
        self._tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tab_widget.tabBarDoubleClicked.connect(self._on_tab_bar_double_clicked)
        self._host_window.setCentralWidget(self._tab_widget)

        # Right-click on the tab bar offers per-tab close/pop-out actions.
        tab_bar = self._tab_widget.tabBar()
        if tab_bar is not None:
            tab_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            tab_bar.customContextMenuRequested.connect(
                self._on_tab_context_menu_requested
            )

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

    @property
    def launcher_context(self) -> LauncherContext:
        """Shared in-process context for opt-in embedded tools."""
        return self._launcher_context

    # ------------------------------------------------------------------
    # Tab API
    # ------------------------------------------------------------------

    def _build_tool_widget(self, tool_id: str) -> tuple[Any, QWidget]:
        """Resolve ``tool_id`` and build its widget, help affordance included.

        The tab path and the dock path both need exactly this sequence, and
        the DRY gate rightly rejects it written twice (issue #9413).

        Args:
            tool_id: Registry tile id of the tool to open.

        Returns:
            The resolved tool and its freshly built main widget, with the
            tile's F1 help already attached.

        Raises:
            ValueError: If the tool's ``create_main_widget`` returns ``None``.
        """
        tool = _resolve_tool(tool_id)
        _safe_set_launcher_context(tool, self._launcher_context)
        widget = tool.create_main_widget(self)
        if widget is None:
            raise ValueError(f"tool {tool_id!r} create_main_widget returned None")
        attach_tile_help(widget, tool_id)
        return tool, widget

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

        # If the tool is popped out into its own window, re-dock it
        # rather than constructing a fresh widget.
        if tool_id in self._popped_out:
            return self.dock_back(tool_id)

        # Re-surface a backgrounded (paused, stashed) widget instead of
        # rebuilding it, preserving its in-memory state (#6013).
        stashed = self._backgrounded.pop(tool_id, None)
        if stashed is not None:
            return self._resurface_backgrounded(stashed)

        tool, widget = self._build_tool_widget(tool_id)

        index = self._tab_widget.addTab(widget, tool.tool_id)
        self._tab_widget.setCurrentIndex(index)
        self._active_tabs[tool_id] = _OpenTab(tool=tool, widget=widget, index=index)
        self._launcher_context.emit(
            "tab.opened", {"tool_id": tool_id, "surface": "tab"}
        )
        return index

    def _resurface_backgrounded(self, stashed: _Backgrounded) -> int:
        """Re-mount a stashed widget as a tab and resume the tool."""
        widget = stashed.widget
        widget.setParent(self._tab_widget)
        widget.show()
        index = self._tab_widget.addTab(widget, stashed.tool.tool_id)
        self._tab_widget.setCurrentIndex(index)
        self._active_tabs[stashed.tool.tool_id] = _OpenTab(
            tool=stashed.tool, widget=widget, index=index
        )
        _safe_resume(stashed.tool, stashed.widget)
        self._launcher_context.emit(
            "tab.opened", {"tool_id": stashed.tool.tool_id, "surface": "tab"}
        )
        return index

    def close_tab(self, target: int | str, *, destroy: bool = True) -> bool:
        """Close a tab by index or by ``tool_id``.

        If the tool reports :meth:`EmbeddableTool.is_dirty`, the user is
        prompted with a :class:`QMessageBox`; cancelling the prompt
        returns ``False`` and leaves the tab open.

        Args:
            target: Tab index (int) or tool id (str).
            destroy: When ``True`` (default) the tool is cleaned up and
                its widget destroyed (legacy behaviour). When ``False``
                and the tool reports :meth:`EmbeddableTool.can_background`
                is truthy, the tool is paused and stashed (kept running,
                hidden) instead of destroyed. If the tool cannot be
                backgrounded this falls back to destroy.

        Returns:
            ``True`` if the tab was closed (destroyed or backgrounded);
            ``False`` if the tab does not exist or the user cancelled a
            dirty-close prompt.
        """
        record = self._lookup_tab(target)
        if record is None:
            return False

        background = not destroy and _safe_can_background(record.tool)

        # The dirty prompt only matters for destructive closes; a
        # backgrounded tool keeps its state, so there is nothing to lose.
        if (
            destroy
            and _safe_is_dirty(record.tool)
            and not self._confirm_dirty_close(record.tool)
        ):
            return False

        if background:
            self._background_tab(record)
        else:
            _safe_cleanup(record.tool)
            self._remove_tab_widget(record)
        self._launcher_context.emit(
            "tab.closed",
            {
                "tool_id": record.tool.tool_id,
                "surface": "tab",
                "destroyed": not background,
            },
        )
        return True

    def _background_tab(self, record: _OpenTab) -> None:
        """Pause ``record``'s tool and stash its widget hidden (#6013)."""
        _safe_pause(record.tool, record.widget)
        index = self._tab_widget.indexOf(record.widget)
        if index != -1:
            self._tab_widget.removeTab(index)
        record.widget.setParent(None)
        record.widget.hide()
        self._active_tabs.pop(record.tool.tool_id, None)
        self._backgrounded[record.tool.tool_id] = _Backgrounded(
            tool=record.tool, widget=record.widget
        )
        self._reindex_tabs()

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
        """Slot connected to ``QTabWidget.tabCloseRequested``.

        When the tool supports backgrounding, prompt the user to choose
        between keeping it running (paused + stashed) and destroying it.
        Tools that cannot be backgrounded take the legacy destroy path
        without a prompt.
        """
        record = self._lookup_tab(index)
        if record is None:
            return
        if not _safe_can_background(record.tool):
            self.close_tab(index, destroy=True)
            return

        destroy = self._prompt_close_disposition(record.tool)
        if destroy is None:
            return  # user cancelled
        self.close_tab(index, destroy=destroy)

    def _prompt_close_disposition(self, tool: EmbeddableTool) -> bool | None:
        """Ask whether to destroy a tool or keep it running in background.

        Returns:
            ``True`` to destroy, ``False`` to background (keep running),
            or ``None`` if the user cancelled.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Close tab")
        box.setText(
            f"Close {tool.tool_id!r}?\n\n"
            "Keep it running in the background, or destroy it now."
        )
        keep = box.addButton("Close (keep running)", QMessageBox.ButtonRole.AcceptRole)
        destroy = box.addButton("Destroy", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep:
            return False
        if clicked is destroy:
            return True
        return None

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
    # Tab context menu + pop-out / dock-back (#6013)
    # ------------------------------------------------------------------

    def _on_tab_context_menu_requested(self, point: QPoint) -> None:
        """Slot: build and show the per-tab right-click context menu."""
        tab_bar = self._tab_widget.tabBar()
        if tab_bar is None:
            return
        index = tab_bar.tabAt(point)
        if index < 0:
            return
        record = self._lookup_tab(index)
        if record is None:
            return

        menu = QMenu(self)
        if _safe_can_background(record.tool):
            keep = menu.addAction("Close (keep running)")
            keep.triggered.connect(
                lambda _=False, i=index: self.close_tab(i, destroy=False)
            )
        destroy = menu.addAction("Destroy")
        destroy.triggered.connect(
            lambda _=False, i=index: self.close_tab(i, destroy=True)
        )
        if _safe_detach_to_window(record.tool):
            pop_out = menu.addAction("Pop out")
            pop_out.triggered.connect(
                lambda _=False, tid=record.tool.tool_id: self.pop_out_tab(tid)
            )
        menu.exec(tab_bar.mapToGlobal(point))

    def pop_out_tab(self, tool_id: str) -> bool:
        """Re-parent ``tool_id``'s tab widget into its own top-level window.

        The tab is removed from the host; closing the popped-out window
        re-docks the tool via :meth:`dock_back`.

        Args:
            tool_id: Registry key of an open tab.

        Returns:
            ``True`` if the tool was popped out; ``False`` if it is not
            open as a tab or its :meth:`EmbeddableTool.detach_to_window`
            hook is pin-only.

        Raises:
            ValueError: If ``tool_id`` is not a non-empty string.
        """
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError("tool_id must be a non-empty string")
        record = self._active_tabs.get(tool_id)
        if record is None:
            return False
        if not _safe_detach_to_window(record.tool):
            logger.info("pop_out_tab: %s is pin-only; ignoring", tool_id)
            return False

        # Bracket the re-parent with pause/resume so tools holding
        # re-parent-sensitive resources (GL contexts, timers, IPC) get a
        # chance to quiesce before the widget changes top-level windows.
        _safe_pause(record.tool, record.widget)

        index = self._tab_widget.indexOf(record.widget)
        if index != -1:
            self._tab_widget.removeTab(index)
        self._active_tabs.pop(tool_id, None)
        self._reindex_tabs()

        window = _PopOutWindow(self, tool_id)
        window.setWindowTitle(record.tool.tool_id)
        window.setCentralWidget(record.widget)
        record.widget.show()
        self._popped_out[tool_id] = _PoppedOut(
            tool=record.tool, widget=record.widget, window=window
        )
        window.show()
        window.raise_()
        _safe_resume(record.tool, record.widget)
        return True

    def dock_back(self, tool_id: str) -> int:
        """Re-dock a popped-out tool as a tab and return its tab index.

        The reverse of :meth:`pop_out_tab`. Safe to call from the
        popped-out window's close handler.

        Args:
            tool_id: Registry key of a popped-out tool.

        Returns:
            The integer tab index of the re-docked tab, or ``-1`` if no
            popped-out window exists for ``tool_id``.
        """
        record = self._popped_out.pop(tool_id, None)
        if record is None:
            return -1

        # Same pause/resume bracket as pop_out_tab: the widget is about
        # to migrate between top-level windows.
        _safe_pause(record.tool, record.widget)

        widget = record.widget
        widget.setParent(self._tab_widget)
        widget.show()
        index = self._tab_widget.addTab(widget, record.tool.tool_id)
        self._tab_widget.setCurrentIndex(index)
        self._active_tabs[tool_id] = _OpenTab(
            tool=record.tool, widget=widget, index=index
        )

        # Detach the (now empty) window and dispose of it.
        record.window.setCentralWidget(None)
        record.window.close()
        record.window.deleteLater()
        _safe_resume(record.tool, widget)
        return index

    def backgrounded_tools(self) -> set[str]:
        """Return the set of tool ids currently paused in the background.

        Surfaced for callers (e.g. the training-controller tab) that
        need to know which tools are alive-but-hidden.
        """
        return set(self._backgrounded.keys())

    def popped_out_tools(self) -> set[str]:
        """Return the set of tool ids currently in their own windows."""
        return set(self._popped_out.keys())

    def open_tool_ids(self) -> list[str]:
        """Return the tool ids of open tabs in current display order.

        Display order reflects user drag-reordering of the movable tab
        bar, so indices are computed live rather than from cached
        bookkeeping.
        """
        by_index: list[tuple[int, str]] = []
        for tool_id, record in self._active_tabs.items():
            index = self._tab_widget.indexOf(record.widget)
            if index != -1:
                by_index.append((index, tool_id))
        return [tool_id for _, tool_id in sorted(by_index)]

    def active_tool_id(self) -> str | None:
        """Return the tool id of the currently focused tab, or ``None``."""
        current = self._tab_widget.currentIndex()
        if current < 0:
            return None
        record = self._lookup_tab(current)
        return record.tool.tool_id if record is not None else None

    def focus_tab(self, tool_id: str) -> None:
        """Bring ``tool_id``'s tab to the front.

        Backgrounded and popped-out tools are re-surfaced via
        :meth:`open_tab` (which re-docks / resumes them).

        Raises:
            KeyError: If ``tool_id`` is not an open, backgrounded, or
                popped-out tool.
        """
        if (
            tool_id not in self._active_tabs
            and tool_id not in self._backgrounded
            and tool_id not in self._popped_out
        ):
            raise KeyError(tool_id)
        self.open_tab(tool_id)

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

        tool, widget = self._build_tool_widget(tool_id)

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
        """Return the set of currently mounted tool ids.

        Includes tabs, docks, and popped-out windows. Backgrounded
        (paused, hidden) tools are intentionally excluded — query
        :meth:`backgrounded_tools` for those.
        """
        return (
            set(self._active_tabs.keys())
            | set(self._active_docks.keys())
            | set(self._popped_out.keys())
        )

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
        """Run :meth:`EmbeddableTool.cleanup` on every live tool.

        Covers tabs, docks, backgrounded (paused) tools, and popped-out
        windows — host shutdown destroys everything regardless of how it
        was mounted.
        """
        for tab_record in list(self._active_tabs.values()):
            _safe_cleanup(tab_record.tool)
        for dock_record in list(self._active_docks.values()):
            _safe_cleanup(dock_record.tool)
        for bg_record in list(self._backgrounded.values()):
            _safe_cleanup(bg_record.tool)
        for pop_record in list(self._popped_out.values()):
            _safe_cleanup(pop_record.tool)
            pop_record.window.setCentralWidget(None)
            pop_record.window.close()
            pop_record.window.deleteLater()
        self._active_tabs.clear()
        self._active_docks.clear()
        self._backgrounded.clear()
        self._popped_out.clear()
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
