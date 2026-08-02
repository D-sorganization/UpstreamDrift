"""Dock-chrome controller collaborator for :class:`UnifiedToolsSidebar` (F4).

``DockChromeController`` owns the collapse/minimize geometry, dock-area,
floating state, title-bar installation, and shortcut registration that were
previously mixed into the sidebar's own methods.  The sidebar stores a
reference to this object and delegates all dock-chrome concerns to it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .dock_title_bar import SidekickDockTitleBar
from .qt_compat import (
    QtGui,
    QtWidgets,
    all_sidebar_dock_features,
    dock_area,
)
from .state import SidebarState

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)

__all__ = ["DockChromeController"]

# Arbitrary large int used to restore Qt's unlimited max-width.
_QT_MAX_WIDTH = 16_777_215


class DockChromeController:
    """Manage collapse / minimize, dock widget, and title bar for a sidebar.

    Args:
        sidebar_widget: The :class:`~PyQt6.QtWidgets.QWidget` that will be
            wrapped in a :class:`~PyQt6.QtWidgets.QDockWidget`.
        tabs_widget: The :class:`~PyQt6.QtWidgets.QTabWidget` whose visibility
            is toggled on collapse.
        initial_state: The :class:`~.state.SidebarState` to seed width / area
            / floating from.

    Raises:
        TypeError: If *sidebar_widget* or *tabs_widget* is ``None``.
    """

    def __init__(
        self,
        sidebar_widget: QtWidgets.QWidget,
        tabs_widget: QtWidgets.QTabWidget,
        initial_state: SidebarState,
    ) -> None:
        if sidebar_widget is None:
            raise TypeError("sidebar_widget must not be None")
        if tabs_widget is None:
            raise TypeError("tabs_widget must not be None")
        self._sidebar = sidebar_widget
        self._tabs = tabs_widget
        self._dock_widget: QtWidgets.QDockWidget | None = None
        self._title_bar: SidekickDockTitleBar | None = None
        self._is_collapsed: bool = False
        self._expanded_width: int = initial_state.width

    # ── Read accessors ────────────────────────────────────────────────────────

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget | None:
        """Return the installed dock widget, if any."""
        return self._dock_widget

    @property
    def title_bar(self) -> SidekickDockTitleBar | None:
        """Return the custom title-bar widget, if installed."""
        return self._title_bar

    @property
    def is_collapsed(self) -> bool:
        """Return ``True`` when the sidebar is in collapsed (icon-strip) mode."""
        return self._is_collapsed

    # ── Collapse / minimize ───────────────────────────────────────────────────

    def toggle_collapsed(self) -> None:
        """Toggle between collapsed (icon-strip) and expanded states."""
        if self._is_collapsed:
            self._is_collapsed = False
            self._tabs.setVisible(True)
            self._sidebar.setMaximumWidth(_QT_MAX_WIDTH)
            self._apply_expanded_width()
        else:
            self._is_collapsed = True
            self._expanded_width = max(self._sidebar.width(), self._expanded_width)
            self._tabs.setVisible(False)
            self._sidebar.setMaximumWidth(56)

    def set_minimized(self, minimized: bool, expanded_width: int) -> None:
        """Collapse or expand programmatically (state restore path).

        Args:
            minimized: ``True`` to collapse to icon strip.
            expanded_width: Width to restore when expanding.
        """
        self._expanded_width = max(expanded_width, 240)
        if minimized:
            self._tabs.setVisible(False)
            self._sidebar.setMaximumWidth(56)
        else:
            self._tabs.setVisible(True)
            self._sidebar.setMaximumWidth(_QT_MAX_WIDTH)
            self._apply_expanded_width()

    def _apply_expanded_width(self) -> None:
        target = max(self._expanded_width, 240)
        parent = self._sidebar.parent()
        if isinstance(parent, QtWidgets.QSplitter):
            sizes = parent.sizes()
            idx = parent.indexOf(self._sidebar)
            if idx != -1 and sum(sizes) > 0:
                diff = target - sizes[idx]
                sizes[idx] = target
                if idx > 0:
                    sizes[idx - 1] = max(0, sizes[idx - 1] - diff)
                parent.setSizes(sizes)
                return
        self._sidebar.resize(target, self._sidebar.height())

    # ── Dock visibility ───────────────────────────────────────────────────────

    def toggle_visibility(self) -> None:
        """Toggle the dock's visibility (Ctrl+B shortcut handler)."""
        if self._dock_widget is None:
            return
        if self._dock_widget.isVisible():
            self._dock_widget.hide()
        else:
            self._dock_widget.show()

    # ── Dock area ─────────────────────────────────────────────────────────────

    def set_dock_area(self, area: str, state: SidebarState) -> bool:
        """Move the installed dock widget to the left or right side.

        Args:
            area: ``"left"`` or ``"right"``.
            state: :class:`~.state.SidebarState` whose ``dock_area`` will be
                updated in-place.

        Returns:
            ``True`` on success; ``False`` for unknown area strings.
        """
        if area not in {"left", "right"}:
            return False
        state.dock_area = area
        if self._dock_widget is not None:
            host = self._dock_widget.parent()
            if hasattr(host, "addDockWidget"):
                host.addDockWidget(dock_area(area), self._dock_widget)
        return True

    # ── Install / snapshot ────────────────────────────────────────────────────

    def install_as_dock(
        self,
        main_window: QtWidgets.QMainWindow,
        state: SidebarState,
        *,
        area: str | None = None,
        title: str = "Tools",
        state_path: str | Path | None = None,
        on_collapse: Callable[[], None] | None = None,
    ) -> QtWidgets.QDockWidget:
        """Install *sidebar_widget* as a :class:`~PyQt6.QtWidgets.QDockWidget`.

        Args:
            main_window: Host window to attach the dock to.
            state: Sidebar state (read for floating/area/size).
            area: Optional override for the initial dock area.
            title: Window title shown in the dock's title bar.
            state_path: Optional path from which to load state before install.
            on_collapse: Callable forwarded to the custom title bar's collapse
                button.  Defaults to :meth:`toggle_collapsed`.

        Returns:
            The newly created :class:`~PyQt6.QtWidgets.QDockWidget`.
        """
        if state_path is not None:
            state = SidebarState.load_json(state_path)

        dock = QtWidgets.QDockWidget(title, main_window)
        dock.setObjectName("SidekickDockWidget")
        dock.setFeatures(all_sidebar_dock_features())
        dock.setWidget(self._sidebar)
        dock.setFloating(state.floating)
        dock.resize(state.width, state.height)
        main_window.addDockWidget(dock_area(area or state.dock_area), dock)
        self._dock_widget = dock

        _on_collapse = on_collapse or self.toggle_collapsed
        self._title_bar = SidekickDockTitleBar(
            title,
            on_close=self.toggle_visibility,
            on_collapse=_on_collapse,
            parent=dock,
        )
        dock.setTitleBarWidget(self._title_bar)
        return dock

    def snapshot(
        self,
        fallback_state: SidebarState,
    ) -> tuple[str, bool, int, int]:
        """Return ``(dock_area, floating, width, height)`` from the live dock.

        Falls back to *fallback_state* fields when no dock is installed.
        """
        dock_area_name = fallback_state.dock_area
        floating = fallback_state.floating
        width = self._sidebar.width()
        height = self._sidebar.height()

        if self._dock_widget is not None:
            floating = self._dock_widget.isFloating()
            width = self._dock_widget.width()
            height = self._dock_widget.height()
            host = self._dock_widget.parent()
            if hasattr(host, "dockWidgetArea"):
                area = host.dockWidgetArea(self._dock_widget)
                if area == dock_area("left"):
                    dock_area_name = "left"
                elif area == dock_area("right"):
                    dock_area_name = "right"

        return dock_area_name, floating, width, height

    # ── Shortcuts ─────────────────────────────────────────────────────────────

    def register_shortcuts(
        self,
        main_window: QtWidgets.QMainWindow,
        on_toggle_visibility: Callable[[], None],
        on_toggle_collapsed: Callable[[], None],
    ) -> None:
        """Wire ``Ctrl+B`` / ``Ctrl+Shift+B`` shortcuts onto *main_window*.

        Args:
            main_window: The host main window.
            on_toggle_visibility: Connected to ``Ctrl+B``.
            on_toggle_collapsed: Connected to ``Ctrl+Shift+B``.
        """
        qshortcut = getattr(QtGui, "QShortcut", None) or getattr(
            QtWidgets, "QShortcut", None
        )
        if qshortcut is None:
            return

        sc_toggle = qshortcut(QtGui.QKeySequence("Ctrl+B"), main_window)
        sc_toggle.activated.connect(on_toggle_visibility)

        sc_collapse = qshortcut(QtGui.QKeySequence("Ctrl+Shift+B"), main_window)
        sc_collapse.activated.connect(on_toggle_collapsed)
