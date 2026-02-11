"""Draggable Tab System (Shared Module).

Provides an enhanced QTabWidget with draggable, detachable tabs and
comprehensive redocking functionality. Ported from Gasification_Model's
implementation for fleet-wide reuse.

Features:
    - Drag and drop tab detachment (drag tab outside bar to pop out)
    - Right-click context menus (close, pop out, redock)
    - Protected core tabs that cannot be closed
    - Multiple redocking methods (Ctrl+D, double-click, right-click, menu)
    - Closed-tab memory with factory-based reopening

Usage:
    from src.shared.python.gui_pkg.draggable_tabs import DraggableTabWidget

    tabs = DraggableTabWidget(core_tabs={"Home", "Settings"})
    tabs.addTab(my_widget, "My Tab")

Dependencies: PyQt6
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCursor, QIcon, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabBar,
    QTabWidget,
    QWidget,
)

logger = logging.getLogger(__name__)


class DraggableTabWidget(QTabWidget):
    """Enhanced QTabWidget with draggable, detachable tabs.

    Signals:
        tab_detached: Emitted when a tab is dragged out (index, position).
        tab_moved: Emitted when tabs are reordered (from_index, to_index).
    """

    tab_detached = pyqtSignal(int, QPoint)
    tab_moved = pyqtSignal(int, int)

    def __init__(
        self,
        parent: QWidget | None = None,
        core_tabs: set[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)

        self.detached_tabs: dict[DetachedTabWindow, tuple[QWidget, str, QIcon]] = {}
        self.drag_start_pos = QPoint()

        # Track closed tabs for "Open Tab" functionality
        self.closed_tabs: dict[str, Callable[[], QWidget | None]] = {}
        self.tab_factories: dict[str, Callable[[], QWidget | None]] = {}

        # Core tabs cannot be closed
        self.core_tabs: set[str] = core_tabs if core_tabs is not None else set()

        # Enable context menu and event filter on tab bar
        tab_bar = self.tabBar()
        if tab_bar:
            tab_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            tab_bar.customContextMenuRequested.connect(self._show_tab_context_menu)
            tab_bar.installEventFilter(self)

    # ── Tab lifecycle overrides ─────────────────────────────────────

    def addTab(self, widget: QWidget, *args) -> int:  # type: ignore[override]
        """Override to apply UX enhancements on new tabs."""
        index = super().addTab(widget, *args)
        self._update_tab_ux(index)
        return index

    def insertTab(self, index: int, widget: QWidget, *args) -> int:  # type: ignore[override]
        """Override to apply UX enhancements on inserted tabs."""
        ret_index = super().insertTab(index, widget, *args)
        self._update_tab_ux(ret_index)
        return ret_index

    def _update_tab_ux(self, index: int) -> None:
        """Hide close button for core tabs and add tooltip hints."""
        tab_text = self.tabText(index)

        if tab_text in self.core_tabs:
            tab_bar = self.tabBar()
            if tab_bar:
                tab_bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, None)
                tab_bar.setTabButton(index, QTabBar.ButtonPosition.LeftSide, None)

        hint = "Right-click for options | Drag to detach"
        existing = self.tabToolTip(index)
        if not existing:
            self.setTabToolTip(index, hint)
        elif hint not in existing:
            self.setTabToolTip(index, f"{existing}\n\n{hint}")

    # ── Close / reopen ──────────────────────────────────────────────

    def close_tab(self, index: int) -> None:
        """Close a non-core tab (with confirmation)."""
        if index < 0 or index >= self.count():
            return

        tab_text = self.tabText(index)

        if tab_text in self.core_tabs:
            QMessageBox.information(
                self,
                "Cannot Close",
                f"'{tab_text}' is a core tab and cannot be closed.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Close Tab",
            f"Close the '{tab_text}' tab?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if tab_text in self.tab_factories:
            self.closed_tabs[tab_text] = self.tab_factories[tab_text]

        widget = self.widget(index)
        self.removeTab(index)
        if widget:
            widget.deleteLater()

    def reopen_closed_tab(self, tab_name: str) -> None:
        """Reopen a previously closed tab by name."""
        if tab_name not in self.closed_tabs:
            return
        try:
            widget = self.closed_tabs[tab_name]()
            if widget:
                self.addTab(widget, tab_name)
                self.setCurrentIndex(self.count() - 1)
                del self.closed_tabs[tab_name]
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to reopen tab '{tab_name}': {e}")

    def reopen_all_closed_tabs(self) -> None:
        """Reopen all previously closed tabs."""
        for name in list(self.closed_tabs):
            self.reopen_closed_tab(name)

    # ── Drag-to-detach ──────────────────────────────────────────────

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        """Detect tab drag outside the bar to trigger detachment."""
        if watched != self.tabBar() or event is None:
            return super().eventFilter(watched, event)

        if event.type() == QEvent.Type.MouseButtonPress:
            if (
                isinstance(event, QMouseEvent)
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.drag_start_pos = event.globalPosition().toPoint()

        elif event.type() == QEvent.Type.MouseMove:
            if isinstance(event, QMouseEvent) and (
                event.buttons() & Qt.MouseButton.LeftButton
            ):
                pos = event.globalPosition().toPoint()
                if (pos - self.drag_start_pos).manhattanLength() >= (
                    QApplication.startDragDistance()
                ):
                    local = self.mapFromGlobal(pos)
                    bar = self.tabBar()
                    if bar and not bar.geometry().contains(local):
                        idx = self.currentIndex()
                        if idx >= 0:
                            self.detach_tab(idx, pos)
                            return True

        return super().eventFilter(watched, event)

    def detach_tab(self, index: int, pos: QPoint) -> None:
        """Detach a tab into a separate window."""
        if index < 0 or index >= self.count():
            return
        widget = self.widget(index)
        if not widget:
            return

        text = self.tabText(index)
        icon = self.tabIcon(index)
        self.removeTab(index)

        win = DetachedTabWindow(widget, text, icon, self)
        win.move(pos)
        win.show()
        self.detached_tabs[win] = (widget, text, icon)
        win.tab_reattached.connect(self.reattach_tab)
        self.tab_detached.emit(index, pos)

    def detach_tab_from_menu(self, index: int) -> None:
        """Detach via context menu (uses cursor position)."""
        self.detach_tab(index, QCursor.pos())

    def reattach_tab(self, detached_window: DetachedTabWindow) -> None:
        """Reattach a previously detached tab."""
        if detached_window not in self.detached_tabs:
            return
        widget, text, icon = self.detached_tabs[detached_window]
        if widget.parent():
            widget.setParent(None)
        idx = self.addTab(widget, icon, text)
        self.setCurrentIndex(idx)
        widget.show()
        del self.detached_tabs[detached_window]
        detached_window.close()

    def redock_all_tabs(self) -> None:
        """Redock all detached tabs (suppresses close dialogs)."""
        windows = list(self.detached_tabs)
        for w in windows:
            w.suppress_close_dialog = True
        for w in windows:
            self.reattach_tab(w)

    # ── Context menu ────────────────────────────────────────────────

    def _show_tab_context_menu(self, position: QPoint) -> None:
        """Show right-click menu for a tab."""
        bar = self.tabBar()
        if not bar:
            return
        idx = bar.tabAt(position)
        if idx < 0:
            return

        text = self.tabText(idx)
        menu = QMenu(self)

        if text not in self.core_tabs:
            close_action = QAction("Close Tab", self)
            close_action.triggered.connect(lambda: self.close_tab(idx))
            menu.addAction(close_action)
            menu.addSeparator()

        pop_action = QAction("Pop Out Tab", self)
        pop_action.triggered.connect(lambda: self.detach_tab_from_menu(idx))
        menu.addAction(pop_action)
        menu.addSeparator()

        if self.detached_tabs:
            redock_menu = menu.addMenu("Redock Tabs")
            if redock_menu:
                for win, (_, title, icon) in self.detached_tabs.items():
                    act = QAction(f"Redock: {title}", redock_menu)
                    act.setIcon(icon)
                    act.triggered.connect(lambda checked, w=win: self.reattach_tab(w))
                    redock_menu.addAction(act)
                redock_menu.addSeparator()
                all_act = QAction("Redock All Tabs", redock_menu)
                all_act.triggered.connect(self.redock_all_tabs)
                redock_menu.addAction(all_act)

        if self.closed_tabs:
            open_menu = menu.addMenu("Open Tab")
            if open_menu:
                for name in sorted(self.closed_tabs):
                    act = QAction(name, open_menu)
                    act.triggered.connect(
                        lambda checked, n=name: self.reopen_closed_tab(n)
                    )
                    open_menu.addAction(act)
                open_menu.addSeparator()
                all_act = QAction("Open All Tabs", open_menu)
                all_act.triggered.connect(self.reopen_all_closed_tabs)
                open_menu.addAction(all_act)

        menu.exec(bar.mapToGlobal(position))


class DetachedTabWindow(QMainWindow):
    """Standalone window for a detached tab with redocking support."""

    tab_reattached = pyqtSignal(object)

    def __init__(
        self,
        widget: QWidget,
        title: str,
        icon: QIcon,
        parent_tab_widget: DraggableTabWidget,
    ) -> None:
        super().__init__()
        self.parent_tab_widget = parent_tab_widget
        self.widget = widget
        self.original_title = title
        self.suppress_close_dialog = False

        self.setWindowTitle(title)
        self.setWindowIcon(icon)

        if widget.parent():
            widget.setParent(None)
        self.setCentralWidget(widget)
        self.resize(800, 600)
        self.setMinimumSize(400, 300)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowSystemMenuHint
        )
        widget.show()

        self._setup_menus()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        status = self.statusBar()
        if status:
            status.showMessage("Right-click to redock, or use View menu / Ctrl+D", 0)

    def _setup_menus(self) -> None:
        """Create View menu with redock actions."""
        menubar = self.menuBar()
        if not menubar:
            return

        view_menu = menubar.addMenu("View")
        if not view_menu:
            return

        redock = QAction("Redock Tab", self)
        redock.setShortcut("Ctrl+D")
        redock.triggered.connect(self._trigger_redock)
        view_menu.addAction(redock)

        redock_all = QAction("Redock All Tabs", self)
        redock_all.setShortcut("Ctrl+Shift+D")
        redock_all.triggered.connect(self._trigger_redock_all)
        view_menu.addAction(redock_all)

        view_menu.addSeparator()

        stay = QAction("Always on Top", self)
        stay.setCheckable(True)
        stay.toggled.connect(self._toggle_on_top)
        view_menu.addAction(stay)

    def _show_context_menu(self, position: QPoint) -> None:
        """Right-click context menu for redocking."""
        menu = QMenu(self)

        act = QAction(f"Redock '{self.original_title}'", self)
        act.triggered.connect(self._trigger_redock)
        menu.addAction(act)

        if self.parent_tab_widget and len(self.parent_tab_widget.detached_tabs) > 1:
            all_act = QAction("Redock All Tabs", self)
            all_act.triggered.connect(self._trigger_redock_all)
            menu.addAction(all_act)

        menu.addSeparator()
        close_act = QAction("Close Window", self)
        close_act.triggered.connect(self.close)
        menu.addAction(close_act)

        menu.exec(self.mapToGlobal(position))

    def _trigger_redock(self) -> None:
        self.tab_reattached.emit(self)

    def _trigger_redock_all(self) -> None:
        self.suppress_close_dialog = True
        if hasattr(self.parent_tab_widget, "redock_all_tabs"):
            self.parent_tab_widget.redock_all_tabs()

    def _toggle_on_top(self, checked: bool) -> None:
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(
                self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint
            )
        self.show()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Double-click title bar area to redock."""
        if event.position().y() < 40:
            self._trigger_redock()
        else:
            super().mouseDoubleClickEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """On close: offer redock instead of losing the tab."""
        if self.suppress_close_dialog:
            self._trigger_redock()
            event.ignore()
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Close Window")
        msg.setText(f"What to do with '{self.original_title}'?")
        redock_btn = msg.addButton("Redock", QMessageBox.ButtonRole.YesRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(redock_btn)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.exec()

        if msg.clickedButton() == redock_btn:
            self._trigger_redock()
        event.ignore()


__all__ = ["DetachedTabWindow", "DraggableTabWidget"]
