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
from typing import Any

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCursor, QIcon, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class DockedTabWrapper(QWidget):
    """Wrapper to host QMainWindow and its QMenuBar inside a parent QTabWidget.

    In Qt, when a QMainWindow is parented inside a layout or another widget,
    its native menu bar is hidden or doesn't render. This wrapper extracts
    the menu bar and lays it out explicitly above the main window content
    to keep menus functional and visible.
    """

    def __init__(self, main_window: QMainWindow, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.menu_bar = main_window.menuBar()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Reparent the menu bar and the QMainWindow inside the wrapper
        if self.menu_bar:
            layout.addWidget(self.menu_bar)
            self.menu_bar.show()
        layout.addWidget(self.main_window)
        self.main_window.show()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.main_window, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in ("main_window", "menu_bar")
            or name.startswith("_")
            or hasattr(type(self), name)
            or name in self.__dict__
        ):
            super().__setattr__(name, value)
        else:
            setattr(self.main_window, name, value)


class DraggableTabWidget(QTabWidget):
    """Enhanced QTabWidget with draggable, detachable tabs.

    Signals:
        tab_detached: Emitted when a tab is dragged out (index, position).
        tab_moved: Emitted when tabs are reordered (from_index, to_index).
    """

    tab_detached = pyqtSignal(int, QPoint)
    tab_moved = pyqtSignal(int, int)
    tab_backgrounded = pyqtSignal(str)
    tab_restored = pyqtSignal(str)

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

        # Background tabs: closing a background-eligible tab hides it and
        # retains the live widget + its state instead of deleting it, so it
        # keeps running and can be restored. Keyed by tab title.
        self.background_tabs: dict[str, tuple[QWidget, QIcon]] = {}

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
        if widget is None:
            raise ValueError("widget must be provided")
        if isinstance(widget, QMainWindow) and not isinstance(widget, DockedTabWrapper):
            widget = DockedTabWrapper(widget)
        index = super().addTab(widget, *args)
        self._drop_background_entry(widget)
        self._update_tab_ux(index)
        return index

    def insertTab(self, index: int, widget: QWidget, *args) -> int:  # type: ignore[override]
        """Override to apply UX enhancements on inserted tabs."""
        if index is None:
            raise ValueError("index must be provided")
        if isinstance(widget, QMainWindow) and not isinstance(widget, DockedTabWrapper):
            widget = DockedTabWrapper(widget)
        ret_index = super().insertTab(index, widget, *args)
        self._drop_background_entry(widget)
        self._update_tab_ux(ret_index)
        return ret_index

    def _drop_background_entry(self, widget: QWidget) -> None:
        """Forget any background record for ``widget`` once it is shown again.

        Keeps the background registry consistent when a widget is re-added
        through a side channel (e.g. the console toggle) rather than via
        ``restore_background_tab``.
        """
        for title, (bg_widget, _icon) in list(self.background_tabs.items()):
            if bg_widget is widget:
                del self.background_tabs[title]

    def _update_tab_ux(self, index: int) -> None:
        """Hide close button for core tabs and add tooltip hints."""
        if index is None:
            raise ValueError("index must be provided")
        tab_text = self.tabText(index)

        tab_bar = self.tabBar()
        if tab_bar:
            if tab_bar.contextMenuPolicy() != Qt.ContextMenuPolicy.CustomContextMenu:
                tab_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                tab_bar.customContextMenuRequested.connect(self._show_tab_context_menu)
                tab_bar.installEventFilter(self)

            if tab_text in self.core_tabs:
                tab_bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, None)
                tab_bar.setTabButton(index, QTabBar.ButtonPosition.LeftSide, None)

        hint = "Right-click for options | Drag to detach"
        existing = self.tabToolTip(index)
        if not existing:
            self.setTabToolTip(index, hint)
        elif hint not in existing:
            self.setTabToolTip(index, f"{existing}\n\n{hint}")

    # ── Close / reopen ──────────────────────────────────────────────

    def _is_widget_dirty(self, widget: QWidget | None) -> bool:
        """Check recursively if a widget or its central/embedded widget has unsaved changes."""
        if widget is None:
            return False

        if isinstance(widget, DockedTabWrapper):
            widget = widget.main_window

        # Check attributes and methods
        for attr in ("is_dirty", "isDirty"):
            val = getattr(widget, attr, None)
            if val is not None:
                if callable(val):
                    try:
                        if val():
                            return True
                    except Exception:  # noqa: BLE001
                        pass
                elif bool(val):
                    return True

        # Check central widget if it is a QMainWindow
        if isinstance(widget, QMainWindow):
            central = widget.centralWidget()
            if central is not None and self._is_widget_dirty(central):
                return True

        # Check common inner widget / tool attributes
        for attr in ("widget", "tool", "_widget", "_tool"):
            inner = getattr(widget, attr, None)
            if (
                inner is not None
                and inner is not widget
                and self._is_widget_dirty(inner)
            ):
                return True

        return False

    def close_tab(self, index: int) -> None:
        """Close a non-core tab (with confirmation based on user preference)."""
        if index is None:
            raise ValueError("index must be provided")
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

        # Check user preferences for tab close confirmation behavior
        try:
            from src.shared.python.ui.preferences_dialog import UserPreferences

            prefs = UserPreferences.load()
            confirm_pref = getattr(prefs, "confirm_close_tabs", "unsaved")
        except Exception:  # noqa: BLE001
            confirm_pref = "unsaved"

        widget = self.widget(index)

        need_prompt = True
        if confirm_pref == "never":
            need_prompt = False
        elif confirm_pref == "unsaved":
            need_prompt = self._is_widget_dirty(widget)

        if need_prompt:
            reply = QMessageBox.question(
                self,
                "Close Tab",
                f"Close the '{tab_text}' tab?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Background-eligible tabs keep running hidden instead of being
        # destroyed. This subsumes the legacy ``prevent_deletion_on_close``
        # flag into a single, generic lifecycle (DRY).
        if widget is not None and self._is_background_eligible(widget):
            self._background_tab(index, tab_text, widget)
            return

        if tab_text in self.tab_factories:
            self.closed_tabs[tab_text] = self.tab_factories[tab_text]

        self.removeTab(index)
        if widget:
            widget.deleteLater()

    # ── Background (keep-running-hidden) lifecycle ──────────────────

    @staticmethod
    def _is_background_eligible(widget: QWidget) -> bool:
        """Return True if closing ``widget`` should background it, not delete.

        Opt-in is generic: any widget carrying a truthy ``background_eligible``
        attribute qualifies. The legacy ``prevent_deletion_on_close`` flag is
        honoured as an alias so existing call sites keep working.
        """
        return bool(
            getattr(widget, "background_eligible", False)
            or getattr(widget, "prevent_deletion_on_close", False)
        )

    def add_background_tab(self, widget: QWidget, *args) -> int:
        """Add a tab whose widget keeps running when its tab is closed.

        Marks ``widget`` background-eligible and adds it like ``addTab``.
        Returns the new tab index.
        """
        if widget is None:
            raise ValueError("widget must be provided")
        widget.background_eligible = True
        return self.addTab(widget, *args)

    def _background_tab(self, index: int, title: str, widget: QWidget) -> None:
        """Remove ``widget``'s tab but retain the live widget + state."""
        if not title:
            raise ValueError("title must be provided for a background tab")
        icon = self.tabIcon(index)
        self.removeTab(index)
        widget.setParent(None)
        widget.hide()
        self.background_tabs[title] = (widget, icon)
        self.tab_backgrounded.emit(title)

    def list_background_tabs(self) -> list[str]:
        """Return titles of tabs currently running hidden in the background."""
        return sorted(self.background_tabs)

    def restore_background_tab(self, title: str) -> None:
        """Restore a backgrounded tab back into the tab bar by title."""
        if title is None:
            raise ValueError("title must be provided")
        entry = self.background_tabs.pop(title, None)
        if entry is None:
            return
        widget, icon = entry
        if widget.parent():
            widget.setParent(None)
        idx = self.addTab(widget, icon, title)
        widget.show()
        self.setCurrentIndex(idx)
        self.tab_restored.emit(title)

    def restore_all_background_tabs(self) -> None:
        """Restore every backgrounded tab."""
        for title in list(self.background_tabs):
            self.restore_background_tab(title)

    def reopen_closed_tab(self, tab_name: str) -> None:
        """Reopen a previously closed tab by name."""
        if tab_name is None:
            raise ValueError("tab_name must be provided")
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
        """Detect tab drag outside the bar to trigger detachment or right-click to show menu."""
        if watched != self.tabBar() or event is None:
            return super().eventFilter(watched, event)

        if event.type() == QEvent.Type.MouseButtonPress:
            if (
                isinstance(event, QMouseEvent)
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.drag_start_pos = event.globalPosition().toPoint()

        elif (
            event.type() == QEvent.Type.MouseMove
            and isinstance(event, QMouseEvent)
            and (event.buttons() & Qt.MouseButton.LeftButton)
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
        elif (
            event.type() == QEvent.Type.MouseButtonRelease
            and isinstance(event, QMouseEvent)
            and event.button() == Qt.MouseButton.RightButton
        ):
            position = event.position().toPoint()
            bar = self.tabBar()
            if bar is not None and bar.tabAt(position) >= 0:
                self._show_tab_context_menu(position)
                return True

        return super().eventFilter(watched, event)

    def detach_tab(self, index: int, pos: QPoint) -> None:
        """Detach a tab into a separate window."""
        if index is None:
            raise ValueError("index must be provided")
        if index < 0 or index >= self.count():
            return
        widget = self.widget(index)
        if not widget:
            return

        text = self.tabText(index)
        icon = self.tabIcon(index)
        self.removeTab(index)

        win = DetachedTabWindow(widget, text, icon, self)
        try:
            from src.shared.python.theme import get_theme_manager

            mgr = get_theme_manager()
            if mgr:
                mgr.apply_theme_to_window(win)
        except Exception:  # noqa: BLE001
            pass
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
        if detached_window is None:
            raise ValueError("detached_window must be provided")
        if detached_window not in self.detached_tabs:
            return
        widget, text, icon = self.detached_tabs[detached_window]

        if isinstance(widget, DockedTabWrapper):
            main_window = widget.main_window
            menu_bar = widget.menu_bar

            # Remove the View menu added by DetachedTabWindow
            if hasattr(detached_window, "_view_menu") and detached_window._view_menu:
                if menu_bar:
                    menu_bar.removeAction(detached_window._view_menu.menuAction())
                detached_window._view_menu.deleteLater()
                detached_window._view_menu = None

            # Remove them from DetachedTabWindow
            main_window.setParent(None)
            if menu_bar:
                menu_bar.setParent(None)
                widget.layout().addWidget(menu_bar)
                menu_bar.show()
            widget.layout().addWidget(main_window)
            main_window.show()

        if widget.parent():
            widget.setParent(None)
        idx = self.addTab(widget, icon, text)
        self.setCurrentIndex(idx)
        widget.show()
        del self.detached_tabs[detached_window]
        detached_window.suppress_close_dialog = True
        detached_window.close()

    def redock_all_tabs(self) -> None:
        """Redock all detached tabs (suppresses close dialogs)."""
        windows = list(self.detached_tabs)
        for w in windows:
            w.suppress_close_dialog = True
        for w in windows:
            self.reattach_tab(w)

    # ── Context menu ────────────────────────────────────────────────

    def _show_tab_context_menu(self, position: QPoint) -> None:  # noqa: C901
        """Show right-click menu for a tab."""
        if position is None:
            raise ValueError("position must be provided")
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

        pop_action = QAction("Undock Tab", self)
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
        if widget is None:
            raise ValueError("widget must be provided")
        super().__init__()
        self.parent_tab_widget = parent_tab_widget
        self.widget = widget
        self.original_title = title
        self.suppress_close_dialog = False

        self.setWindowTitle(title)
        self.setWindowIcon(icon)

        if isinstance(widget, DockedTabWrapper):
            main_window = widget.main_window
            menu_bar = widget.menu_bar
            if main_window.parent():
                main_window.setParent(None)
            if menu_bar:
                if menu_bar.parent():
                    menu_bar.setParent(None)
                self.setMenuBar(menu_bar)
                menu_bar.show()
            self.setCentralWidget(main_window)
            main_window.show()
        else:
            if widget.parent():
                widget.setParent(None)
            self.setCentralWidget(widget)
            widget.show()

        self.resize(800, 600)
        self.setMinimumSize(400, 300)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowSystemMenuHint
        )

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

        self._view_menu = menubar.addMenu("View")
        if not self._view_menu:
            return

        redock = QAction("Redock Tab", self)
        redock.setShortcut("Ctrl+D")
        redock.triggered.connect(self._trigger_redock)
        self._view_menu.addAction(redock)

        redock_all = QAction("Redock All Tabs", self)
        redock_all.setShortcut("Ctrl+Shift+D")
        redock_all.triggered.connect(self._trigger_redock_all)
        self._view_menu.addAction(redock_all)

        self._view_menu.addSeparator()

        stay = QAction("Always on Top", self)
        stay.setCheckable(True)
        stay.toggled.connect(self._toggle_on_top)
        self._view_menu.addAction(stay)

    def _show_context_menu(self, position: QPoint) -> None:
        """Right-click context menu for redocking."""
        if position is None:
            raise ValueError("position must be provided")
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

    def closeEvent(self, event: Any) -> None:  # type: ignore[override]
        """On close: offer redock instead of losing the tab."""
        if event is None:
            raise ValueError("event must be provided")
        if self.suppress_close_dialog:
            event.accept()
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Close Window")
        msg.setText(f"What would you like to do with the '{self.original_title}' tab?")
        redock_btn = msg.addButton("Redock", QMessageBox.ButtonRole.YesRole)
        msg.addButton("Close Tab", QMessageBox.ButtonRole.NoRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(redock_btn)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked and clicked.text() == "Redock":
            self.suppress_close_dialog = True
            self._trigger_redock()
            event.accept()
        elif clicked and clicked.text() == "Close Tab":
            self.suppress_close_dialog = True
            # Remove from parent's detached tabs registry
            if self.parent_tab_widget and self in self.parent_tab_widget.detached_tabs:
                del self.parent_tab_widget.detached_tabs[self]
            # Background-eligible widgets keep running hidden; others are
            # destroyed. Mirrors DraggableTabWidget.close_tab (DRY intent).
            if self.widget:
                parent_tabs = self.parent_tab_widget
                background = DraggableTabWidget._is_background_eligible(self.widget)
                if background and parent_tabs is not None:
                    if isinstance(self.widget, DockedTabWrapper):
                        self.widget.main_window.setParent(None)
                    self.widget.setParent(None)
                    self.widget.hide()
                    parent_tabs.background_tabs[self.original_title] = (
                        self.widget,
                        self.windowIcon(),
                    )
                    parent_tabs.tab_backgrounded.emit(self.original_title)
                else:
                    if isinstance(self.widget, DockedTabWrapper):
                        self.widget.main_window.deleteLater()
                    self.widget.deleteLater()
            event.accept()
        else:
            event.ignore()


__all__ = ["DetachedTabWindow", "DraggableTabWidget"]
