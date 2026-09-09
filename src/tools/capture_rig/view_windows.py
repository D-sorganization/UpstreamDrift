"""Qt window presentation without duplicating a camera or video player (#9913)."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget


def window_shortcuts(
    widget: QWidget, toggle: Callable[[], None], leave: Callable[[], None]
) -> None:
    """Install familiar F11/Escape behavior for the existing viewing window."""
    for key, action in (("F11", toggle), ("Escape", leave)):
        shortcut = QShortcut(QKeySequence(key), widget)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(action)


class _ViewWindow(QMainWindow):
    """Closing returns the original widget to its host, like the app's pop-outs."""

    def __init__(self, parent: QWidget, redock: Callable[[], None]) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Live Cameras — Capture Rig")
        self._redock = redock

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        self._redock()
        if event is not None:
            event.ignore()


class CentralView:
    """Temporarily move a host's central widget into a real Qt window.

    The widget and its transport/camera state are retained. A visible return
    button occupies its original location; closing or resetting redocks it.
    """

    def __init__(self, host: QMainWindow) -> None:
        self.host = host
        self.window = _ViewWindow(host, self.redock)
        self._fullscreen_was_detached: bool | None = None
        self.placeholder = QWidget(host)
        layout = QVBoxLayout(self.placeholder)
        message = QLabel("Live cameras are open in another window.")
        message.setWordWrap(True)
        button = QPushButton("Return Live Cameras Here")
        button.clicked.connect(self.redock)
        layout.addStretch()
        layout.addWidget(message)
        layout.addWidget(button)
        layout.addStretch()
        self.placeholder.hide()
        content = host.centralWidget()
        if content is not None:
            window_shortcuts(content, self.fullscreen, self.leave_fullscreen)

    @property
    def detached(self) -> bool:
        return self.window.centralWidget() is not None

    def pop_out(self) -> None:
        """Show the same central content in a movable window on this screen."""
        if not self.detached:
            content = self.host.takeCentralWidget()
            if content is None:
                return
            self.host.setCentralWidget(self.placeholder)
            self.window.setCentralWidget(content)
            screen = self.host.screen()
            if screen is not None:
                available = screen.availableGeometry()
                self.window.resize(
                    min(1000, available.width()), min(720, available.height())
                )
                self.window.move(available.center() - self.window.rect().center())
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def redock(self) -> None:
        """Restore content without destroying or restarting it."""
        if not self.detached:
            return
        self._fullscreen_was_detached = None
        self.window.showNormal()
        content = self.window.takeCentralWidget()
        self.host.takeCentralWidget()
        self.placeholder.hide()
        self.host.setCentralWidget(content)
        self.window.hide()
        if content is not None:
            content.show()

    def fullscreen(self) -> None:
        """Toggle fullscreen; Escape restores the previous presentation."""
        if self.window.isFullScreen():
            self.leave_fullscreen()
            return
        self._fullscreen_was_detached = self.detached
        self.pop_out()
        self.window.showFullScreen()

    def leave_fullscreen(self) -> None:
        if self._fullscreen_was_detached is None:
            return
        was_detached = self._fullscreen_was_detached
        self._fullscreen_was_detached = None
        self.window.showNormal()
        if not was_detached:
            self.redock()
