"""Standalone entry point for the Pose Subscriber demo.

Wraps :class:`MainWidget` in a tiny ``QMainWindow`` so the tool can
launch outside of the embedded host as well. The launcher uses this
path when :class:`LaunchMode.NEW_WINDOW` is selected; the embed path
goes through :class:`_PoseSubscriberDemoEmbedAdapter`.
"""

from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from src.tools.pose_subscriber_demo.gui import MainWidget


def get_dockable_ui() -> QtWidgets.QMainWindow:
    """Return the main window instance for docking in the unified launcher."""
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Pose Subscriber (demo)")
    widget = MainWidget(win)
    win.setCentralWidget(widget)
    win.resize(640, 640)
    return win

def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m src.tools.pose_subscriber_demo``."""
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Pose Subscriber (demo)")
    widget = MainWidget(win)
    win.setCentralWidget(widget)
    win.resize(640, 640)

    def _cleanup_on_quit() -> None:
        widget.cleanup()

    app.aboutToQuit.connect(_cleanup_on_quit)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
