"""Standalone entry point: ``python -m src.tools.spine_viewer``."""

from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from src.tools.spine_viewer.gui import SpineViewerWidget


def get_dockable_ui() -> QtWidgets.QMainWindow:
    """Return the main window for docking in the unified launcher."""
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Spine Viewer")
    win.setCentralWidget(SpineViewerWidget(win))
    win.resize(900, 640)
    return win


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m src.tools.spine_viewer``."""
    args = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(args)
    win = get_dockable_ui()
    widget = win.centralWidget()
    app.aboutToQuit.connect(widget.cleanup)  # type: ignore[union-attr]
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
