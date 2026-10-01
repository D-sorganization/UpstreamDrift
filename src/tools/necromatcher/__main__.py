"""Standalone Necromatcher launcher."""

from __future__ import annotations
import sys
from PyQt6.QtWidgets import QApplication
from .gui import NecromatcherWidget


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    widget = NecromatcherWidget()
    widget.setWindowTitle("Necromatcher")
    widget.resize(1100, 800)
    app.aboutToQuit.connect(widget.cleanup)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
