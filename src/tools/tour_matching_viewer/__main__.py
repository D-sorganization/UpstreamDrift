"""Standalone entry point for the Tour Matching Viewer."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.tools.tour_matching_viewer.gui import TourMatchingViewerWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = TourMatchingViewerWindow()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.exists():
            window.widget.load_file(path)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
