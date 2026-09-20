"""Standalone entry point for the Matched Swing Results Browser (MS-80, #10353)."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.tools.matched_swing_browser.gui import MatchedSwingBrowserWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MatchedSwingBrowserWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
