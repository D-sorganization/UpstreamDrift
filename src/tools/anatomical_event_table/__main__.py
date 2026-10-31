"""Standalone entry point for the anatomical event table (HRG-E4, #13175)."""

from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from src.tools.anatomical_event_table.gui import AnatomicalEventTablePanel

    app = QApplication.instance() or QApplication(sys.argv)
    panel = AnatomicalEventTablePanel()
    panel.setWindowTitle("Anatomical Event Table")
    panel.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
