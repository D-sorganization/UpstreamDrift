"""Standalone entry point for the Grip Wrench Plots tool (GCV-10, #11716)."""

from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from src.tools.grip_wrench_plots.gui import GripWrenchPlotWidget

    app = QApplication.instance() or QApplication(sys.argv)
    widget = GripWrenchPlotWidget()
    widget.setWindowTitle("Grip Wrench Plots")
    widget.resize(560, 760)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
