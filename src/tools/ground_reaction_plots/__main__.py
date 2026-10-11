"""Standalone entry point for the Ground Reaction Plots tool (GCV-5, #11711)."""

from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from src.tools.ground_reaction_plots.gui import GroundReactionPlotWidget

    app = QApplication.instance() or QApplication(sys.argv)
    widget = GroundReactionPlotWidget()
    widget.setWindowTitle("Ground Reaction Plots")
    widget.resize(960, 820)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
