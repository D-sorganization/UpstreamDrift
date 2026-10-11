"""Standalone entry point for the Impact Parameters panel (GCV-17, #11723)."""

from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from src.tools.impact_parameters_panel.gui import ImpactParametersWidget

    app = QApplication.instance() or QApplication(sys.argv)
    widget = ImpactParametersWidget()
    widget.setWindowTitle("Impact Parameters")
    widget.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
