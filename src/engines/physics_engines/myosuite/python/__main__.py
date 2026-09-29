"""Standalone entry point for the MyoSuite dashboard.

Run with ``python -m src.engines.physics_engines.myosuite.python``. The
launcher prefers the embedded path (see ``default_launch: tab`` in
``models.yaml``); this entry point exists for back-compat and for
developers who want to drive the dashboard outside the launcher.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from src.shared.python.logging_pkg.logging_config import get_logger

from .gui import MainWindow

logger = get_logger(__name__)


def main() -> int:
    """Launch the standalone MyoSuite dashboard window."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    logger.info("MyoSuite dashboard launched (standalone)")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
