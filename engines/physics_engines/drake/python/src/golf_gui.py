"""Golf Analysis Suite GUI Entry Point."""

import logging
import sys

from PyQt6 import QtWidgets

from .drake_gui_app import DrakeSimApp
from .logger_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Golf Analysis GUI."""
    setup_logging()
    logger.info("Starting Golf Analysis Suite (PyQt + Drake)...")

    app = QtWidgets.QApplication(sys.argv)

    # Create and show the main window
    window = DrakeSimApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
