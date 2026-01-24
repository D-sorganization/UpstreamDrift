#!/usr/bin/env python3
"""Launch script for the Interactive URDF Generator."""

import sys

from src.shared.python.logging_config import configure_gui_logging, get_logger
from src.shared.python.path_utils import setup_import_paths

# Add the project root to the Python path
setup_import_paths()

from src.tools.urdf_generator.main_window import main  # noqa: E402

if __name__ == "__main__":
    # Set up logging
    configure_gui_logging()
    logger = get_logger(__name__)
    logger.info("Starting Interactive URDF Generator")

    try:
        main()
    except Exception as e:
        logger.error(f"Failed to start URDF Generator: {e}")
        sys.exit(1)
