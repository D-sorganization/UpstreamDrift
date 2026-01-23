#!/usr/bin/env python3
"""Launch script for the Interactive URDF Generator."""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tools.urdf_generator.main_window import main  # noqa: E402

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(project_root / "urdf_generator.log"),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Interactive URDF Generator")

    try:
        main()
    except Exception as e:
        logger.error(f"Failed to start URDF Generator: {e}")
        sys.exit(1)
