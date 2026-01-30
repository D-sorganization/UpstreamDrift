#!/usr/bin/env python3
"""Launch script for the Model Explorer."""

import sys
from pathlib import Path

# Add project root to path for src imports when run as standalone script
# Path: src/tools/model_explorer/launch_model_explorer.py -> need 4 parents
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# CRITICAL: Import MuJoCo BEFORE any Qt imports to avoid DLL conflicts on Windows.
# Qt's OpenGL context initialization conflicts with MuJoCo's plugin loading.
try:
    import mujoco  # noqa: F401
except ImportError:
    pass  # MuJoCo not installed, will fall back to grid view
except OSError:
    # Handle DLL loading failures on Windows (e.g., Python 3.13 + mujoco 3.3.4)
    pass  # Will fall back to grid view

from src.shared.python.logging_config import (  # noqa: E402
    configure_gui_logging,
    get_logger,
)
from src.tools.model_explorer.main_window import main  # noqa: E402

if __name__ == "__main__":
    # Set up logging
    configure_gui_logging()
    logger = get_logger(__name__)
    logger.info("Starting Model Explorer")

    try:
        main()
    except Exception as e:
        logger.error(f"Failed to start Model Explorer: {e}")
        sys.exit(1)
