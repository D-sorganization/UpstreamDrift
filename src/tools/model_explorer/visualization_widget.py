"""3D visualization widget for URDF preview."""

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.shared.python.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

# Check MuJoCo availability
MUJOCO_AVAILABLE = False
try:
    pass

    from .mujoco_viewer import MuJoCoViewerWidget  # noqa: F401

    MUJOCO_AVAILABLE = True
    logger.info("MuJoCo 3D viewer available")
except ImportError as e:
    logger.info(f"MuJoCo not available, using fallback grid view: {e}")
