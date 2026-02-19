"""Shared constants and lazy imports for the GolfLauncher."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import (
    configure_gui_logging,
    get_logger,
)

# Configure Logging using centralized module
configure_gui_logging()
logger = get_logger(__name__)

# Constants
REPOS_ROOT = Path(__file__).parent.parent.parent.resolve()

CONFIG_DIR = REPOS_ROOT / ".kiro" / "launcher"
LAYOUT_CONFIG_FILE = CONFIG_DIR / "layout.json"
GRID_COLUMNS = 4  # Changed to 3x4 grid (12 tiles total)

DOCKER_STAGES = ["all", "mujoco", "pinocchio", "drake", "base"]

# Windows-specific subprocess constants
CREATE_NO_WINDOW: int
CREATE_NEW_CONSOLE: int

if sys.platform == "win32":
    try:
        CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        CREATE_NEW_CONSOLE = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    except AttributeError:
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_CONSOLE = 0x00000010
else:
    CREATE_NO_WINDOW = 0
    CREATE_NEW_CONSOLE = 0


# Lazy imports for heavy modules (mutable holder avoids 'global' keyword)
_lazy_imports: dict[str, Any] = {
    "EngineManager": None,
    "EngineType": None,
    "ModelRegistry": None,
}


def _lazy_load_engine_manager() -> tuple[Any, Any]:
    """Lazily load EngineManager to speed up initial import."""
    if _lazy_imports["EngineManager"] is None:
        from src.shared.python.engine_core.engine_manager import EngineManager as _EM
        from src.shared.python.engine_core.engine_manager import EngineType as _ET

        _lazy_imports["EngineManager"] = _EM
        _lazy_imports["EngineType"] = _ET
    return _lazy_imports["EngineManager"], _lazy_imports["EngineType"]


def _lazy_load_model_registry() -> Any:
    """Lazily load ModelRegistry to speed up initial import."""
    if _lazy_imports["ModelRegistry"] is None:
        from src.shared.python.config.model_registry import ModelRegistry as _MR

        _lazy_imports["ModelRegistry"] = _MR
    return _lazy_imports["ModelRegistry"]


# Feature availability checks using importlib for graceful degradation
THEME_AVAILABLE = importlib.util.find_spec("src.shared.python.theme") is not None

AI_AVAILABLE: bool
try:
    importlib.util.find_spec("src.shared.python.ai.gui")
    # Actually try importing to verify it works (catches missing deps)
    import src.shared.python.ai.gui  # noqa: F401

    AI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AI_AVAILABLE = False

HELP_SYSTEM_AVAILABLE: bool
try:
    import src.shared.python.help_system  # noqa: F401

    HELP_SYSTEM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HELP_SYSTEM_AVAILABLE = False

UI_COMPONENTS_AVAILABLE: bool
try:
    import src.shared.python.ui  # noqa: F401

    UI_COMPONENTS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    UI_COMPONENTS_AVAILABLE = False
