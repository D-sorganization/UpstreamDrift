"""Shared constants and lazy imports for the GolfLauncher."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from enum import IntEnum
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


def _get_config_dir() -> Path:
    """Return the platform-appropriate config directory for upstream-drift.

    Migrates from the defunct .kiro/ path (issue #5713) to a proper location:
    - Linux/macOS: ~/.config/upstream-drift/launcher  (via platformdirs)
    - Windows:     %LOCALAPPDATA%/UpstreamDrift/launcher (via platformdirs)

    Backward compatibility: if the old .kiro/launcher path exists and the new
    path does not yet contain a layout.json, existing config is copied on first
    run.

    DbC postcondition: returned path does not contain '.kiro'.
    """
    try:
        from platformdirs import user_config_dir

        new_dir = Path(user_config_dir("upstream-drift")) / "launcher"
    except ImportError:
        # Graceful fallback if platformdirs is somehow unavailable at runtime
        if sys.platform == "win32":
            new_dir = Path.home() / "AppData" / "Local" / "UpstreamDrift" / "launcher"
        else:
            new_dir = Path.home() / ".config" / "upstream-drift" / "launcher"

    # Backward-compatibility migration: copy config from old .kiro/ path on first run
    old_dir = REPOS_ROOT / ".kiro" / "launcher"
    if old_dir.exists() and not (new_dir / "layout.json").exists():
        try:
            new_dir.mkdir(parents=True, exist_ok=True)
            for item in old_dir.iterdir():
                dest = new_dir / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest))
                    else:
                        shutil.copy2(str(item), str(dest))
            logger.info(
                "Migrated launcher config from %s to %s (issue #5713)",
                old_dir,
                new_dir,
            )
        except OSError as exc:
            logger.warning("Could not migrate old .kiro/ config: %s", exc)

    # DbC postcondition
    assert ".kiro" not in str(new_dir), f"Config dir must not be under .kiro/: {new_dir}"
    return new_dir


CONFIG_DIR = _get_config_dir()
LAYOUT_CONFIG_FILE = CONFIG_DIR / "layout.json"
GRID_COLUMNS = 4  # Changed to 3x4 grid (12 tiles total)

# Tile sizing constants for the resizable launcher tile system.
# Reference (1.0x) values match the original hard-coded layout.
TILE_SCALE_MIN = 0.25
TILE_SCALE_MAX = 2.0
TILE_SCALE_DEFAULT = 0.5
TILE_BASE_IMAGE_PX = 200
TILE_BASE_FONT_PT = 11
TILE_BASE_PADDING_PX = 12
# Floor for derived font sizes so chip/desc text stays legible at min scale.
TILE_MIN_FONT_PT = 9


class ViewMode(IntEnum):
    """Launcher tile-grid layout modes.

    Each mode maps to a (tile_scale, columns, show_description, is_list) tuple
    via :func:`view_mode_settings`.
    """

    COMFORTABLE = 0
    COMPACT = 1
    DENSE = 2
    LIST = 3


# (tile_scale, columns, show_description, is_list)
_VIEW_MODE_TABLE: dict[ViewMode, tuple[float, int, bool, bool]] = {
    ViewMode.COMFORTABLE: (1.0, 4, False, False),
    ViewMode.COMPACT: (0.5, 6, False, False),
    ViewMode.DENSE: (0.35, 8, False, False),
    ViewMode.LIST: (0.30, 1, True, True),
}


def view_mode_settings(mode: ViewMode) -> tuple[float, int, bool, bool]:
    """Return the (tile_scale, columns, show_description, is_list) for a mode.

    Raises:
        TypeError: if ``mode`` is not a :class:`ViewMode`.
        ValueError: if ``mode`` is not one of the known view modes.
    """
    if not isinstance(mode, ViewMode):
        raise TypeError(f"mode must be a ViewMode IntEnum, got {type(mode).__name__}")
    if mode not in _VIEW_MODE_TABLE:
        raise ValueError(f"Unknown ViewMode: {mode!r}")
    return _VIEW_MODE_TABLE[mode]


def validate_tile_scale(scale: float) -> float:
    """Validate and clamp a tile-scale value into [MIN, MAX].

    Raises:
        TypeError: if ``scale`` is not a real number.
        ValueError: if ``scale`` is NaN or non-positive.
    """
    if isinstance(scale, bool) or not isinstance(scale, int | float):
        raise TypeError(f"tile_scale must be a real number, got {type(scale).__name__}")
    f = float(scale)
    if f != f:  # NaN check
        raise ValueError("tile_scale must not be NaN")
    if f <= 0.0:
        raise ValueError(f"tile_scale must be positive, got {f}")
    if f < TILE_SCALE_MIN:
        return TILE_SCALE_MIN
    if f > TILE_SCALE_MAX:
        return TILE_SCALE_MAX
    return f


def scaled_image_px(scale: float) -> int:
    """Return the tile image size in pixels for a given tile_scale."""
    return max(1, int(TILE_BASE_IMAGE_PX * validate_tile_scale(scale)))


def scaled_font_pt(scale: float, base_pt: int = TILE_BASE_FONT_PT) -> int:
    """Return a font point size for a given tile_scale, with a legibility floor."""
    return max(TILE_MIN_FONT_PT, int(round(base_pt * validate_tile_scale(scale))))


def scaled_padding_px(scale: float) -> int:
    """Return padding/margin in pixels for a given tile_scale."""
    return max(2, int(round(TILE_BASE_PADDING_PX * validate_tile_scale(scale))))


DOCKER_STAGES: tuple[str, ...] = ("all", "mujoco", "pinocchio", "drake", "base")


def validate_docker_stage(stage: str) -> str:
    """Validate Docker build stage names used by launcher components."""
    if stage not in DOCKER_STAGES:
        allowed = ", ".join(DOCKER_STAGES)
        raise ValueError(f"Invalid Docker stage '{stage}'. Expected one of: {allowed}")
    return stage


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
