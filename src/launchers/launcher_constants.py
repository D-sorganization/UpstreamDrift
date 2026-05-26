"""Shared constants and lazy imports for the UpstreamDriftLauncher."""

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
import contextlib

# Configure Logging using centralized module
configure_gui_logging()
logger = get_logger(__name__)

# Constants
import os

# Allow environment variable override for REPOS_ROOT, fallback to relative path
REPOS_ROOT = Path(
    os.environ.get(
        "UPSTREAM_DRIFT_REPOS_ROOT", Path(__file__).parent.parent.parent.resolve()
    )
).resolve()


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
    except (ImportError, NotImplementedError):
        # Graceful fallback if platformdirs is somehow unavailable or unsupported at runtime
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
    assert ".kiro" not in str(new_dir), (
        f"Config dir must not be under .kiro/: {new_dir}"
    )
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

    LARGE = 0
    MEDIUM = 1
    SMALL = 2
    LIST_LARGE = 3
    LIST_SMALL = 4

    # Backward compat alias
    LIST = 3  # maps to LIST_LARGE


# (tile_scale, columns, show_description, is_list)
_VIEW_MODE_TABLE: dict[ViewMode, tuple[float, int, bool, bool]] = {
    ViewMode.LARGE: (1.0, 4, False, False),
    ViewMode.MEDIUM: (0.5, 6, False, False),
    ViewMode.SMALL: (0.35, 8, False, False),
    ViewMode.LIST_LARGE: (0.30, 1, True, True),
    ViewMode.LIST_SMALL: (0.20, 1, False, True),
}


def view_mode_settings(mode: ViewMode) -> tuple[float, int, bool, bool]:
    """Return the (tile_scale, columns, show_description, is_list) for a mode.

    Raises:
        TypeError: if ``mode`` is not a :class:`ViewMode`.
        ValueError: if ``mode`` is not one of the known view modes.
    """
    is_vm = isinstance(mode, ViewMode) or (
        hasattr(mode, "__class__") and mode.__class__.__name__ == "ViewMode"
    )
    if not is_vm:
        raise TypeError(f"mode must be a ViewMode IntEnum, got {type(mode).__name__}")
    # Convert to local ViewMode enum instance if it's a reloaded/foreign one
    if not isinstance(mode, ViewMode):
        try:
            mode = ViewMode(int(mode))
        except (ValueError, TypeError):
            with contextlib.suppress(AttributeError, KeyError):
                mode = ViewMode[mode.name]
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


def _load_docker_profiles() -> tuple[str, ...]:
    """Load Docker profile names from docker/profiles.yaml.

    Falls back to a legacy hard-coded tuple if the YAML cannot be read so
    that the launcher stays functional in environments where the repo root
    is not accessible (e.g. a sandboxed test without the full tree).

    DbC postcondition: returned tuple is non-empty and contains only strings.
    """
    _legacy = ("slim", "standard", "research", "biomech", "full", "gpu-training")
    profiles_path = REPOS_ROOT / "docker" / "profiles.yaml"
    try:
        text = profiles_path.read_text(encoding="utf-8")
    except OSError:
        logger.debug(
            "docker/profiles.yaml not found at %s; using legacy profile list",
            profiles_path,
        )
        return _legacy

    names: list[str] = []
    in_profiles = False
    for raw in text.splitlines():
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if indent == 0 and stripped.startswith("profiles:"):
            in_profiles = True
            continue
        if in_profiles and indent == 2 and stripped.endswith(":"):
            name = stripped.rstrip(":")
            if name and not name.startswith("#"):
                names.append(name)

    if not names:
        logger.warning(
            "No profiles found in %s; using legacy profile list", profiles_path
        )
        return _legacy

    result = tuple(names)
    assert all(isinstance(n, str) for n in result), "All profile names must be strings"
    return result


DOCKER_STAGES: tuple[str, ...] = _load_docker_profiles()


def validate_docker_stage(stage: str) -> str:
    """Validate Docker profile names used by launcher components.

    Accepts any profile name declared in docker/profiles.yaml (loaded at
    module import time via :func:`_load_docker_profiles`).

    Args:
        stage: The profile/stage string to validate.

    Returns:
        The validated stage string (unchanged).

    Raises:
        ValueError: If *stage* is not a known Docker profile.
    """
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
    if importlib.util.find_spec("src.shared.python.ai.gui"):
        AI_AVAILABLE = True
    else:
        AI_AVAILABLE = False
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
