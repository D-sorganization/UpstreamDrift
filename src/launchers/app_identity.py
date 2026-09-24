"""Canonical shared application identity and asset resolution for UpstreamDrift.

Provides a dependency-free, immutable source of truth for:
- Windows AppUserModelID (for consistent taskbar grouping and pinning)
- Application names and shortcut filenames
- Multi-resolution icon hierarchy and resolution
- Favicon verification and synchronization between desktop assets and web UI
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "APP_DISPLAY_NAME",
    "APP_NAME",
    "APP_USER_MODEL_ID",
    "CANONICAL_ICON_CANDIDATES",
    "GOLF_SUITE_NAME",
    "LEGACY_SHORTCUT_FILENAME",
    "SHORTCUT_FILENAME",
    "AppIdentity",
    "get_app_identity",
    "resolve_canonical_icon",
    "resolve_launcher_executable",
    "sync_web_favicon",
    "verify_favicon_consistency",
]

_logger = logging.getLogger(__name__)

# Canonical Windows Application User Model ID used across GUI, taskbar, and shortcuts
APP_USER_MODEL_ID: str = "D-sorganization.UpstreamDrift"

# Human-readable and system application names
APP_NAME: str = "UpstreamDrift"
APP_DISPLAY_NAME: str = "Upstream Drift"
GOLF_SUITE_NAME: str = "Golf Modeling Suite"

# Standard Windows shortcut filenames
SHORTCUT_FILENAME: str = "UpstreamDrift.lnk"
LEGACY_SHORTCUT_FILENAME: str = "Golf Modeling Suite.lnk"

# Prioritized list of icon candidate filenames (Windows multi-res ICO prioritized over PNG)
CANONICAL_ICON_CANDIDATES: tuple[str, ...] = (
    "golf_logo.ico",
    "golf_suite_unified.ico",
    "golf_robot_windows_optimized.ico",
    "golf_robot_ultra_sharp.ico",
    "golf_robot_icon.ico",
    "golf_icon.ico",
    "golf_logo.png",
)


@dataclass(frozen=True)
class AppIdentity:
    """Immutable application identity contract."""

    app_id: str
    app_name: str
    display_name: str
    shortcut_name: str
    legacy_shortcut_name: str

    def __post_init__(self) -> None:
        """Validate identity contract invariants (Design by Contract)."""
        if not isinstance(self.app_id, str) or not self.app_id.strip():
            raise ValueError("app_id must be a non-empty string")
        if not isinstance(self.app_name, str) or not self.app_name.strip():
            raise ValueError("app_name must be a non-empty string")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        if not self.shortcut_name.endswith(".lnk"):
            raise ValueError("shortcut_name must end with .lnk")
        if not self.legacy_shortcut_name.endswith(".lnk"):
            raise ValueError("legacy_shortcut_name must end with .lnk")


_CANONICAL_IDENTITY = AppIdentity(
    app_id=APP_USER_MODEL_ID,
    app_name=APP_NAME,
    display_name=APP_DISPLAY_NAME,
    shortcut_name=SHORTCUT_FILENAME,
    legacy_shortcut_name=LEGACY_SHORTCUT_FILENAME,
)


def get_app_identity() -> AppIdentity:
    """Return the canonical immutable AppIdentity."""
    return _CANONICAL_IDENTITY


def resolve_canonical_icon(assets_dir: Path | None = None) -> Path | None:
    """Resolve the best available icon file from the assets directory.

    Args:
        assets_dir: Directory containing icon assets. Defaults to
            ``src/launchers/assets`` relative to this repository.

    Returns:
        Path to the first existing candidate icon, or None if none exist.
    """
    if assets_dir is None:
        assets_dir = Path(__file__).resolve().parent / "assets"

    if not assets_dir.is_dir():
        _logger.warning("Icon assets directory does not exist: %s", assets_dir)
        return None

    for candidate_name in CANONICAL_ICON_CANDIDATES:
        candidate_path = assets_dir / candidate_name
        if candidate_path.is_file():
            return candidate_path

    _logger.warning("No canonical icon candidate found in %s", assets_dir)
    return None


def resolve_launcher_executable(
    python_exe: Path | None = None, prefer_windowed: bool = True
) -> Path:
    """Resolve the appropriate Python executable for shortcut launching.

    On Windows, when ``prefer_windowed`` is True, prefers ``pythonw.exe``
    if it sits alongside the active ``python.exe`` to avoid a transient
    console window.

    Args:
        python_exe: Optional python executable. Defaults to ``sys.executable``.
        prefer_windowed: Whether to prefer ``pythonw.exe`` on Windows.

    Returns:
        Path to the selected executable.
    """
    if python_exe is None:
        python_exe = Path(sys.executable)

    if prefer_windowed and sys.platform == "win32":
        pythonw = python_exe.with_name("pythonw.exe")
        if pythonw.is_file():
            return pythonw

    return python_exe


def verify_favicon_consistency(repo_root: Path | None = None) -> bool:
    """Verify that desktop assets favicon and web UI favicon are byte-identical.

    Args:
        repo_root: Root of the repository. Defaults to parent of ``src/``.

    Returns:
        True if both favicons exist and have identical content, False otherwise.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    asset_favicon = repo_root / "src" / "launchers" / "assets" / "favicon.ico"
    web_favicon = repo_root / "ui" / "public" / "favicon.ico"

    if not asset_favicon.is_file() or not web_favicon.is_file():
        return False

    return asset_favicon.read_bytes() == web_favicon.read_bytes()


def sync_web_favicon(repo_root: Path | None = None) -> bool:
    """Ensure the web UI public favicon matches the desktop assets favicon.

    Args:
        repo_root: Root of the repository. Defaults to parent of ``src/``.

    Returns:
        True if the web favicon is synchronized and byte-identical, False otherwise.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    asset_favicon = repo_root / "src" / "launchers" / "assets" / "favicon.ico"
    web_public_dir = repo_root / "ui" / "public"
    web_favicon = web_public_dir / "favicon.ico"

    if not asset_favicon.is_file():
        _logger.warning("Source asset favicon not found at %s", asset_favicon)
        return False

    web_public_dir.mkdir(parents=True, exist_ok=True)
    web_favicon.write_bytes(asset_favicon.read_bytes())
    _logger.info("Synchronized web favicon to %s", web_favicon)
    return True
