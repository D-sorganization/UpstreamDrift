"""
Tests for launcher_constants.py CONFIG_DIR migration away from .kiro/.

Fixes issue #5713: .kiro/ is a defunct IDE folder that confuses new contributors.

DbC postconditions:
- CONFIG_DIR must not use .kiro/ (defunct IDE folder)
- CONFIG_DIR must end with "launcher" for compatibility with existing tests
- CONFIG_DIR must be under user home or AppData
- LAYOUT_CONFIG_FILE must be under CONFIG_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path


def test_config_dir_not_in_kiro() -> None:
    """DbC postcondition: CONFIG_DIR must not use .kiro/ (defunct IDE folder)."""
    from src.launchers.launcher_constants import CONFIG_DIR

    assert ".kiro" not in str(CONFIG_DIR), (
        f"CONFIG_DIR should not be in .kiro/: {CONFIG_DIR}\n"
        "Fix: migrate to platformdirs user_config_dir or ~/.config/upstream-drift/"
    )


def test_config_dir_ends_with_launcher() -> None:
    """CONFIG_DIR must end with 'launcher' for compatibility with existing code."""
    from src.launchers.launcher_constants import CONFIG_DIR

    assert str(CONFIG_DIR).endswith("launcher"), (
        f"CONFIG_DIR should end with 'launcher', got: {CONFIG_DIR}"
    )


def test_config_dir_is_platform_appropriate() -> None:
    """DbC postcondition: CONFIG_DIR should be under user home or AppData."""
    from src.launchers.launcher_constants import CONFIG_DIR

    config_str = str(CONFIG_DIR)
    home_str = str(Path.home())

    is_under_home = config_str.startswith(home_str)
    is_under_appdata = "AppData" in config_str

    assert is_under_home or is_under_appdata, (
        f"CONFIG_DIR should be under user home ({home_str}) or AppData, "
        f"but got: {config_str}"
    )


def test_layout_config_file_under_config_dir() -> None:
    """DbC postcondition: LAYOUT_CONFIG_FILE must be under CONFIG_DIR."""
    from src.launchers.launcher_constants import CONFIG_DIR, LAYOUT_CONFIG_FILE

    assert str(LAYOUT_CONFIG_FILE).startswith(str(CONFIG_DIR)), (
        f"LAYOUT_CONFIG_FILE ({LAYOUT_CONFIG_FILE}) must be under CONFIG_DIR ({CONFIG_DIR})"
    )


def test_config_dir_is_path_instance() -> None:
    """CONFIG_DIR must be a pathlib.Path instance."""
    from src.launchers.launcher_constants import CONFIG_DIR

    assert isinstance(CONFIG_DIR, Path), (
        f"CONFIG_DIR must be Path, got {type(CONFIG_DIR)}"
    )


def test_layout_config_file_is_path_instance() -> None:
    """LAYOUT_CONFIG_FILE must be a pathlib.Path instance."""
    from src.launchers.launcher_constants import LAYOUT_CONFIG_FILE

    assert isinstance(LAYOUT_CONFIG_FILE, Path), (
        f"LAYOUT_CONFIG_FILE must be Path, got {type(LAYOUT_CONFIG_FILE)}"
    )


def test_config_dir_uses_app_name_upstream_drift() -> None:
    """CONFIG_DIR path should identify the upstream-drift application."""
    from src.launchers.launcher_constants import CONFIG_DIR

    config_str = str(CONFIG_DIR).lower().replace("-", "").replace("_", "")
    assert "upstreamdrift" in config_str, (
        f"CONFIG_DIR path should contain 'upstream-drift' or 'upstreamdrift', "
        f"got: {CONFIG_DIR}"
    )


def test_repos_root_not_used_for_config_dir() -> None:
    """Config state must not be stored under the repository root (REPOS_ROOT)."""
    from src.launchers.launcher_constants import REPOS_ROOT, CONFIG_DIR

    assert not str(CONFIG_DIR).startswith(str(REPOS_ROOT)), (
        f"CONFIG_DIR ({CONFIG_DIR}) must not be under REPOS_ROOT ({REPOS_ROOT}). "
        "Runtime state must not be stored in the repository tree."
    )
