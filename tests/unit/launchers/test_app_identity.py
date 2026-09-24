"""Unit tests for UpstreamDrift shared app identity contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.launchers.app_identity import (
    APP_DISPLAY_NAME,
    APP_NAME,
    APP_USER_MODEL_ID,
    CANONICAL_ICON_CANDIDATES,
    LEGACY_SHORTCUT_FILENAME,
    SHORTCUT_FILENAME,
    AppIdentity,
    get_app_identity,
    resolve_canonical_icon,
    resolve_launcher_executable,
)

pytestmark = pytest.mark.unit


def test_app_user_model_id_constant() -> None:
    """Canonical Windows AppUserModelID must match the fleet-standard identity."""
    assert APP_USER_MODEL_ID == "D-sorganization.UpstreamDrift"
    assert APP_NAME == "UpstreamDrift"
    assert APP_DISPLAY_NAME == "Upstream Drift"
    assert SHORTCUT_FILENAME == "UpstreamDrift.lnk"
    assert LEGACY_SHORTCUT_FILENAME == "Golf Modeling Suite.lnk"


def test_app_identity_immutability() -> None:
    """AppIdentity must be an immutable frozen dataclass."""
    identity = get_app_identity()
    assert identity.app_id == APP_USER_MODEL_ID
    assert identity.app_name == APP_NAME
    assert identity.display_name == APP_DISPLAY_NAME
    assert identity.shortcut_name == SHORTCUT_FILENAME
    assert identity.legacy_shortcut_name == LEGACY_SHORTCUT_FILENAME

    with pytest.raises(FrozenInstanceError):
        identity.app_id = "Other.App"  # type: ignore[misc]


def test_app_identity_contract_validation() -> None:
    """AppIdentity must reject empty or invalid values (DbC)."""
    with pytest.raises(ValueError, match="app_id must be a non-empty string"):
        AppIdentity(
            app_id="",
            app_name=APP_NAME,
            display_name=APP_DISPLAY_NAME,
            shortcut_name=SHORTCUT_FILENAME,
            legacy_shortcut_name=LEGACY_SHORTCUT_FILENAME,
        )

    with pytest.raises(ValueError, match="shortcut_name must end with .lnk"):
        AppIdentity(
            app_id=APP_USER_MODEL_ID,
            app_name=APP_NAME,
            display_name=APP_DISPLAY_NAME,
            shortcut_name="UpstreamDrift",
            legacy_shortcut_name=LEGACY_SHORTCUT_FILENAME,
        )


def test_canonical_icon_candidates_order() -> None:
    """Icon candidates must prioritize high-resolution Windows ICO formats."""
    assert isinstance(CANONICAL_ICON_CANDIDATES, tuple)
    assert CANONICAL_ICON_CANDIDATES[0] == "golf_logo.ico"
    assert "golf_suite_unified.ico" in CANONICAL_ICON_CANDIDATES
    assert "golf_robot_windows_optimized.ico" in CANONICAL_ICON_CANDIDATES


def test_resolve_canonical_icon_finds_first_existing(tmp_path: Path) -> None:
    """resolve_canonical_icon returns the first candidate present on disk."""
    third_icon = tmp_path / "golf_robot_windows_optimized.ico"
    third_icon.write_bytes(b"ico-data-3")

    resolved = resolve_canonical_icon(assets_dir=tmp_path)
    assert resolved == third_icon


def test_resolve_canonical_icon_returns_none_if_missing(tmp_path: Path) -> None:
    """resolve_canonical_icon returns None if no candidate exists."""
    assert resolve_canonical_icon(assets_dir=tmp_path) is None


def test_resolve_launcher_executable_prefers_pythonw_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On Windows, resolve_launcher_executable prefers pythonw.exe when present."""
    fake_python = tmp_path / "python.exe"
    fake_python.write_bytes(b"")
    fake_pythonw = tmp_path / "pythonw.exe"
    fake_pythonw.write_bytes(b"")

    monkeypatch.setattr("sys.platform", "win32")
    exe = resolve_launcher_executable(python_exe=fake_python, prefer_windowed=True)
    assert exe == fake_pythonw
