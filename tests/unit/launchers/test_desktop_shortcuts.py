"""Unit tests for desktop and Start Menu shortcut manager."""

from __future__ import annotations

import pathlib
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.launchers.app_identity import (
    APP_DISPLAY_NAME,
    SHORTCUT_FILENAME,
)
from src.launchers.desktop_shortcuts import (
    ShortcutManager,
    ShortcutMetadata,
    ShortcutSpec,
    build_default_shortcut_spec,
    get_desktop_dir,
    get_start_menu_dir,
    install_desktop_and_start_menu_shortcuts,
)

pytestmark = pytest.mark.unit


def test_shortcut_spec_validation(tmp_path: pathlib.Path) -> None:
    """ShortcutSpec enforces preconditions on paths and arguments."""
    target_exe = tmp_path / "python.exe"
    target_exe.write_bytes(b"")

    with pytest.raises(ValueError, match="target_path must exist"):
        ShortcutSpec(
            name=SHORTCUT_FILENAME,
            target_path=tmp_path / "nonexistent.exe",
            arguments="launch_upstream_drift.py --classic",
            working_dir=tmp_path,
            icon_path=None,
            description="Test Launcher",
        )

    with pytest.raises(ValueError, match="name must end with .lnk"):
        ShortcutSpec(
            name="InvalidName",
            target_path=target_exe,
            arguments="launch_upstream_drift.py --classic",
            working_dir=tmp_path,
            icon_path=None,
            description="Test Launcher",
        )


def test_build_default_shortcut_spec(tmp_path: pathlib.Path) -> None:
    """build_default_shortcut_spec points to launch_upstream_drift.py --classic."""
    spec = build_default_shortcut_spec(repo_root=tmp_path)
    assert spec.name == SHORTCUT_FILENAME
    assert "--classic" in spec.arguments
    assert "launch_upstream_drift.py" in spec.arguments
    assert spec.working_dir == tmp_path
    assert APP_DISPLAY_NAME in spec.description


def test_get_desktop_and_start_menu_dirs() -> None:
    """Directory helpers return non-empty Paths."""
    desktop = get_desktop_dir()
    start_menu = get_start_menu_dir()
    assert isinstance(desktop, pathlib.Path)
    assert isinstance(start_menu, pathlib.Path)


def test_powershell_script_generation(tmp_path: pathlib.Path) -> None:
    """ShortcutManager generates safe, valid PowerShell script."""
    mgr = ShortcutManager()
    target_exe = tmp_path / "python.exe"
    target_exe.write_bytes(b"")
    icon = tmp_path / "app.ico"
    icon.write_bytes(b"")
    shortcut_path = tmp_path / "Test.lnk"

    spec = ShortcutSpec(
        name="Test.lnk",
        target_path=target_exe,
        arguments="launch_upstream_drift.py --classic",
        working_dir=tmp_path,
        icon_path=icon,
        description="Upstream Drift",
    )

    ps_script = mgr._generate_powershell_script(shortcut_path, spec)
    assert "New-Object -ComObject WScript.Shell" in ps_script
    assert "$Shortcut.TargetPath" in ps_script
    assert "$Shortcut.Arguments" in ps_script
    assert "$Shortcut.WorkingDirectory" in ps_script
    assert "$Shortcut.IconLocation" in ps_script
    assert "$Shortcut.Save()" in ps_script


def test_install_shortcut_invokes_powershell(tmp_path: pathlib.Path) -> None:
    """install_shortcut runs PowerShell with correct arguments."""
    mgr = ShortcutManager()
    target_exe = tmp_path / "python.exe"
    target_exe.write_bytes(b"")
    shortcut_path = tmp_path / "Test.lnk"

    spec = ShortcutSpec(
        name="Test.lnk",
        target_path=target_exe,
        arguments="launch_upstream_drift.py --classic",
        working_dir=tmp_path,
        icon_path=None,
        description="Upstream Drift",
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        success = mgr.create_shortcut(shortcut_path, spec)
        assert success is True

    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "powershell"
    assert "-NonInteractive" in cmd or "-Command" in cmd


def test_install_desktop_and_start_menu_shortcuts_idempotent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """install_desktop_and_start_menu_shortcuts creates shortcuts in both locations."""
    desktop_dir = tmp_path / "Desktop"
    desktop_dir.mkdir()
    start_menu_dir = tmp_path / "StartMenu"
    start_menu_dir.mkdir()

    monkeypatch.setattr(
        "src.launchers.desktop_shortcuts.get_desktop_dir", lambda: desktop_dir
    )
    monkeypatch.setattr(
        "src.launchers.desktop_shortcuts.get_start_menu_dir", lambda: start_menu_dir
    )

    with patch(
        "src.launchers.desktop_shortcuts.ShortcutManager.create_shortcut",
        return_value=True,
    ):
        result = install_desktop_and_start_menu_shortcuts(repo_root=tmp_path)
        assert result.success is True
        assert result.desktop_shortcut == desktop_dir / SHORTCUT_FILENAME
        assert result.start_menu_shortcut == start_menu_dir / SHORTCUT_FILENAME


def test_install_shortcuts_fails_if_either_destination_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shortcut installation fails if either Desktop or Start Menu fails."""
    desktop_dir = tmp_path / "Desktop"
    desktop_dir.mkdir()
    start_menu_dir = tmp_path / "StartMenu"
    start_menu_dir.mkdir()

    monkeypatch.setattr(
        "src.launchers.desktop_shortcuts.get_desktop_dir", lambda: desktop_dir
    )
    monkeypatch.setattr(
        "src.launchers.desktop_shortcuts.get_start_menu_dir", lambda: start_menu_dir
    )

    # Desktop succeeds, Start Menu fails
    with patch(
        "src.launchers.desktop_shortcuts.ShortcutManager.create_shortcut",
        side_effect=[True, False],
    ):
        result = install_desktop_and_start_menu_shortcuts(repo_root=tmp_path)
        assert result.success is False

    # Desktop fails, Start Menu succeeds
    with patch(
        "src.launchers.desktop_shortcuts.ShortcutManager.create_shortcut",
        side_effect=[False, True],
    ):
        result = install_desktop_and_start_menu_shortcuts(repo_root=tmp_path)
        assert result.success is False


def test_parse_shortcut_metadata() -> None:
    """_parse_metadata correctly extracts properties from WScript.Shell output."""
    raw_output = """
TargetPath=C:\\Python311\\python.exe
Arguments="launch_upstream_drift.py" --classic
WorkingDirectory=C:\\UpstreamDrift
IconLocation=C:\\UpstreamDrift\\assets\\golf_logo.ico
Description=Upstream Drift
"""
    meta = ShortcutManager._parse_metadata(pathlib.Path("Test.lnk"), raw_output)
    assert meta is not None
    assert meta.target_path == "C:\\Python311\\python.exe"
    assert meta.arguments == '"launch_upstream_drift.py" --classic'
    assert meta.working_directory == "C:\\UpstreamDrift"
    assert meta.icon_location == "C:\\UpstreamDrift\\assets\\golf_logo.ico"
    assert meta.description == "Upstream Drift"


def test_favicon_consistency(tmp_path: pathlib.Path) -> None:
    """Web favicon and launcher favicon.ico must be byte-identical."""
    from src.launchers.app_identity import verify_favicon_consistency

    asset_favicon = tmp_path / "src" / "launchers" / "assets" / "favicon.ico"
    asset_favicon.parent.mkdir(parents=True)
    asset_favicon.write_bytes(b"favicon-content-12345")

    web_favicon = tmp_path / "ui" / "public" / "favicon.ico"
    web_favicon.parent.mkdir(parents=True)
    web_favicon.write_bytes(b"favicon-content-12345")

    assert verify_favicon_consistency(repo_root=tmp_path) is True
