"""Tests for shell_discovery module.

Verifies that discover_shells() correctly enumerates available OS shells
and that ShellDescriptor has the required fields.

Issue #5617: real OS terminal tab with PTY backend.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_shell_discovery():
    """Import shell_discovery, raising ImportError on missing deps."""
    from src.shared.python.upstream_drift_tools.ui.tools_sidebar.shell_discovery import (
        ShellDescriptor,
        discover_shells,
    )

    return discover_shells, ShellDescriptor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discover_shells_returns_list() -> None:
    """discover_shells() always returns a list of ShellDescriptors."""
    discover_shells, ShellDescriptor = _import_shell_discovery()

    with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
        shells = discover_shells()

    assert isinstance(shells, list)
    assert all(isinstance(s, ShellDescriptor) for s in shells)


@pytest.mark.unit
@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_discover_shells_on_windows_finds_pwsh_or_powershell() -> None:
    """On Windows, discover_shells() finds pwsh or powershell."""
    discover_shells, _ = _import_shell_discovery()

    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/Windows/System32/pwsh.exe" if x in ("pwsh", "powershell") else None
        ),
    ):
        shells = discover_shells()

    names = [s.display_name for s in shells]
    assert any("PowerShell" in n or "pwsh" in n for n in names)


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="Linux/macOS only")
def test_discover_shells_on_posix_finds_bash() -> None:
    """On POSIX, discover_shells() finds bash when it is present."""
    discover_shells, _ = _import_shell_discovery()

    with patch(
        "shutil.which",
        side_effect=lambda x: f"/bin/{x}" if x == "bash" else None,
    ):
        shells = discover_shells()

    names = [s.display_name for s in shells]
    assert "bash" in names or "Bash" in names


@pytest.mark.unit
def test_discover_shells_returns_empty_when_no_shells_found() -> None:
    """discover_shells() returns an empty list when no shells are on PATH."""
    discover_shells, _ = _import_shell_discovery()

    with patch("shutil.which", return_value=None):
        shells = discover_shells()

    assert shells == []


@pytest.mark.unit
def test_shell_descriptor_has_required_fields() -> None:
    """ShellDescriptor exposes display_name, binary, and args."""
    _, ShellDescriptor = _import_shell_discovery()

    sd = ShellDescriptor(display_name="bash", binary="/bin/bash", args=[])

    assert sd.display_name == "bash"
    assert sd.binary == "/bin/bash"
    assert isinstance(sd.args, list)


@pytest.mark.unit
def test_discover_shells_deduplicates_same_binary() -> None:
    """discover_shells() does not return the same binary twice."""
    discover_shells, _ = _import_shell_discovery()

    # Simulate sh and bash pointing to same binary
    with patch("shutil.which", side_effect=lambda x: "/bin/sh"):
        shells = discover_shells()

    binaries = [s.binary for s in shells]
    assert len(binaries) == len(set(binaries)), "Duplicate binaries in shell list"


@pytest.mark.unit
def test_shell_descriptor_args_defaults_to_empty_list() -> None:
    """ShellDescriptor.args defaults to an empty list when not provided."""
    _, ShellDescriptor = _import_shell_discovery()

    sd = ShellDescriptor(display_name="zsh", binary="/bin/zsh")

    assert sd.args == []
