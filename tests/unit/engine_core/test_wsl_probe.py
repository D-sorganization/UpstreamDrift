"""Unit tests for WSL host boundary engine probing (MV-03 #10479).

Validates:
1. Probing WSL runtime availability on Windows host.
2. Detecting native SDK availability inside WSL environment (e.g. Pinocchio).
3. Graceful handling and accurate diagnostic reporting when WSL is unavailable or unconfigured.
4. Prevention of Windows-only probes falsely reporting native engines as completely absent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_core.wsl_probe import (
    WslEngineReport,
    get_wsl_engine_status,
    is_wsl_available,
    probe_wsl_engine,
)

pytestmark = pytest.mark.unit


def test_wsl_probe_when_wsl_cli_missing() -> None:
    with patch("shutil.which", return_value=None):
        assert is_wsl_available() is False
        report = probe_wsl_engine("pinocchio")
        assert isinstance(report, WslEngineReport)
        assert report.available_in_wsl is False
        assert "not available" in report.diagnostic_message.lower()


def test_wsl_probe_when_wsl_cli_present_and_engine_installed() -> None:
    with (
        patch("shutil.which", return_value="C:\\Windows\\System32\\wsl.exe"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="4.1.0\n",
            stderr="",
        )
        report = probe_wsl_engine("pinocchio", distro="Ubuntu-24.04")
        assert report.available_in_wsl is True
        assert report.version == "4.1.0"
        assert "4.1.0" in report.diagnostic_message


def test_wsl_probe_when_engine_not_in_wsl() -> None:
    with (
        patch("shutil.which", return_value="C:\\Windows\\System32\\wsl.exe"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="No module named 'pinocchio'\n",
        )
        report = probe_wsl_engine("pinocchio", distro="Ubuntu-24.04")
        assert report.available_in_wsl is False
        assert report.version is None
        assert "not installed in wsl" in report.diagnostic_message.lower()


def test_get_wsl_engine_status_caching() -> None:
    with (
        patch("shutil.which", return_value="C:\\Windows\\System32\\wsl.exe"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="4.1.0\n",
            stderr="",
        )
        rep1 = get_wsl_engine_status("pinocchio", distro="Ubuntu-24.04")
        rep2 = get_wsl_engine_status("pinocchio", distro="Ubuntu-24.04")
        assert rep1 == rep2
        # Verify cached call didn't invoke subprocess twice
        assert mock_run.call_count == 1
