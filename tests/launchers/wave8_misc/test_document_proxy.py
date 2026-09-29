"""Tests for src.launchers.document_proxy CLI entry point.

Covers the argparse / dispatch / error paths of the documentation
proxy script. We patch ``platform.system`` and the relevant launchers
(``os.startfile`` on Windows, ``subprocess.Popen`` elsewhere) so the
tests stay hermetic and never actually open a document.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import document_proxy


def _invoke(args: list[str]) -> None:
    """Run document_proxy.main() with the given argv tail."""
    with patch.object(sys, "argv", ["document_proxy", *args]):
        document_proxy.main()


def test_missing_file_exits_with_code_1(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.md"
    with pytest.raises(SystemExit) as exc_info:
        _invoke([str(missing)])
    assert exc_info.value.code == 1


def test_windows_uses_startfile(tmp_path) -> None:
    target = tmp_path / "report.pdf"
    target.write_text("dummy")

    fake_startfile = MagicMock()
    with (
        patch.object(document_proxy.platform, "system", return_value="Windows"),
        patch.object(document_proxy.os, "startfile", fake_startfile, create=True),
    ):
        _invoke([str(target)])
    assert fake_startfile.call_count == 1
    called_path = fake_startfile.call_args[0][0]
    assert str(target.resolve()) == called_path


def test_macos_uses_open(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("# hi")

    fake_popen = MagicMock()
    with (
        patch.object(document_proxy.platform, "system", return_value="Darwin"),
        patch.object(document_proxy.subprocess, "Popen", fake_popen),
    ):
        _invoke([str(target)])
    fake_popen.assert_called_once()
    cmd = fake_popen.call_args[0][0]
    assert cmd[0] == "open"
    assert cmd[1] == str(target.resolve())


def test_linux_uses_xdg_open(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("# hi")

    fake_popen = MagicMock()
    with (
        patch.object(document_proxy.platform, "system", return_value="Linux"),
        patch.object(document_proxy.subprocess, "Popen", fake_popen),
    ):
        _invoke([str(target)])
    fake_popen.assert_called_once()
    cmd = fake_popen.call_args[0][0]
    assert cmd[0] == "xdg-open"


def test_launcher_failure_exits_with_code_1(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("# hi")

    def _boom(_cmd):
        raise OSError("simulated launch failure")

    with (
        patch.object(document_proxy.platform, "system", return_value="Linux"),
        patch.object(document_proxy.subprocess, "Popen", side_effect=_boom),
        pytest.raises(SystemExit) as exc_info,
    ):
        _invoke([str(target)])
    assert exc_info.value.code == 1


def test_argparse_requires_file_path() -> None:
    with patch.object(sys, "argv", ["document_proxy"]), pytest.raises(SystemExit):
        document_proxy.main()
