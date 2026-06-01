"""Tests for Windows quoting and deduplication in ProcessManager (#6921, #6922)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.launchers.launcher_process_manager import ProcessManager, _quote_cmd_arg


def test_quote_cmd_arg_windows() -> None:
    """_quote_cmd_arg must quote spaces on Windows and escape double quotes."""
    with patch("os.name", "nt"):
        assert _quote_cmd_arg("python") == '"python"'
        assert (
            _quote_cmd_arg("C:\\Program Files\\python.exe")
            == '"C:\\Program Files\\python.exe"'
        )
        assert _quote_cmd_arg('my "quoted" argument') == '"my \\"quoted\\" argument"'


def test_quote_cmd_arg_posix() -> None:
    """_quote_cmd_arg must use shlex.quote on POSIX."""
    with patch("os.name", "posix"):
        assert _quote_cmd_arg("python") == "python"
        assert _quote_cmd_arg("my argument") == "'my argument'"


def test_launch_script_separate_terminals_quoting(tmp_path: Path) -> None:
    """ProcessManager must use custom Windows quoting in launch_script when use_separate_terminals is active on Windows."""
    pm = ProcessManager(repo_root=tmp_path)
    pm.use_separate_terminals = True

    script = tmp_path / "my script.py"
    script.write_text("print('hello')", encoding="utf-8")

    with (
        patch("os.name", "nt"),
        patch("src.launchers.launcher_process_manager.CREATE_NEW_CONSOLE", 0x10),
        patch("src.launchers.launcher_process_manager.validate_script_path"),
        patch("src.launchers.launcher_process_manager._assign_to_job"),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()

        pm.launch_script(
            name="test_engine",
            script_path=script,
            cwd=tmp_path,
        )

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd_str = args[0]
        # Verify the command is constructed using double quotes suitable for Windows cmd.exe
        assert cmd_str.startswith('cmd /c "')
        # The interpreter is present and the space-containing script path is
        # wrapped in double quotes (cmd.exe quoting via subprocess.list2cmdline).
        assert sys.executable in cmd_str
        assert f'"{script}"' in cmd_str


def test_launch_module_separate_terminals_quoting(tmp_path: Path) -> None:
    """ProcessManager must use custom Windows quoting in launch_module when use_separate_terminals is active on Windows."""
    pm = ProcessManager(repo_root=tmp_path)
    pm.use_separate_terminals = True

    with (
        patch("os.name", "nt"),
        patch("src.launchers.launcher_process_manager.CREATE_NEW_CONSOLE", 0x10),
        patch("src.launchers.launcher_process_manager._assign_to_job"),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()

        pm.launch_module(
            name="test_engine",
            module_name="my_pkg.my_module",
            cwd=tmp_path,
        )

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd_str = args[0]
        assert cmd_str.startswith('cmd /c "')
        assert sys.executable in cmd_str
        assert " -m my_pkg.my_module" in cmd_str
