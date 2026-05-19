"""Tests for src.shared.python.security.subprocess_utils (Issues #1949, #1744)."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.security.subprocess_utils import (
    CommandRunner,
    ProcessManager,
    run_command,
)

# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_python_version_returns_completed_process(self) -> None:
        result = run_command([sys.executable, "--version"])
        assert result is not None
        assert isinstance(result, subprocess.CompletedProcess)

    def test_python_version_returncode_zero(self) -> None:
        result = run_command([sys.executable, "--version"])
        assert result is not None
        assert result.returncode == 0

    def test_python_eval_captures_output(self) -> None:
        result = run_command([sys.executable, "-c", "print('hello')"])
        assert result is not None
        assert result.returncode == 0

    def test_python_exit_1_nonzero(self) -> None:
        result = run_command([sys.executable, "-c", "import sys; sys.exit(1)"])
        assert result is not None
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# ProcessManager
# ---------------------------------------------------------------------------


class TestProcessManagerEmpty:
    def test_initial_processes_empty(self) -> None:
        pm = ProcessManager()
        assert pm.processes == {}

    def test_list_processes_empty(self) -> None:
        pm = ProcessManager()
        assert pm.list_processes() == {}

    def test_is_running_unknown_returns_false(self) -> None:
        pm = ProcessManager()
        assert pm.is_running("nonexistent") is False

    def test_stop_unknown_returns_false(self) -> None:
        pm = ProcessManager()
        assert pm.stop("nonexistent") is False

    def test_get_output_unknown_returns_empty_strings(self) -> None:
        pm = ProcessManager()
        out, err = pm.get_output("nonexistent")
        assert out == ""
        assert err == ""


class TestProcessManagerStart:
    # Use python3 as the only allowed long-running executable
    _LONG_CMD = [sys.executable, "-c", "import time; time.sleep(60)"]

    def test_start_valid_command_returns_true(self) -> None:
        pm = ProcessManager()
        started = pm.start("py", self._LONG_CMD)
        try:
            assert started is True
        finally:
            pm.stop("py")

    def test_started_process_is_running(self) -> None:
        pm = ProcessManager()
        pm.start("py", self._LONG_CMD)
        try:
            assert pm.is_running("py") is True
        finally:
            pm.stop("py")

    def test_start_same_name_twice_returns_false(self) -> None:
        pm = ProcessManager()
        pm.start("py", self._LONG_CMD)
        try:
            second = pm.start("py", self._LONG_CMD)
            assert second is False
        finally:
            pm.stop("py")

    def test_start_invalid_command_returns_false(self) -> None:
        pm = ProcessManager()
        result = pm.start("bad", ["__this_does_not_exist_xyz__"])
        assert result is False

    @patch("src.shared.python.security.subprocess_utils.subprocess.Popen")
    def test_start_hides_windows_by_default_on_windows(self, mock_popen) -> None:
        mock_popen.return_value = MagicMock()
        pm = ProcessManager()

        with patch("src.shared.python.security.subprocess_utils.os.name", "nt"):
            assert pm.start("probe", self._LONG_CMD) is True

        assert mock_popen.call_args.kwargs["creationflags"] == 0x08000000

    def test_stop_returns_true(self) -> None:
        pm = ProcessManager()
        pm.start("py", self._LONG_CMD)
        result = pm.stop("py")
        assert result is True

    def test_stop_all_clears_processes(self) -> None:
        pm = ProcessManager()
        pm.start("p1", self._LONG_CMD)
        pm.start("p2", self._LONG_CMD)
        pm.stop_all()
        assert len(pm.processes) == 0

    def test_list_processes_after_start(self) -> None:
        pm = ProcessManager()
        pm.start("py", self._LONG_CMD)
        try:
            listing = pm.list_processes()
            assert "py" in listing
            assert listing["py"] is True
        finally:
            pm.stop("py")


# ---------------------------------------------------------------------------
# CommandRunner
# ---------------------------------------------------------------------------


class TestCommandRunner:
    def test_run_returns_completed_process(self) -> None:
        runner = CommandRunner()
        result = runner.run([sys.executable, "--version"])
        assert result is not None
        assert result.returncode == 0

    def test_run_check_raises_on_nonzero(self) -> None:
        runner = CommandRunner()
        with pytest.raises(subprocess.CalledProcessError):
            runner.run([sys.executable, "-c", "import sys; sys.exit(2)"], check=True)

    def test_run_async_returns_popen(self) -> None:
        runner = CommandRunner()
        proc = runner.run_async([sys.executable, "-c", "import time; time.sleep(60)"])
        assert proc is not None
        assert proc.poll() is None
        proc.terminate()
        proc.wait()

    def test_run_async_bad_command_returns_none(self) -> None:
        runner = CommandRunner()
        result = runner.run_async(["__does_not_exist_xyz__"])
        assert result is None

    @patch("src.shared.python.security.subprocess_utils.subprocess.Popen")
    def test_run_async_hides_windows_by_default_on_windows(self, mock_popen) -> None:
        mock_popen.return_value = MagicMock()
        runner = CommandRunner()

        with patch("src.shared.python.security.subprocess_utils.os.name", "nt"):
            assert runner.run_async([sys.executable, "--version"]) is not None

        assert mock_popen.call_args.kwargs["creationflags"] == 0x08000000

    def test_cwd_set_on_runner(self) -> None:
        runner = CommandRunner(cwd="/tmp")
        assert str(runner.cwd) == "/tmp"
