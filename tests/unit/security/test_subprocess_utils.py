"""Tests for src.shared.python.security.subprocess_utils (Issues #1949, #1744)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python import SUITE_ROOT
from src.shared.python.security.subprocess_utils import (
    CommandRunner,
    ProcessManager,
    _default_suite_root,
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


# ---------------------------------------------------------------------------
# suite_root plumbing (issue #9221)
#
# PR #9216 enforced the executable-name allowlist on the two secure_popen call
# sites in this module, but neither passed ``suite_root`` -- and
# ``secure_popen`` skips script-path and cwd validation entirely when it is
# falsy. Directory-traversal protection was therefore inert on the background
# launch paths. These tests pin that it now engages, and that a rejection
# stays inside the existing "log and return falsy" contract rather than
# escaping as an exception.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuiteRootIsPlumbedThrough:
    #: A real in-suite script under an ALLOWED_SCRIPT_DIRECTORIES entry.
    _IN_SUITE_SCRIPT = SUITE_ROOT / "src" / "shared" / "python" / "version_info.py"

    def test_default_suite_root_is_the_checkout_root(self) -> None:
        assert _default_suite_root() == SUITE_ROOT
        assert (SUITE_ROOT / "src" / "shared" / "python").is_dir()

    def test_default_suite_root_matches_project_root_detection(self) -> None:
        """Stay consistent with the .git-based root rule from #9224."""
        from src.shared.python.data_io._path_utils import find_project_root

        assert find_project_root() == _default_suite_root()

    def test_process_manager_rejects_out_of_tree_script(self, tmp_path) -> None:
        payload = tmp_path / "payload.py"
        payload.write_text("print('pwned')\n", encoding="utf-8")

        pm = ProcessManager()
        try:
            assert pm.start("evil", [sys.executable, str(payload)]) is False
            assert "evil" not in pm.processes
        finally:
            pm.stop_all()

    def test_process_manager_rejects_out_of_tree_cwd(self, tmp_path) -> None:
        pm = ProcessManager()
        try:
            started = pm.start(
                "evil-cwd",
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=tmp_path,
            )
            assert started is False
            assert "evil-cwd" not in pm.processes
        finally:
            pm.stop_all()

    def test_process_manager_allows_in_suite_script(self) -> None:
        pm = ProcessManager()
        try:
            assert pm.start("ok", [sys.executable, str(self._IN_SUITE_SCRIPT)]) is True
        finally:
            pm.stop_all()

    def test_process_manager_honours_explicit_suite_root(self, tmp_path) -> None:
        """An explicit root re-permits a launch the default root rejects."""
        (tmp_path / "src").mkdir()
        payload = tmp_path / "src" / "payload.py"
        payload.write_text("pass\n", encoding="utf-8")

        pm = ProcessManager()
        try:
            assert pm.start("scoped", [sys.executable, str(payload)]) is False
            assert (
                pm.start(
                    "scoped",
                    [sys.executable, str(payload)],
                    suite_root=tmp_path,
                )
                is True
            )
        finally:
            pm.stop_all()

    def test_command_runner_rejects_out_of_tree_script(self, tmp_path) -> None:
        payload = tmp_path / "payload.py"
        payload.write_text("print('pwned')\n", encoding="utf-8")

        runner = CommandRunner()
        assert runner.run_async([sys.executable, str(payload)]) is None

    def test_command_runner_rejects_out_of_tree_cwd(self, tmp_path) -> None:
        runner = CommandRunner(cwd=tmp_path)
        assert runner.run_async([sys.executable, "--version"]) is None

    def test_command_runner_allows_in_suite_script(self) -> None:
        runner = CommandRunner()
        proc = runner.run_async([sys.executable, str(self._IN_SUITE_SCRIPT)])
        assert proc is not None
        try:
            proc.wait(timeout=60)
        finally:
            proc.terminate()

    def test_command_runner_defaults_suite_root(self) -> None:
        assert CommandRunner().suite_root == _default_suite_root()
        assert CommandRunner(suite_root="/somewhere").suite_root == Path("/somewhere")
