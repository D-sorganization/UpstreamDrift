"""Regression tests for `git` on the subprocess executable allowlist (#9230).

`git_sync_repository()` shells out to `git fetch --all` / `git pull`, but
`git` was absent from ``ALLOWED_EXECUTABLES``. ``SecureSubprocessError`` is a
plain ``Exception`` subclass, so neither ``run_command``'s
``@log_errors(..., reraise=False)`` (which catches only ``RuntimeError``,
``TypeError`` and ``ValueError``) nor ``git_sync_repository``'s own
``except (RuntimeError, ValueError, OSError)`` absorbed it: the helper raised
to its caller instead of logging a warning and returning ``False``.

These tests pin the chosen contract -- `git` is allowlisted, the allowlist is
still an allowlist, and `git_sync_repository` returns a bool -- without ever
touching the network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from src.shared.python import SUITE_ROOT
from src.shared.python.gui_pkg.launcher_utils import git_sync_repository
from src.shared.python.security.secure_subprocess import (
    ALLOWED_EXECUTABLES,
    SecureSubprocessError,
    secure_run,
    validate_executable,
)

pytestmark = pytest.mark.unit


class TestGitOnAllowlist:
    """`git` must be invocable through the secure subprocess wrappers."""

    @pytest.mark.parametrize("name", ["git", "git.exe"])
    def test_git_names_are_listed(self, name: str) -> None:
        assert name in ALLOWED_EXECUTABLES

    @pytest.mark.parametrize("name", ["git", "git.exe", "GIT", "/usr/bin/git"])
    def test_validate_executable_accepts_git(self, name: str) -> None:
        assert validate_executable(name) == name

    def test_secure_run_accepts_git(self) -> None:
        """`secure_run` must reach the subprocess layer rather than reject git."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["git", "--version"], returncode=0, stdout="git version 2.0.0"
            )
            result = secure_run(["git", "--version"], suite_root=SUITE_ROOT)

        assert result.returncode == 0
        assert mock_run.call_args.args[0][0] == "git"


class TestAllowlistStillBlocks:
    """Adding `git` must not turn the allowlist into a pass-through."""

    @pytest.mark.parametrize("name", ["rm", "curl", "/bin/sh", "git-upload-pack"])
    def test_disallowed_executables_still_raise(self, name: str) -> None:
        with pytest.raises(SecureSubprocessError):
            validate_executable(name)

    def test_secure_run_still_rejects_disallowed(self) -> None:
        with pytest.raises(SecureSubprocessError):
            secure_run(["rm", "-rf", "/"], suite_root=SUITE_ROOT)


class TestGitSyncRepositoryContract:
    """`git_sync_repository` reports success as a bool, it does not raise."""

    def _completed(self, returncode: int) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["git"], returncode=returncode)

    def test_returns_true_when_pull_succeeds(self) -> None:
        with patch("src.shared.python.gui_pkg.launcher_utils.run_command") as mock_cmd:
            mock_cmd.return_value = self._completed(0)
            assert git_sync_repository(SUITE_ROOT) is True
        # fetch + pull
        assert mock_cmd.call_count == 2
        assert mock_cmd.call_args_list[0].args[0][:2] == ["git", "fetch"]
        assert mock_cmd.call_args_list[1].args[0][:2] == ["git", "pull"]

    def test_returns_false_when_pull_fails(self) -> None:
        with patch("src.shared.python.gui_pkg.launcher_utils.run_command") as mock_cmd:
            mock_cmd.return_value = self._completed(1)
            assert git_sync_repository(SUITE_ROOT) is False

    def test_returns_false_when_command_reports_none(self) -> None:
        with patch("src.shared.python.gui_pkg.launcher_utils.run_command") as mock_cmd:
            mock_cmd.return_value = None
            assert git_sync_repository(SUITE_ROOT) is False

    @pytest.mark.parametrize("exc", [OSError("offline"), RuntimeError("boom")])
    def test_ordinary_failures_are_swallowed(self, exc: Exception) -> None:
        with patch(
            "src.shared.python.gui_pkg.launcher_utils.run_command", side_effect=exc
        ):
            assert git_sync_repository(SUITE_ROOT) is False

    def test_real_git_invocation_in_throwaway_repo(self, tmp_path: Path) -> None:
        """End-to-end through the real allowlist, no network.

        `git pull` in a repo with no remote fails with a non-zero exit code --
        an ordinary failure -- so the contract is `False`, not an exception.
        """
        repo = tmp_path / "throwaway"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

        # `run_command` validates cwd against the suite root, so point the
        # secure layer at the throwaway repo's own parent.
        with patch(
            "src.shared.python.security.subprocess_utils._default_suite_root",
            return_value=tmp_path,
        ):
            assert git_sync_repository(repo) is False
