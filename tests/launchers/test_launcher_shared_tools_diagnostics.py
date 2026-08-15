"""Contract tests for the extracted shared-Tools freshness probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.launchers.launcher_shared_tools_diagnostics import (
    inspect_shared_tools_freshness,
)

pytestmark = pytest.mark.unit


def test_probe_reports_missing_submodule_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    """The extracted probe keeps the launcher's historical warning contract."""
    commands: list[tuple[tuple[str, ...], Path | None]] = []

    def run_git_cmd(cmd: list[str], cwd: Path | None = None) -> str:
        commands.append((tuple(cmd), cwd))
        if cmd[1:3] == ["ls-files", "--stage"]:
            return "160000 1234567890abcdef1234567890abcdef12345678 0\tvendor/ud-tools"
        return ""

    result = inspect_shared_tools_freshness(
        repo_root=tmp_path,
        run_git_cmd=run_git_cmd,
        find_sibling_root=lambda: None,
    )

    assert result.status == "warning"
    assert result.details["submodule_status"] == "not_initialized"
    assert result.details["is_current"] is False
    assert "not initialized" in result.message
    assert commands == [(("git", "ls-files", "--stage", "vendor/ud-tools"), None)]


def test_probe_reports_synchronized_pin_and_checkout(tmp_path: Path) -> None:
    """A matching initialized submodule remains a passing diagnostic."""
    submodule_dir = tmp_path / "vendor" / "ud-tools"
    submodule_dir.mkdir(parents=True)
    pinned_sha = "1" * 40

    def run_git_cmd(cmd: list[str], cwd: Path | None = None) -> str:
        if cmd[1:3] == ["ls-files", "--stage"]:
            return f"160000 {pinned_sha} 0\tvendor/ud-tools"
        if cmd[1:3] == ["rev-parse", "HEAD"] and cwd == submodule_dir:
            return pinned_sha
        if cmd[1:3] == ["rev-parse", "origin/main"]:
            return pinned_sha
        return ""

    result = inspect_shared_tools_freshness(
        repo_root=tmp_path,
        run_git_cmd=run_git_cmd,
        find_sibling_root=lambda: None,
    )

    assert result.status == "pass"
    assert result.details["submodule_status"] == "synchronized"
    assert result.details["sibling_status"] == "not_found"
    assert result.details["is_current"] is True
    assert result.message == "Shared folders and submodules are fully up to date."
