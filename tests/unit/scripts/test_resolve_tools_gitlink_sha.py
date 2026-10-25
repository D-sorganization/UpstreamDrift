"""Tests for cross-host Tools gitlink resolution (issue #12940)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.packaging.resolve_tools_gitlink_sha import (
    git_executable_for_repo,
    resolve_tools_gitlink_sha,
)

pytestmark = pytest.mark.unit

_PINNED_SHA = "a" * 40


def test_git_executable_uses_git_exe_for_windows_worktree_pointer(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").write_text(
        "gitdir: C:/Users/example/Repositories/UpstreamDrift/.git/worktrees/wt\n",
        encoding="utf-8",
    )

    assert git_executable_for_repo(tmp_path) == "git.exe"


def test_git_executable_uses_unix_git_for_normal_checkout(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    assert git_executable_for_repo(tmp_path) == "git"


def test_resolve_tools_gitlink_sha_invokes_selected_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").write_text(
        "gitdir: D:/repos/UpstreamDrift/.git/worktrees/orch\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run(
        args: tuple[str, ...],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args[0], args))
        assert cwd == tmp_path.resolve()
        return subprocess.CompletedProcess(
            args, 0, stdout=f"{_PINNED_SHA}\n", stderr=""
        )

    monkeypatch.setattr(
        "scripts.packaging.resolve_tools_gitlink_sha.subprocess.run", fake_run
    )

    assert resolve_tools_gitlink_sha(tmp_path) == _PINNED_SHA
    assert calls and calls[0][0] == "git.exe"
    assert "rev-parse" in calls[0][1]
    assert "HEAD:vendor/ud-tools" in calls[0][1]


def test_resolve_tools_gitlink_sha_fails_closed_when_gitlink_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess((), 1, stdout="", stderr="fatal: not a tree")

    monkeypatch.setattr(
        "scripts.packaging.resolve_tools_gitlink_sha.subprocess.run", fake_run
    )

    with pytest.raises(RuntimeError, match="vendor/ud-tools gitlink"):
        resolve_tools_gitlink_sha(tmp_path)


def test_resolve_tools_gitlink_sha_matches_git_ls_tree_on_real_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    dot_git = repo_root / ".git"
    if not dot_git.exists():
        pytest.skip("not a git checkout")
    expected = subprocess.run(
        ["git", "ls-tree", "HEAD", "--", "vendor/ud-tools"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if expected.returncode != 0:
        pytest.skip("git ls-tree unavailable in this checkout")
    fields = expected.stdout.strip().split()
    if len(fields) < 3:
        pytest.skip("vendor/ud-tools gitlink not present")

    assert resolve_tools_gitlink_sha(repo_root) == fields[2].lower()
