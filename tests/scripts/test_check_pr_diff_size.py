"""Tests for PR diff size gate (issue #14741)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.ci.check_pr_diff_size import (
    diff_line_total,
    exempt_deletion_only,
    parse_numstat,
)

pytestmark = pytest.mark.unit


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-b", "main")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "-m", "base")
    return repo


def test_parse_numstat_skips_binary() -> None:
    assert parse_numstat("-	-	bin.dat") is None


def test_parse_numstat_parses_text() -> None:
    assert parse_numstat("3	7	path/foo.txt") == (3, 7, "path/foo.txt")


def test_exempt_deletion_only() -> None:
    assert exempt_deletion_only(0, 10) is True
    assert exempt_deletion_only(1, 10) is False
    assert exempt_deletion_only(0, 0) is False


def test_diff_line_total_counts_added_and_deleted(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    main = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run_git(repo, "checkout", "-b", "feature")
    (repo / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    _run_git(repo, "add", "a.txt")
    _run_git(repo, "commit", "-m", "add file")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    total, total_added, total_deleted = diff_line_total(repo, main, head)
    assert total == 2
    assert total_added == 2
    assert total_deleted == 0


def test_golden_regen_label_exempt_via_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.ci import check_pr_diff_size as mod

    repo = _init_repo(tmp_path)
    main = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run_git(repo, "checkout", "-b", "big")
    big = "\n".join([f"line {i}" for i in range(600)])
    (repo / "huge.txt").write_text(big + "\n", encoding="utf-8")
    _run_git(repo, "add", "huge.txt")
    _run_git(repo, "commit", "-m", "big")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    event = {"pull_request": {"labels": [{"name": "golden-regen"}]}}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    err = mod.check_pr_diff_size(
        repo_root=repo, base_sha=main, head_sha=head, max_lines=500
    )
    assert err is None
