"""Tests for ``scripts/ci/assert_checkout_complete.py`` (issue #9507).

Each test builds a throwaway git repository so the real ``git ls-files`` /
sparse-checkout behaviour is exercised, not a mock of it.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "ci" / "assert_checkout_complete.py"

_spec = importlib.util.spec_from_file_location("assert_checkout_complete", _SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pytestmark = pytest.mark.unit

main = _mod.main
find_missing = _mod.find_missing


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    for rel in (
        ".github/actions/x/action.yml",
        "ui/package-lock.json",
        "src/a.py",
        "docs/readme.md",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_complete_tree_passes(repo, monkeypatch, capsys):
    monkeypatch.setenv("RUNNER_NAME", "runner-1")
    assert main(["--root", str(repo)]) == 0
    assert "sparse-checkout: disabled" in capsys.readouterr().out


def test_deleted_tracked_file_fails_naming_file_and_runner(repo, monkeypatch, capsys):
    monkeypatch.setenv("RUNNER_NAME", "runner-7")
    (repo / "ui" / "package-lock.json").unlink()
    assert main(["--root", str(repo)]) == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "incomplete checkout" in text.lower()
    assert "ui/package-lock.json" in text
    assert "runner-7" in text


def test_missing_listing_is_capped(repo, monkeypatch, capsys):
    monkeypatch.setenv("RUNNER_NAME", "r")
    for i in range(30):
        f = repo / "src" / f"f{i}.py"
        f.write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "more")
    for i in range(30):
        (repo / "src" / f"f{i}.py").unlink()
    assert main(["--root", str(repo), "--max-report", "5"]) == 1
    text = capsys.readouterr().out
    assert "30 tracked file(s) missing" in text
    assert "and 25 more" in text


def test_sparse_checkout_outside_cone_passes(repo, monkeypatch, capsys):
    monkeypatch.setenv("RUNNER_NAME", "r")
    _git(repo, "sparse-checkout", "set", "--cone", "src")
    assert not (repo / "docs").exists()
    assert find_missing(repo) == []
    assert main(["--root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "sparse-checkout: enabled" in out
    assert "src" in out


def test_empty_sparse_pattern_set_that_pruned_files_fails(repo, monkeypatch, capsys):
    monkeypatch.setenv("RUNNER_NAME", "runner-9")
    _git(repo, "sparse-checkout", "set", "--no-cone", "--stdin")  # no patterns
    (repo / ".git" / "info" / "sparse-checkout").write_text("", encoding="utf-8")
    _git(repo, "read-tree", "-mu", "HEAD")
    assert not (repo / "ui" / "package-lock.json").exists()
    assert main(["--root", str(repo)]) == 1
    text = capsys.readouterr()
    combined = text.out + text.err
    assert "empty" in combined.lower()
    assert "runner-9" in combined


def test_sparse_status_is_logged_when_failing(repo, monkeypatch, capsys):
    monkeypatch.setenv("RUNNER_NAME", "r")
    (repo / "src" / "a.py").unlink()
    main(["--root", str(repo)])
    assert "sparse-checkout: disabled" in capsys.readouterr().out


def test_runner_name_falls_back_when_unset(repo, monkeypatch, capsys):
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    (repo / "src" / "a.py").unlink()
    assert main(["--root", str(repo)]) == 1
    assert "unknown" in capsys.readouterr().out


def test_non_git_directory_is_a_usage_error(tmp_path, capsys):
    assert main(["--root", str(tmp_path)]) == 2


def test_invalid_max_report_rejected(repo):
    with pytest.raises(SystemExit):
        main(["--root", str(repo), "--max-report", "0"])


def test_stale_skip_worktree_with_sparse_disabled_fails(repo, monkeypatch, capsys):
    """A leftover skip-worktree bit is not an exclusion when sparse is off."""
    monkeypatch.setenv("RUNNER_NAME", "runner-3")
    _git(repo, "update-index", "--skip-worktree", "ui/package-lock.json")
    (repo / "ui" / "package-lock.json").unlink()
    assert find_missing(repo) == ["ui/package-lock.json"]
    assert main(["--root", str(repo)]) == 1
    text = capsys.readouterr()
    combined = text.out + text.err
    assert "ui/package-lock.json" in combined
    assert "skip-worktree" in combined
    assert "runner-3" in combined


def test_ls_files_failure_is_a_runner_named_diagnostic(repo, monkeypatch, capsys):
    """A corrupt index yields an ::error:: line and exit 2, not a traceback."""
    monkeypatch.setenv("RUNNER_NAME", "runner-5")
    (repo / ".git" / "index").write_bytes(b"garbage-not-an-index" * 8)
    assert main(["--root", str(repo)]) == 2
    text = capsys.readouterr()
    combined = text.out + text.err
    assert "::error::" in combined
    assert "runner-5" in combined
    assert "Traceback" not in combined


def test_fetch_pinned_tools_action_runs_the_assertion_first():
    """The shared composite action asserts checkout completeness first."""
    yaml = pytest.importorskip("yaml")
    action = yaml.safe_load(
        (
            REPO_ROOT / ".github" / "actions" / "fetch-pinned-tools" / "action.yml"
        ).read_text(encoding="utf-8")
    )
    first = action["runs"]["steps"][0]
    assert "scripts/ci/assert_checkout_complete.py" in first["run"]
