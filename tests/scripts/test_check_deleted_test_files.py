"""Tests for scripts/ci/check_deleted_test_files.py (#10751).

Verifies that Python test file deletions in PRs are accurately detected
relative to the merge base, avoiding false positives when tests are added
to the base branch after the PR branched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.ci import check_deleted_test_files as mod

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> str:
    """Run git in the given directory with deterministic author identity."""
    res = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=tester",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


def _setup_git_history(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Create a git repo simulating main and feature branches.

    Commit history:
      C1 (initial): has tests/unit/test_base.py
      feature branch created at C1
      main branch adds C2: adds tests/unit/test_new_on_main.py
      feature branch adds C3: modifies tests/unit/test_base.py

    Returns:
        (repo_path, c1_sha, main_sha, feature_sha)
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "checkout", "-q", "-b", "main")

    # C1: initial commit with test_base.py
    tests_dir = repo / "tests" / "unit"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_base.py").write_text("def test_base(): pass\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial: add test_base.py")
    c1 = _git(repo, "rev-parse", "HEAD")

    # Create feature branch at C1
    _git(repo, "checkout", "-q", "-b", "feature")

    # Back to main: add test_new_on_main.py (C2)
    _git(repo, "checkout", "-q", "main")
    (tests_dir / "test_new_on_main.py").write_text(
        "def test_new(): pass\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "main: add test_new_on_main.py")
    main_sha = _git(repo, "rev-parse", "HEAD")

    # Feature branch: modify test_base.py (C3)
    _git(repo, "checkout", "-q", "feature")
    (tests_dir / "test_base.py").write_text(
        "def test_base(): assert True\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "feature: modify test_base.py")
    feature_sha = _git(repo, "rev-parse", "HEAD")

    return repo, c1, main_sha, feature_sha


def test_resolve_merge_base_finds_common_ancestor(tmp_path: Path) -> None:
    """resolve_merge_base correctly computes common ancestor commit SHA."""
    repo, c1, main_sha, feature_sha = _setup_git_history(tmp_path)
    base = mod.resolve_merge_base("main", "feature", repo_root=repo)
    assert base == c1


def test_resolve_merge_base_returns_none_for_invalid_ref(tmp_path: Path) -> None:
    """resolve_merge_base returns None when git cannot resolve ref."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    base = mod.resolve_merge_base("nonexistent_branch", "feature", repo_root=repo)
    assert base is None


def test_detect_deleted_test_files_clean_branch_returns_empty(tmp_path: Path) -> None:
    """When no test files are deleted, return empty list."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    deleted = mod.detect_deleted_test_files("main", "feature", repo_root=repo)
    assert deleted == []


def test_detect_deleted_test_files_regression_test_added_on_main(
    tmp_path: Path,
) -> None:
    """Regression test for #10751: tests added on main must not be reported as deleted.

    If direct diff against main was used (`git diff --diff-filter=D main feature`),
    test_new_on_main.py would be reported as deleted because it exists on main
    but not on feature. Using the merge base prevents this false positive.
    """
    repo, _, _, _ = _setup_git_history(tmp_path)
    # Direct diff against main tip would falsely report deletion:
    direct_diff_out = _git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=D",
        "main",
        "feature",
        "--",
        "tests/**/*.py",
    )
    assert "tests/unit/test_new_on_main.py" in direct_diff_out

    # Our merge-base aware detector must report NO deletions:
    deleted = mod.detect_deleted_test_files("main", "feature", repo_root=repo)
    assert deleted == []
    assert "tests/unit/test_new_on_main.py" not in deleted


def test_detect_deleted_test_files_genuine_deletion(tmp_path: Path) -> None:
    """When a test file is genuinely deleted by the PR branch, it is detected."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    # On feature branch, delete test_base.py
    _git(repo, "checkout", "-q", "feature")
    (repo / "tests" / "unit" / "test_base.py").unlink()
    _git(repo, "add", "-u")
    _git(repo, "commit", "-q", "-m", "feature: delete test_base.py")

    deleted = mod.detect_deleted_test_files("main", "feature", repo_root=repo)
    assert deleted == ["tests/unit/test_base.py"]


def test_detect_deleted_test_files_unresolvable_merge_base_raises(
    tmp_path: Path,
) -> None:
    """When merge base cannot be resolved, raises MergeBaseResolutionError."""
    repo = tmp_path / "orphan_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "checkout", "-q", "-b", "b1")
    (repo / "f1.txt").write_text("1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c1")

    # Create disconnected orphan branch
    _git(repo, "checkout", "-q", "--orphan", "b2")
    (repo / "f2.txt").write_text("2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c2")

    with pytest.raises(mod.MergeBaseResolutionError, match="Cannot resolve merge-base"):
        mod.detect_deleted_test_files("b1", "b2", repo_root=repo)


def test_cli_main_clean_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI exits 0 when no test files were deleted."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    code = mod.main(
        ["--base-ref", "main", "--head", "feature", "--repo-root", str(repo)]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "Deleted Python test files require review" not in captured.err
    assert "Deleted Python test files require review" not in captured.out


def test_cli_main_genuine_deletion_exits_one_and_writes_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI exits 1 and writes output file when genuine deletions exist."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    # Delete test_base.py on feature
    _git(repo, "checkout", "-q", "feature")
    (repo / "tests" / "unit" / "test_base.py").unlink()
    _git(repo, "add", "-u")
    _git(repo, "commit", "-q", "-m", "feature: delete test_base.py")

    out_file = tmp_path / "deleted.txt"
    code = mod.main(
        [
            "--base-ref",
            "main",
            "--head",
            "feature",
            "--repo-root",
            str(repo),
            "--output",
            str(out_file),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert (
        "::error::Deleted Python test files require review before CI can proceed."
        in captured.err
    )
    assert out_file.read_text(encoding="utf-8").strip() == "tests/unit/test_base.py"


def test_detect_deleted_test_files_fallback_to_base_when_merge_base_fails(
    tmp_path: Path,
) -> None:
    """When merge-base resolution fails and fallback_to_base is True, uses base_ref."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    # Using a nonexistent base ref raises error when fallback_to_base is False:
    with pytest.raises(mod.MergeBaseResolutionError):
        mod.detect_deleted_test_files(
            "nonexistent_ref", "feature", repo_root=repo, fallback_to_base=False
        )

    # When fallback_to_base is True, diff against base_ref is attempted.
    # If base_ref is valid commit with no common ancestor, it diffs against base_ref.
    # For a completely invalid ref, git diff fails with RuntimeError:
    with pytest.raises(RuntimeError):
        mod.detect_deleted_test_files(
            "nonexistent_ref", "feature", repo_root=repo, fallback_to_base=True
        )


def test_cli_fallback_to_base_flag_passed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI accepts and passes --fallback-to-base flag."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    code = mod.main(
        [
            "--base-ref",
            "main",
            "--head",
            "feature",
            "--repo-root",
            str(repo),
            "--fallback-to-base",
        ]
    )
    assert code == 0


# --- Reviewed deletions allowlist (#11601) ---------------------------------


def _delete_on_feature(repo: Path, *rel_paths: str) -> None:
    """Delete the given test files on the feature branch and commit."""
    _git(repo, "checkout", "-q", "feature")
    for rel in rel_paths:
        (repo / rel).unlink()
    _git(repo, "add", "-u")
    _git(repo, "commit", "-q", "-m", "feature: delete tests")


def _write_allowlist(repo: Path, payload: object) -> Path:
    """Write a reviewed-deletions allowlist at the repo's default location."""
    import json

    path = repo / mod.DEFAULT_REVIEWED_DELETIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def _entry(path: str, issue: str = "#11589", reason: str = "stale") -> dict[str, str]:
    return {"path": path, "issue": issue, "reason": reason}


def _run_cli(repo: Path, out_file: Path) -> int:
    return mod.main(
        [
            "--base-ref",
            "main",
            "--head",
            "feature",
            "--repo-root",
            str(repo),
            "--output",
            str(out_file),
        ]
    )


def test_cli_reviewed_deletion_passes_with_notice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deletion listed in the reviewed allowlist no longer fails the guard."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    _delete_on_feature(repo, "tests/unit/test_base.py")
    _write_allowlist(repo, {"reviewed_deletions": [_entry("tests/unit/test_base.py")]})

    out_file = tmp_path / "deleted.txt"
    assert _run_cli(repo, out_file) == 0
    captured = capsys.readouterr()
    assert "require review" not in captured.err
    assert "::notice::" in captured.err and "tests/unit/test_base.py" in captured.err
    assert out_file.read_text(encoding="utf-8") == ""


def test_cli_unreviewed_deletion_still_fails_alongside_reviewed_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only listed paths are exempt; any other deletion still fails closed."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    _git(repo, "checkout", "-q", "feature")
    (repo / "tests" / "unit" / "test_other.py").write_text(
        "def test_other(): pass\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "feature: add test_other.py")
    # Land test_other.py on main so it exists at the merge base; deleting it
    # on feature afterwards is then a genuine deletion the guard must see.
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-edit", "feature")
    _delete_on_feature(repo, "tests/unit/test_base.py", "tests/unit/test_other.py")
    _write_allowlist(repo, {"reviewed_deletions": [_entry("tests/unit/test_base.py")]})

    out_file = tmp_path / "deleted.txt"
    assert _run_cli(repo, out_file) == 1
    captured = capsys.readouterr()
    assert "Deleted Python test files require review" in captured.err
    assert out_file.read_text(encoding="utf-8").strip() == "tests/unit/test_other.py"


def test_cli_absent_allowlist_keeps_failing_on_deletion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no allowlist file, behaviour is unchanged: deletions fail."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    _delete_on_feature(repo, "tests/unit/test_base.py")
    assert not (repo / mod.DEFAULT_REVIEWED_DELETIONS).exists()

    out_file = tmp_path / "deleted.txt"
    assert _run_cli(repo, out_file) == 1
    assert out_file.read_text(encoding="utf-8").strip() == "tests/unit/test_base.py"


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        {"reviewed_deletions": "tests/unit/test_base.py"},
        {"wrong_key": []},
        {"reviewed_deletions": [_entry("src/not_a_test.py")]},
        {"reviewed_deletions": [_entry("tests/unit/notes.txt")]},
        {"reviewed_deletions": [_entry("tests/unit/test_base.py", issue="11589")]},
        {"reviewed_deletions": [_entry("tests/unit/test_base.py", reason="  ")]},
        {"reviewed_deletions": [{"path": "tests/unit/test_base.py"}]},
        {
            "reviewed_deletions": [
                _entry("tests/unit/test_base.py"),
                _entry("tests/unit/test_base.py"),
            ]
        },
    ],
    ids=[
        "invalid-json",
        "list-not-array",
        "missing-key",
        "path-outside-tests",
        "path-not-python",
        "bad-issue-ref",
        "blank-reason",
        "missing-fields",
        "duplicate-path",
    ],
)
def test_cli_malformed_allowlist_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: object
) -> None:
    """A malformed allowlist is an error, never a silent exemption."""
    repo, _, _, _ = _setup_git_history(tmp_path)
    _delete_on_feature(repo, "tests/unit/test_base.py")
    _write_allowlist(repo, payload)

    assert _run_cli(repo, tmp_path / "deleted.txt") == 1
    assert "::error::" in capsys.readouterr().err


def test_load_reviewed_deletions_accepts_issue_url(tmp_path: Path) -> None:
    """Issue references may be '#N' or a GitHub issue/PR URL."""
    path = tmp_path / "allow.json"
    path.write_text(
        '{"reviewed_deletions": [{"path": "tests/a/test_x.py", '
        '"issue": "https://github.com/D-sorganization/UpstreamDrift/issues/11589", '
        '"reason": "retired"}]}',
        encoding="utf-8",
    )
    assert mod.load_reviewed_deletions(path) == {"tests/a/test_x.py"}
