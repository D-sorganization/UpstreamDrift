"""Tests for scripts/ci/check_phantom_guard_paths.py (UD #9091)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.ci import check_phantom_guard_paths as mod


PR_BODY = "Fixes #9091 and Resolves #700, closes #800."


def _stub_loader(
    bodies: dict[str, str | None],
) -> mod.IssueBodyLoader:
    def loader(number: str) -> str | None:
        return bodies.get(number)

    return loader


def _init_repo(tmp_path: Path) -> tuple[str, str, Path]:
    """Create a real two-commit git repo; return (base_sha, head_sha, root)."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            [
                "git", "-C", str(repo),
                "-c", "user.email=t@example.com",
                "-c", "user.name=t",
                *args,
            ],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("checkout", "-q", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("a\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git("checkout", "-q", "-b", "feature")
    (repo / "src" / "b.py").write_text("b\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "feat: b")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return base, head, repo


# ----- pure helpers -----


def test_referenced_issue_numbers_extracts_unique_sorted() -> None:
    assert mod._referenced_issue_numbers(PR_BODY) == ["700", "800", "9091"]


def test_referenced_issue_numbers_ignores_plain_mentions() -> None:
    assert mod._referenced_issue_numbers("see #123 and fixes #456") == ["456"]


def test_issue_referenced_paths_extracts_unique_sorted() -> None:
    body = "touch api/x.py, src/x.py and api/x.py again"
    assert mod._issue_referenced_paths(body) == ["api/x.py", "src/x.py"]


def test_issue_referenced_paths_ignores_unprefixed_paths() -> None:
    assert mod._issue_referenced_paths("docs/other.py and random.txt") == []


def test_first_path_match_directory_prefix() -> None:
    changed = ["src/dashboard/deep/nested/main.py"]
    assert mod._first_path_match(changed, ["src/dashboard/deep"]) == (
        "src/dashboard/deep"
    )


def test_first_path_match_none() -> None:
    assert mod._first_path_match(["docs/readme.md"], ["src/dashboard/main.py"]) is None


def test_env_changed_files_parses_lines() -> None:
    value = "src/a.py\n\nsrc/b.py\r\n"
    assert mod._env_changed_files(value) == ["src/a.py", "src/b.py"]
    assert mod._env_changed_files("") == []
    assert mod._env_changed_files(None) == []


# ----- git changed-file source -----


def test_git_changed_files_merge_base_success(tmp_path: Path) -> None:
    base, head, repo = _init_repo(tmp_path)
    assert mod._git_changed_files(base, head, repo) == ["src/b.py"]


def test_git_changed_files_returns_none_when_merge_base_fails(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    repo = tmp_path / "repo"
    # A bogus pair of SHAs cannot be merged-base against in a healthy repo.
    assert mod._git_changed_files("0" * 40, "1" * 40, repo) is None


def test_git_changed_files_survives_empty_diff(tmp_path: Path) -> None:
    base, _head, repo = _init_repo(tmp_path)
    # base vs base is a valid, successful empty diff.
    assert mod._git_changed_files(base, base, repo) == []


# ----- changed-file source resolution (UD #9091 fallback chain) -----


def test_resolve_prefers_git_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base, head, _repo = _init_repo(tmp_path)

    def api_loader() -> list[str] | None:
        raise AssertionError("API must not be consulted when git succeeds")

    monkeypatch.setattr(
        mod, "_git_changed_files", lambda b, h, r: ["src/b.py"]
    )
    files, source = mod._resolve_changed_files(base, head, "", api_loader)
    assert files == ["src/b.py"]
    assert source == "git"


def test_resolve_falls_back_to_env_list_when_merge_base_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def api_loader() -> list[str] | None:
        raise AssertionError("API must not run when the env list is present")

    monkeypatch.setattr(mod, "_git_changed_files", lambda b, h, r: None)
    files, source = mod._resolve_changed_files(
        "base", "head", "src/dashboard/main.py\n", api_loader
    )
    assert files == ["src/dashboard/main.py"]
    assert source == "env"


def test_resolve_falls_back_to_api_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def api_loader() -> list[str] | None:
        return ["src/dashboard/main.py"]

    monkeypatch.setattr(mod, "_git_changed_files", lambda b, h, r: None)
    files, source = mod._resolve_changed_files(None, None, "", api_loader)
    assert files == ["src/dashboard/main.py"]
    assert source == "api"


def test_resolve_fails_closed_with_diagnostic_when_both_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def api_loader() -> list[str] | None:
        return None

    monkeypatch.setattr(mod, "_git_changed_files", lambda b, h, r: None)
    with pytest.raises(mod.ChangedFilesUnavailableError, match="shallow"):
        mod._resolve_changed_files(None, None, "", api_loader)


# ----- rule-3 evaluation -----


def test_evaluate_passes_when_list_contains_referenced_paths() -> None:
    changed = ["src/dashboard/main.py", "docs/x.md"]
    failures = mod._evaluate_rule3(
        PR_BODY,
        changed,
        _stub_loader({"9091": "see src/dashboard/main.py please"}),
    )
    assert failures == []


def test_evaluate_fails_when_list_lacks_referenced_paths() -> None:
    failures = mod._evaluate_rule3(
        PR_BODY,
        ["docs/x.md"],
        _stub_loader({"9091": "src/dashboard/main.py is the target"}),
    )
    assert len(failures) == 1
    assert "#9091" in failures[0]
    assert "none of the paths" in failures[0]


def test_evaluate_skips_issue_without_path_references() -> None:
    failures = mod._evaluate_rule3("Fixes #1", [], _stub_loader({"1": "no paths"}))
    assert failures == []


def test_evaluate_skips_inaccessible_issue() -> None:
    failures = mod._evaluate_rule3("Fixes #1", [], _stub_loader({"1": None}))
    assert failures == []


# ----- main() wiring (API layer mocked; no network) -----


def test_main_shallow_fetch_failure_uses_api_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_merge_base", lambda b, h, r: None)
    monkeypatch.setattr(
        mod,
        "_api_changed_files",
        lambda n, r: ["src/dashboard/main.py"],
    )
    monkeypatch.setattr(
        mod,
        "_load_issue_body",
        lambda n, r: "target src/dashboard/main.py",
    )
    argv = [
        "--base-sha", "1111111111111111111111111111111111111111",
        "--head-sha", "2222222222222222222222222222222222222222",
        "--pr-number", "9091",
        "--repo", "D-sorganization/UpstreamDrift",
        "--pr-body", "Fixes #9091",
    ]
    assert mod.main(argv) == 0


def test_main_api_list_without_referenced_paths_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(mod, "_merge_base", lambda b, h, r: None)
    monkeypatch.setattr(mod, "_api_changed_files", lambda n, r: ["docs/x.md"])
    monkeypatch.setattr(
        mod,
        "_load_issue_body",
        lambda n, r: "target src/dashboard/main.py",
    )
    argv = [
        "--base-sha", "1" * 40,
        "--head-sha", "2" * 40,
        "--pr-number", "9091",
        "--repo", "D-sorganization/UpstreamDrift",
        "--pr-body", "Fixes #9091",
    ]
    assert mod.main(argv) == 1
    assert "none of the paths" in capsys.readouterr().out


def test_main_missing_both_sources_fails_with_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(mod, "_merge_base", lambda b, h, r: None)
    monkeypatch.setattr(mod, "_api_changed_files", lambda n, r: None)
    argv = [
        "--base-sha", "1" * 40,
        "--head-sha", "2" * 40,
        "--pr-number", "9091",
        "--repo", "D-sorganization/UpstreamDrift",
        "--pr-body", "Fixes #9091",
    ]
    assert mod.main(argv) == 1
    out = capsys.readouterr().out
    assert "unable to determine changed files" in out


def test_main_merge_base_success_uses_local_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod, "_git_changed_files", lambda b, h, r: ["src/dashboard/main.py"]
    )
    monkeypatch.setattr(
        mod,
        "_load_issue_body",
        lambda n, r: "target src/dashboard/main.py",
    )
    argv = [
        "--base-sha", "1" * 40,
        "--head-sha", "2" * 40,
        "--pr-number", "9091",
        "--repo", "D-sorganization/UpstreamDrift",
        "--pr-body", "Fixes #9091",
    ]
    assert mod.main(argv) == 0