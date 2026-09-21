"""Tests for scripts/check_agent_docs_consistency.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_agent_docs_consistency as mod


def test_load_coverage_threshold() -> None:
    text = "[tool.coverage.report]\nfail_under = 75\n"
    assert mod._load_coverage_threshold(text) == 75


def test_iter_duplicate_paragraphs_detects_dupes() -> None:
    text = "hello world\n\nhello world\n\nunique"
    dupes = mod._iter_duplicate_paragraphs(text)
    assert dupes == ["hello world"]


def test_iter_duplicate_paragraphs_no_dupes() -> None:
    text = "para one\n\npara two\n\npara three"
    assert mod._iter_duplicate_paragraphs(text) == []


def test_iter_repo_relative_paths_collects() -> None:
    text = "see `src/api/foo.py` and `README.md` and `https://x.com` and `with space`"
    paths = mod._iter_repo_relative_paths(text)
    assert "src/api/foo.py" in paths
    assert "README.md" in paths
    assert "https://x.com" not in paths
    assert "with space" not in paths


def test_iter_repo_relative_paths_strips_at_prefix() -> None:
    text = "ref `@docs/x.md`"
    paths = mod._iter_repo_relative_paths(text)
    assert "docs/x.md" in paths


def test_iter_repo_relative_paths_ignores_non_repo_roots() -> None:
    text = "see `random/x.py` and `/abs/path`"
    paths = mod._iter_repo_relative_paths(text)
    assert "random/x.py" not in paths


def test_iter_repo_relative_paths_deduplicates() -> None:
    text = "see `src/x.py` and again `src/x.py`"
    paths = mod._iter_repo_relative_paths(text)
    assert paths.count("src/x.py") == 1


def test_assert_contains_appends_error() -> None:
    errors: list[str] = []
    mod._assert_contains("abc", "z", "msg", errors)
    assert errors == ["msg"]
    mod._assert_contains("abc", "b", "msg2", errors)
    assert errors == ["msg"]


def test_assert_not_contains_appends_error() -> None:
    errors: list[str] = []
    mod._assert_not_contains("abc", "b", "msg", errors)
    assert errors == ["msg"]
    mod._assert_not_contains("abc", "z", "ok", errors)
    assert errors == ["msg"]


def test_assert_coverage_alignment_match() -> None:
    errors: list[str] = []
    claude = "Coverage gate is 75% in CI.\n"
    ci = "pytest --cov-fail-under=75 ..."
    mod._assert_coverage_alignment(claude, ci, 75, errors)
    assert errors == []


def test_assert_coverage_alignment_claude_mismatch() -> None:
    errors: list[str] = []
    claude = "Coverage is 80%\n"
    ci = "--cov-fail-under=75"
    mod._assert_coverage_alignment(claude, ci, 75, errors)
    assert any("80%" in e for e in errors)


def test_assert_coverage_alignment_no_gate() -> None:
    errors: list[str] = []
    mod._assert_coverage_alignment("Coverage is 75%", "no gate here", 75, errors)
    assert any("must define a --cov-fail-under gate" in e for e in errors)


def test_assert_coverage_alignment_ci_mismatch() -> None:
    errors: list[str] = []
    mod._assert_coverage_alignment("Coverage is 75%", "--cov-fail-under=50", 75, errors)
    assert any("CI workflow coverage floor" in e for e in errors)


def test_assert_path_references_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "exists.py").touch()
    errors: list[str] = []
    mod._assert_path_references_exist(
        "see `src/exists.py` and `src/missing.py`", errors
    )
    assert any("src/missing.py" in e for e in errors)
    assert not any("src/exists.py" in e for e in errors)


def test_assert_path_references_skips_glob_and_brace_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Glob/brace references describe patterns, not literal files (#6620)."""
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    errors: list[str] = []
    mod._assert_path_references_exist(
        "see `scripts/**` and `tests/**` and "
        "`src/shared/python/codemap/{cli,watcher,mcp_server}.py`",
        errors,
    )
    assert errors == []


def test_is_glob_pattern() -> None:
    assert mod._is_glob_pattern("scripts/**")
    assert mod._is_glob_pattern("src/x/{a,b}.py")
    assert mod._is_glob_pattern("src/x/f?.py")
    assert mod._is_glob_pattern("src/x/[ab].py")
    assert not mod._is_glob_pattern("src/x/literal.py")


def test_assert_no_duplicate_paragraphs() -> None:
    errors: list[str] = []
    mod._assert_no_duplicate_paragraphs("a\n\nb\n\nc", errors)
    assert errors == []
    mod._assert_no_duplicate_paragraphs("a\n\na", errors)
    assert errors == ["CLAUDE.md contains duplicate paragraphs."]


def _setup_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "README.md").write_text("UpstreamDrift README\n")
    claude = (
        "`CLAUDE.md` is the authoritative contributor and agent policy file.\n"
        "\nCoverage is 75% per CI.\n"
    )
    (tmp_path / "CLAUDE.md").write_text(claude)
    (tmp_path / "CONTRIBUTING.md").write_text(
        "`CLAUDE.md` is the authoritative source for repository rules and "
        "quality gates.\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[tool.coverage.report]\nfail_under = 75\n"
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "All notable changes to UpstreamDrift will be documented in this file.\n"
    )
    (tmp_path / ".github" / "workflows" / "ci-standard.yml").write_text(
        "run: pytest --cov-fail-under=75\n"
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "README", tmp_path / "README.md")
    monkeypatch.setattr(mod, "CLAUDE", tmp_path / "CLAUDE.md")
    monkeypatch.setattr(mod, "CONTRIBUTING", tmp_path / "CONTRIBUTING.md")
    monkeypatch.setattr(mod, "PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(mod, "CHANGELOG", tmp_path / "CHANGELOG.md")
    monkeypatch.setattr(
        mod, "CI_STANDARD", tmp_path / ".github" / "workflows" / "ci-standard.yml"
    )
    return tmp_path


def test_main_passes_with_clean_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_docs(tmp_path, monkeypatch)
    assert mod.main() == 0
    assert "OK" in capsys.readouterr().out


def test_main_fails_with_stale_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _setup_docs(tmp_path, monkeypatch)
    (root / "README.md").write_text("code%20style-black is bad\n")
    assert mod.main() == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_fails_with_stale_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_docs(tmp_path, monkeypatch)
    (root / "pyproject.toml").write_text(
        "[tool.coverage.report]\nfail_under = 75\n\n[tool.black]\nline-length=88\n"
    )
    assert mod.main() == 1


def test_main_fails_with_stale_contributing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_docs(tmp_path, monkeypatch)
    (root / "CONTRIBUTING.md").write_text("Format with black and ruff\n")
    assert mod.main() == 1
