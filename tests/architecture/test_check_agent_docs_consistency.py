from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_agent_docs_consistency as checker

pytestmark = pytest.mark.unit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_agent_docs_consistency_passes_for_aligned_files(
    tmp_path: Path, monkeypatch
) -> None:
    readme = tmp_path / "README.md"
    claude = tmp_path / "CLAUDE.md"
    contributing = tmp_path / "CONTRIBUTING.md"
    pyproject = tmp_path / "pyproject.toml"
    changelog = tmp_path / "CHANGELOG.md"
    ci_standard = tmp_path / ".github" / "workflows" / "ci-standard.yml"
    rust_core = tmp_path / "rust_core"

    _write(readme, "Format code with Ruff\n")
    _write(
        claude,
        "`CLAUDE.md` is the authoritative contributor and agent policy file.\n"
        "\n"
        "A unified platform for golf swing analysis across multiple physics engines.\n"
        "\n"
        "- `rust_core/` contains optional Rust kernels built with Maturin.\n"
        "- Python support starts at 3.10, and CI runs Python 3.11.\n"
        "- Coverage threshold is the value of `fail_under` in `pyproject.toml [tool.coverage.report]`.\n"
        "- PRs target `main`; use focused topic branches such as `fix/...`, `feat/...`, `chore/...`, or `claude/...`.\n",
    )
    _write(
        contributing,
        "# Contributing to UpstreamDrift\n"
        "`CLAUDE.md` is the authoritative source for repository rules and quality gates.\n"
        "Format with Ruff.\n",
    )
    _write(
        pyproject,
        '[project]\nname = "upstream-drift"\n[tool.coverage.report]\nfail_under = 45\n',
    )
    _write(
        changelog,
        "All notable changes to UpstreamDrift will be documented in this file.\n",
    )
    _write(ci_standard, "--cov-fail-under=45\n")
    rust_core.mkdir(parents=True)

    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "README", readme)
    monkeypatch.setattr(checker, "CLAUDE", claude)
    monkeypatch.setattr(checker, "CONTRIBUTING", contributing)
    monkeypatch.setattr(checker, "PYPROJECT", pyproject)
    monkeypatch.setattr(checker, "CHANGELOG", changelog)
    monkeypatch.setattr(checker, "CI_STANDARD", ci_standard)

    assert checker.main() == 0


def test_agent_docs_consistency_fails_on_black_and_old_coverage(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    readme = tmp_path / "README.md"
    claude = tmp_path / "CLAUDE.md"
    contributing = tmp_path / "CONTRIBUTING.md"
    pyproject = tmp_path / "pyproject.toml"
    changelog = tmp_path / "CHANGELOG.md"
    ci_standard = tmp_path / ".github" / "workflows" / "ci-standard.yml"

    _write(readme, "code%20style-black\nFormat code with black and ruff\n")
    _write(
        claude,
        "`CLAUDE.md` is the authoritative contributor and agent policy file.\n"
        "\n"
        "`CLAUDE.md` is the authoritative contributor and agent policy file.\n"
        "\n"
        "NOT Black\n"
        "**10% coverage minimum**\n"
        "`rust/` contains optional Rust kernels built with Maturin.\n",
    )
    _write(contributing, "Golf Modeling Suite\nBlack (default settings)\n")
    _write(
        pyproject,
        '[project]\nname = "upstream-drift"\n'
        'black_dep = "\\"black>=26.3.1\\""\n'
        "[tool.black]\n"
        "line-length = 88\n"
        "[tool.coverage.report]\nfail_under = 45\n",
    )
    _write(
        changelog,
        "All notable changes to the Golf Modeling Suite will be documented in this file.\n",
    )
    _write(ci_standard, "--cov-fail-under=10\n")

    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "README", readme)
    monkeypatch.setattr(checker, "CLAUDE", claude)
    monkeypatch.setattr(checker, "CONTRIBUTING", contributing)
    monkeypatch.setattr(checker, "PYPROJECT", pyproject)
    monkeypatch.setattr(checker, "CHANGELOG", changelog)
    monkeypatch.setattr(checker, "CI_STANDARD", ci_standard)

    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "README.md still advertises Black instead of Ruff." in output
    assert (
        "CLAUDE.md cites 10% coverage but pyproject.toml [tool.coverage.report] fail_under is 45%."
        in output
    )
    assert "CLAUDE.md references a missing path: rust/" in output
    assert "CLAUDE.md contains duplicate paragraphs." in output
    assert (
        "CI workflow coverage floor 10% does not match pyproject.toml fail_under 45%."
        in output
    )
    assert "pyproject.toml still defines a [tool.black] section." in output


def test_glob_and_brace_references_are_not_treated_as_missing_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """Glob/brace patterns are documentation, not literal files (#6620)."""
    errors: list[str] = []
    claude = (
        "See `scripts/**` and `tests/**` for tooling.\n"
        "Stdout exceptions: `src/shared/python/codemap/{cli,watcher,mcp_server}.py` "
        "and `src/shared/python/sidekick/{__main__,standalone/runner}.py`.\n"
    )

    monkeypatch.setattr(checker, "ROOT", tmp_path)
    checker._assert_path_references_exist(claude, errors)

    assert errors == []


def test_is_glob_pattern_detects_metacharacters() -> None:
    assert checker._is_glob_pattern("scripts/**")
    assert checker._is_glob_pattern("src/a/{b,c}.py")
    assert checker._is_glob_pattern("src/a/file?.py")
    assert checker._is_glob_pattern("src/a/[abc].py")
    assert not checker._is_glob_pattern("src/launchers/embedded_host.py")


def test_optional_and_explicit_hub_paths_keep_required_local_checks(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    text = (
        "Check `docs/codemap.md` when present.\n"
        "Create it from Repository_Management's `docs/templates/HANDOFF.md`.\n"
        "Read `docs/required.md` before editing.\n"
        "Read `docs/codemap.md` before release.\n"
        "Read `docs/mandatory.md`. Check `docs/optional.md` when present.\n"
        "When `docs/agent_context/catalog.json` exists, read `docs/contract.md`.\n"
        "If `docs/optional.json` exists, inspect its content.\n"
    )
    errors: list[str] = []
    checker._assert_path_references_exist(text, errors)
    assert errors == [
        "CLAUDE.md references a missing path: docs/required.md",
        "CLAUDE.md references a missing path: docs/codemap.md",
        "CLAUDE.md references a missing path: docs/mandatory.md",
        "CLAUDE.md references a missing path: docs/contract.md",
    ]


def test_managed_notices_and_rules_are_not_duplicate_instructions() -> None:
    notice = (
        "> This section is managed centrally by Repository_Management and synced fleet-wide.\n"
        "> Do NOT edit it directly in individual repositories — edit the source in "
        "Repository_Management/AGENTS.md."
    )
    text = f"---\n\n{notice}\n\n---\n\n{notice}\n\nRepeated rule.\n\nRepeated rule."
    assert checker._iter_duplicate_paragraphs(text) == ["Repeated rule."]


def test_repeated_headings_are_not_flagged_as_duplicate_paragraphs() -> None:
    text = (
        "# Top Level\n\n"
        "### The Rules\n\n"
        "Rule 1 explanation.\n\n"
        "### Next Section\n\n"
        "### The Rules\n\n"
        "Rule 2 explanation.\n"
    )
    assert checker._iter_duplicate_paragraphs(text) == []


@pytest.mark.parametrize(
    "text",
    [
        # Exact fleet-managed sentence synced into CLAUDE.md by #10741 (#10743).
        "issues. Standard: Repository_Management `docs/fleet-deferred-validation.md`.\n",
        "Create it from Repository_Management's `docs/templates/HANDOFF.md`.\n",
        "See Repository_Management’s `docs/fleet-deferred-validation.md`.\n",
    ],
)
def test_sibling_repo_qualified_paths_are_not_local(
    tmp_path: Path, monkeypatch, text: str
) -> None:
    """A path immediately qualified by a sibling repo name is cross-repo (#10743)."""
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    errors: list[str] = []
    checker._assert_path_references_exist(text, errors)
    assert errors == []


def test_bare_local_missing_path_still_fails_near_sibling_repo_mention(
    tmp_path: Path, monkeypatch
) -> None:
    """The exemption must not leak to unqualified local paths (#10743)."""
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    text = (
        "Repository_Management owns fleet policy. Read `docs/local_missing.md`.\n"
        "Read `docs/also_missing.md` before Repository_Management `docs/x.md`.\n"
    )
    errors: list[str] = []
    checker._assert_path_references_exist(text, errors)
    assert errors == [
        "CLAUDE.md references a missing path: docs/local_missing.md",
        "CLAUDE.md references a missing path: docs/also_missing.md",
    ]


def test_fleet_managed_sections_are_exempt_from_local_path_checks(
    tmp_path: Path, monkeypatch
) -> None:
    """Paths inside <!-- BEGIN FLEET-MANAGED --> blocks are synced from RM and not local."""
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    text = (
        "<!-- BEGIN FLEET-MANAGED: agent-lanes -->\n"
        "Staff Hub role playbooks `docs/fleet-night-watch.md`.\n"
        "<!-- END FLEET-MANAGED: agent-lanes -->\n"
        "Local required file `docs/required_local.md`.\n"
    )
    errors: list[str] = []
    checker._assert_path_references_exist(text, errors)
    assert errors == [
        "CLAUDE.md references a missing path: docs/required_local.md",
    ]
