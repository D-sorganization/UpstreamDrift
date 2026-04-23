from __future__ import annotations

from pathlib import Path

from scripts import check_agent_docs_consistency as checker


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

    _write(readme, "Format code with Ruff\n")
    _write(
        claude,
        "`CLAUDE.md` is the authoritative contributor and agent policy file.\n"
        "**30% coverage minimum**\n"
        "Ruff format.\n",
    )
    _write(
        contributing,
        "# Contributing to UpstreamDrift\n"
        "`CLAUDE.md` is the authoritative source for repository rules and quality gates.\n"
        "Format with Ruff.\n",
    )
    _write(pyproject, '[project]\nname = "upstream-drift"\n')
    _write(
        changelog,
        "All notable changes to UpstreamDrift will be documented in this file.\n",
    )
    _write(ci_standard, "--cov-fail-under=30\n")

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
    _write(claude, "NOT Black\n**10% coverage minimum**\n")
    _write(contributing, "Golf Modeling Suite\nBlack (default settings)\n")
    _write(pyproject, '"black>=26.3.1"\n[tool.black]\n')
    _write(
        changelog,
        "All notable changes to the Golf Modeling Suite will be documented in this file.\n",
    )
    _write(ci_standard, "--cov-fail-under=30\n")

    monkeypatch.setattr(checker, "README", readme)
    monkeypatch.setattr(checker, "CLAUDE", claude)
    monkeypatch.setattr(checker, "CONTRIBUTING", contributing)
    monkeypatch.setattr(checker, "PYPROJECT", pyproject)
    monkeypatch.setattr(checker, "CHANGELOG", changelog)
    monkeypatch.setattr(checker, "CI_STANDARD", ci_standard)

    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "README.md still advertises Black instead of Ruff." in output
    assert "CLAUDE.md must match the CI coverage floor of 30%." in output
    assert "pyproject.toml still defines a [tool.black] section." in output
