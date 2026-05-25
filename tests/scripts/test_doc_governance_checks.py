from __future__ import annotations

import json
from pathlib import Path

from scripts import check_doc_catalog, check_doc_size_budget, check_docs_governance


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_doc_size_budget_rejects_oversized_markdown_without_exception(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path / "docs" / "large.md", "x" * (check_doc_size_budget.MAX_BYTES + 1))
    _write(
        tmp_path / "scripts" / "config" / "doc_size_budget.json",
        json.dumps({"max_bytes": check_doc_size_budget.MAX_BYTES, "exceptions": []}),
    )
    monkeypatch.setattr(check_doc_size_budget, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_doc_size_budget,
        "CONFIG_PATH",
        tmp_path / "scripts" / "config" / "doc_size_budget.json",
    )

    assert check_doc_size_budget.main() == 1


def test_doc_size_budget_allows_owned_unexpired_exception(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path / "docs" / "large.md", "x" * (check_doc_size_budget.MAX_BYTES + 1))
    _write(
        tmp_path / "scripts" / "config" / "doc_size_budget.json",
        json.dumps(
            {
                "max_bytes": check_doc_size_budget.MAX_BYTES,
                "exceptions": [
                    {
                        "path": "docs/large.md",
                        "owner": "@docs-team",
                        "expires_on": "2099-01-01",
                    }
                ],
            }
        ),
    )
    monkeypatch.setattr(check_doc_size_budget, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_doc_size_budget,
        "CONFIG_PATH",
        tmp_path / "scripts" / "config" / "doc_size_budget.json",
    )

    assert check_doc_size_budget.main() == 0


def test_doc_catalog_requires_every_docs_directory_with_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "docs" / "testing").mkdir()
    _write(
        tmp_path / "docs" / "index.md",
        "# Docs\n\n| Directory | Owner | Stability | Description |\n"
        "| --- | --- | --- | --- |\n"
        "| `api/` | @docs-team | stable | API reference and integration docs. |\n",
    )
    monkeypatch.setattr(check_doc_catalog, "ROOT", tmp_path)
    monkeypatch.setattr(check_doc_catalog, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(check_doc_catalog, "INDEX_PATH", tmp_path / "docs" / "index.md")
    monkeypatch.setattr(check_doc_catalog, "README_PATH", tmp_path / "README.md")
    monkeypatch.setattr(
        check_doc_catalog, "PYPROJECT_PATH", tmp_path / "pyproject.toml"
    )

    assert check_doc_catalog.main() == 1


def test_doc_catalog_accepts_complete_catalog_and_rendered_docs_link(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "docs" / "testing").mkdir()
    _write(
        tmp_path / "docs" / "index.md",
        "# Docs\n\n| Directory | Owner | Stability | Description |\n"
        "| --- | --- | --- | --- |\n"
        "| `api/` | @docs-team | stable | API reference and integration docs. |\n"
        "| `testing/` | @quality-team | draft | Test strategy and validation guidance. |\n",
    )
    _write(
        tmp_path / "README.md",
        "For detailed documentation, please visit the "
        "**[Documentation Hub](https://upstream-drift.readthedocs.io)**.\n",
    )
    _write(
        tmp_path / "pyproject.toml",
        '[project.urls]\nDocumentation = "https://upstream-drift.readthedocs.io"\n',
    )
    monkeypatch.setattr(check_doc_catalog, "ROOT", tmp_path)
    monkeypatch.setattr(check_doc_catalog, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(check_doc_catalog, "INDEX_PATH", tmp_path / "docs" / "index.md")
    monkeypatch.setattr(check_doc_catalog, "README_PATH", tmp_path / "README.md")
    monkeypatch.setattr(
        check_doc_catalog, "PYPROJECT_PATH", tmp_path / "pyproject.toml"
    )

    assert check_doc_catalog.main() == 0


def test_docs_governance_rejects_duplicate_root_process_directories(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path / "docs" / "README.md", "# Docs\n")
    _write(tmp_path / "docs" / "assessments" / "README.md", "# Assessments\n")
    _write(tmp_path / "docs" / "adr" / "README.md", "# ADRs\n")
    _write(tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md", "# ADR Template\n")
    _write(
        tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        "# Docs Governance\n",
    )
    (tmp_path / "docs" / "issues").mkdir(parents=True)
    (tmp_path / "issues").mkdir()
    (tmp_path / "assessments").mkdir()
    (tmp_path / ".github" / "issues").mkdir(parents=True)
    monkeypatch.setattr(check_docs_governance, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_docs_governance,
        "REQUIRED_FILES",
        [
            tmp_path / "docs" / "README.md",
            tmp_path / "docs" / "assessments" / "README.md",
            tmp_path / "docs" / "adr" / "README.md",
            tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md",
            tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        ],
    )
    monkeypatch.setattr(check_docs_governance, "_git_changed_files", list)

    assert check_docs_governance.main() == 1


def test_docs_governance_rejects_duplicate_source_of_truth_headings(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path / "docs" / "README.md", "# Docs\n")
    _write(tmp_path / "docs" / "assessments" / "README.md", "# Assessments\n")
    _write(tmp_path / "docs" / "adr" / "README.md", "# ADRs\n")
    _write(tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md", "# ADR Template\n")
    _write(
        tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        "# Docs Governance\n",
    )
    _write(
        tmp_path / "SPEC.md",
        "# SPEC\n\n"
        "## SPEC Ownership and Update Cadence\n\n"
        "- **Owner:** @diete\n\n"
        "## 1. Identity\n\n"
        "## SPEC Ownership and Update Cadence\n\n"
        "- **Owner:** D-sorganization\n",
    )
    monkeypatch.setattr(check_docs_governance, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_docs_governance,
        "REQUIRED_FILES",
        [
            tmp_path / "docs" / "README.md",
            tmp_path / "docs" / "assessments" / "README.md",
            tmp_path / "docs" / "adr" / "README.md",
            tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md",
            tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        ],
    )
    monkeypatch.setattr(check_docs_governance, "_git_changed_files", list)

    assert check_docs_governance.main() == 1


def test_docs_governance_rejects_duplicate_adr_numbers(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _write(tmp_path / "docs" / "README.md", "# Docs\n")
    _write(tmp_path / "docs" / "assessments" / "README.md", "# Assessments\n")
    _write(tmp_path / "docs" / "adr" / "README.md", "# ADRs\n")
    _write(tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md", "# ADR Template\n")
    _write(
        tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        "# Docs Governance\n",
    )
    _write(tmp_path / "docs" / "adr" / "0005-first.md", "# ADR-0005: First\n")
    _write(tmp_path / "docs" / "adr" / "0005-second.md", "# ADR-0005: Second\n")
    monkeypatch.setattr(check_docs_governance, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_docs_governance,
        "REQUIRED_FILES",
        [
            tmp_path / "docs" / "README.md",
            tmp_path / "docs" / "assessments" / "README.md",
            tmp_path / "docs" / "adr" / "README.md",
            tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md",
            tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        ],
    )
    monkeypatch.setattr(check_docs_governance, "_git_changed_files", list)

    assert check_docs_governance.main() == 1
    captured = capsys.readouterr()
    assert "Duplicate ADR numbering detected" in captured.err
    assert "duplicate ADR number 0005: 0005-first.md, 0005-second.md" in captured.err

    captured = capsys.readouterr()
    assert "Duplicate ADR numbering detected:" in captured.err
    assert "duplicate ADR number 0005: 0005-first.md, 0005-second.md" in captured.err


def test_docs_governance_rejects_missing_examples_entries(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path / "docs" / "README.md", "# Docs\n")
    _write(tmp_path / "docs" / "assessments" / "README.md", "# Assessments\n")
    _write(tmp_path / "docs" / "adr" / "README.md", "# ADRs\n")
    _write(tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md", "# ADR Template\n")
    _write(
        tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        "# Docs Governance\n",
    )
    _write(
        tmp_path / "docs" / "examples" / "index.rst",
        "Examples\n========\n\n.. toctree::\n   :maxdepth: 1\n\n   basic_swing\n",
    )
    monkeypatch.setattr(check_docs_governance, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_docs_governance,
        "REQUIRED_FILES",
        [
            tmp_path / "docs" / "README.md",
            tmp_path / "docs" / "assessments" / "README.md",
            tmp_path / "docs" / "adr" / "README.md",
            tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md",
            tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        ],
    )
    monkeypatch.setattr(check_docs_governance, "_git_changed_files", list)

    assert check_docs_governance.main() == 1


def test_docs_governance_allows_unique_adr_numbers(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "docs" / "README.md", "# Docs\n")
    _write(tmp_path / "docs" / "assessments" / "README.md", "# Assessments\n")
    _write(tmp_path / "docs" / "adr" / "README.md", "# ADRs\n")
    _write(tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md", "# ADR Template\n")
    _write(
        tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        "# Docs Governance\n",
    )
    _write(tmp_path / "docs" / "adr" / "0005-first.md", "# ADR-0005: First\n")
    _write(tmp_path / "docs" / "adr" / "0006-second.md", "# ADR-0006: Second\n")
    monkeypatch.setattr(check_docs_governance, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_docs_governance,
        "REQUIRED_FILES",
        [
            tmp_path / "docs" / "README.md",
            tmp_path / "docs" / "assessments" / "README.md",
            tmp_path / "docs" / "adr" / "README.md",
            tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md",
            tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        ],
    )
    monkeypatch.setattr(check_docs_governance, "_git_changed_files", list)

    assert check_docs_governance.main() == 0


def test_docs_governance_accepts_examples_entries_with_backing_docs(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path / "docs" / "README.md", "# Docs\n")
    _write(tmp_path / "docs" / "assessments" / "README.md", "# Assessments\n")
    _write(tmp_path / "docs" / "adr" / "README.md", "# ADRs\n")
    _write(tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md", "# ADR Template\n")
    _write(
        tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        "# Docs Governance\n",
    )
    _write(
        tmp_path / "docs" / "examples" / "index.rst",
        "Examples\n========\n\n.. toctree::\n   :maxdepth: 1\n\n   basic_swing\n",
    )
    _write(
        tmp_path / "docs" / "examples" / "basic_swing.rst", "Basic Swing\n===========\n"
    )
    monkeypatch.setattr(check_docs_governance, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_docs_governance,
        "REQUIRED_FILES",
        [
            tmp_path / "docs" / "README.md",
            tmp_path / "docs" / "assessments" / "README.md",
            tmp_path / "docs" / "adr" / "README.md",
            tmp_path / "docs" / "adr" / "ADR_TEMPLATE.md",
            tmp_path / "docs" / "governance" / "DOCS_GOVERNANCE.md",
        ],
    )
    monkeypatch.setattr(check_docs_governance, "_git_changed_files", list)

    assert check_docs_governance.main() == 0
