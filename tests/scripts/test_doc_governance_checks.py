from __future__ import annotations

import json
from pathlib import Path

from scripts import check_doc_catalog, check_doc_size_budget


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
