"""Tests for Mermaid C4 architecture map contract (Repository_Management #1594, #1595)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.architecture_map_contract import (
    ArchitectureMapContractError,
    validate_architecture_map,
)

ROOT = Path(__file__).resolve().parents[1]
C4_PATH = ROOT / "docs" / "architecture" / "C4.md"


pytestmark = pytest.mark.unit


def test_validate_architecture_map_real_doc() -> None:
    """The repository's canonical docs/architecture/C4.md must validate cleanly."""
    assert C4_PATH.is_file(), f"Expected architecture map at {C4_PATH}"
    result = validate_architecture_map(C4_PATH)
    assert result.is_valid is True
    assert result.context_views >= 1
    assert result.container_views >= 1
    assert len(result.feature_map_entries) >= 3
    assert len(result.changelog_entries) >= 1


def test_validate_architecture_map_rejects_missing_views(tmp_path: Path) -> None:
    """Validator must reject docs missing C4Context or C4Container."""
    doc = tmp_path / "C4.md"
    doc.write_text(
        '# Architecture Map\n\n```mermaid\nC4Context\nPerson(u, "User")\n```\n',
        encoding="utf-8",
    )
    with pytest.raises(ArchitectureMapContractError, match="C4Container"):
        validate_architecture_map(doc)


def test_validate_architecture_map_rejects_missing_feature_map(tmp_path: Path) -> None:
    """Validator must reject docs missing the Feature Map table."""
    doc = tmp_path / "C4.md"
    doc.write_text(
        "# Architecture Map\n\n"
        '```mermaid\nC4Context\nPerson(u, "User")\n```\n\n'
        '```mermaid\nC4Container\nContainer(c, "App")\n```\n',
        encoding="utf-8",
    )
    with pytest.raises(ArchitectureMapContractError, match="Feature Map"):
        validate_architecture_map(doc)


def test_validate_architecture_map_rejects_missing_changelog(tmp_path: Path) -> None:
    """Validator must reject docs missing the Architecture Change Log."""
    doc = tmp_path / "C4.md"
    doc.write_text(
        "# Architecture Map\n\n"
        '```mermaid\nC4Context\nPerson(u, "User")\n```\n\n'
        '```mermaid\nC4Container\nContainer(c, "App")\n```\n\n'
        "## Feature Map\n\n"
        "| Capability | Component | Interface | Evidence |\n"
        "| --- | --- | --- | --- |\n"
        "| Core | App | CLI | tests/test_core.py |\n",
        encoding="utf-8",
    )
    with pytest.raises(ArchitectureMapContractError, match="Change Log"):
        validate_architecture_map(doc)
