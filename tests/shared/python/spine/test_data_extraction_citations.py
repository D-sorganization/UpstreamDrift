"""Citation contract for the SPN-0b primary-data extraction (#12737).

Every journal article in the Source Index of
``docs/research/segmental_spine/data_extraction.md`` carries a DOI, and no
author list is left as recalled from memory. Book chapters (``In:``) may
have none.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
EXTRACTION = REPO / "docs" / "research" / "segmental_spine" / "data_extraction.md"
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")


def _source_index() -> dict[str, tuple[str, str]]:
    """Return ``{source_id: (citation, doi)}`` from the Source Index table."""
    text = EXTRACTION.read_text(encoding="utf-8")
    section = text.split("## Source Index", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, tuple[str, str]] = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("source_id", "---"):
            continue
        rows[cells[0]] = (cells[1], cells[2])
    return rows


def test_source_index_lists_the_extracted_sources() -> None:
    assert {"bruno2015", "pearcy1984", "fujii2007", "wilke1999"} <= set(_source_index())


def test_every_journal_article_has_a_doi() -> None:
    missing = {
        source: doi
        for source, (citation, doi) in _source_index().items()
        if " In: " not in citation and not DOI_PATTERN.match(doi)
    }
    assert not missing, f"journal articles without a DOI: {missing}"


def test_no_author_list_is_from_memory() -> None:
    text = EXTRACTION.read_text(encoding="utf-8")
    assert "from\nmemory" not in text and "from memory" not in text
    assert "et al." not in text.split("## Source Index", 1)[1].split("\n## ", 1)[0]
