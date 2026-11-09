"""Citation contract for the SPN-0b primary-data extraction (#12737).

Every journal article in the Source Index of
``docs/research/segmental_spine/data_extraction.md`` carries a DOI, and no
author list is left as recalled from memory. Book chapters (``In:``) may
have none.
"""

from __future__ import annotations

import json
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
    truncated = [
        s for s, (citation, _) in _source_index().items() if "et al." in citation
    ]
    assert not truncated, f"author lists cut to 'et al.': {truncated}"


def _extraction_block(table_id: str) -> dict:
    text = EXTRACTION.read_text(encoding="utf-8")
    for raw in re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL):
        block = json.loads(raw)
        if block.get("table_id") == table_id:
            return block
    raise AssertionError(f"no JSON block for table_id={table_id!r}")


def test_bruno_b2_supplement_is_explicitly_unverified_with_empty_rows() -> None:
    """#12737: Bruno Supplemental Table 2 stays unverified until owner supplies PDF."""
    block = _extraction_block("B2_per_joint_coupling_supplemental_table_2")
    assert block["source_id"] == "bruno2015"
    assert block["verified_from_primary"] is False
    assert block["rows"] == []


def test_pearcy_p1_primary_table_stays_unverified_with_empty_rows() -> None:
    """#12737: Pearcy 1984 per-level table is not transcribed without primary access."""
    block = _extraction_block("P1_axial_rotation_coupled_bend")
    assert block["source_id"] == "pearcy1984"
    assert block["verified_from_primary"] is False
    assert block["rows"] == []


def test_hosea_g1_loads_are_secondary_citation_only() -> None:
    """#12737: Hosea 1990 numbers stay flagged until the chapter is read from primary."""
    block = _extraction_block("G1_L3L4_loads")
    assert block["source_id"] == "hosea1990"
    assert block["verified_from_primary"] is False
    assert block["rows"], "secondary-citation rows must remain explicit, not primary-verified"
