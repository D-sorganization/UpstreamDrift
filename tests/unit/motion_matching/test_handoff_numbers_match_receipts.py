"""Tests asserting that metrics quoted in HANDOFF.md match cited receipt fields (MS-03 #10324).

Enforces:
1. Every numerical metric (in mm) inside HANDOFF.md "Status and Next Steps" and
   "Two Implementations, One Program" must cite an explicit receipt path and field:
   ``<receipt_path>.json#<field.path>``.
2. The quoted value in mm must equal the cited JSON field value within 0.05 mm.
3. Every cited receipt path must exist on disk and be valid JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, NamedTuple

import pytest

from src.shared.python.contracts import postcondition, precondition

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
HANDOFF_PATH = REPO_ROOT / "docs" / "development" / "full_body_models" / "HANDOFF.md"

# Regex to match: <value> mm ... (<receipt_path>#<field_path>) or `<receipt_path>#<field_path>`
CITATION_PATTERN = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*mm\s*(?:[^\n\d]*?)?[`\(](?P<path>[^\s`\(\)#]+?\.json)#(?P<field>[a-zA-Z0-9_\.]+)[`\)]"
)

# Regex to detect lines mentioning mm metrics
MM_METRIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*mm\b")


class MetricCitation(NamedTuple):
    line_number: int
    quoted_value_mm: float
    receipt_relpath: str
    field_path: str
    raw_line: str


@precondition(lambda text: isinstance(text, str) and len(text) > 0)
@postcondition(lambda result: isinstance(result, list))
def extract_metric_citations(
    text: str, start_line_offset: int = 1
) -> list[MetricCitation]:
    """Extract all metric citations from markdown text with explicit receipt links.

    DbC:
    - Precondition: `text` is a non-empty string.
    - Postcondition: Returns a list of MetricCitation instances.
    """
    citations: list[MetricCitation] = []
    for idx, line in enumerate(text.splitlines(), start=start_line_offset):
        for m in CITATION_PATTERN.finditer(line):
            citations.append(
                MetricCitation(
                    line_number=idx,
                    quoted_value_mm=float(m.group("val")),
                    receipt_relpath=m.group("path"),
                    field_path=m.group("field"),
                    raw_line=line.strip(),
                )
            )
    return citations


@precondition(
    lambda data, field_path: (
        isinstance(data, dict) and isinstance(field_path, str) and len(field_path) > 0
    )
)
def resolve_json_field(data: dict[str, Any], field_path: str) -> float:
    """Resolve a dot-separated field path in a nested dictionary and return as float.

    DbC:
    - Precondition: `data` is a dict, `field_path` is a non-empty string.
    - Postcondition: Returns float value or raises KeyError/ValueError.
    """
    curr: Any = data
    parts = field_path.split(".")
    for part in parts:
        if not isinstance(curr, dict) or part not in curr:
            keys = list(curr.keys()) if isinstance(curr, dict) else type(curr)
            raise KeyError(
                f"Field '{part}' not found while resolving '{field_path}' in {keys}"
            )
        curr = curr[part]
    if curr is None:
        raise ValueError(f"Field '{field_path}' resolved to None")
    return float(curr)


def get_handoff_monitored_sections(content: str) -> list[tuple[int, str]]:
    """Extract monitored section line ranges and text from HANDOFF.md.

    Monitored sections:
    - "## Two Implementations, One Program"
    - "## Status and Next Steps"
    """
    lines = content.splitlines()
    monitored_lines: list[tuple[int, str]] = []
    in_section = False
    monitored_headers = (
        "## Two Implementations, One Program",
        "## Status and Next Steps",
    )

    for i, line in enumerate(lines, start=1):
        if line.startswith(monitored_headers):
            in_section = True
            monitored_lines.append((i, line))
            continue
        if line.startswith("## ") and in_section:
            in_section = False

        if in_section:
            monitored_lines.append((i, line))

    return monitored_lines


def test_handoff_has_no_uncited_mm_metrics_in_monitored_sections() -> None:
    """Every line in monitored sections containing 'mm' must cite an explicit receipt path and field."""
    assert HANDOFF_PATH.is_file(), f"HANDOFF.md not found at {HANDOFF_PATH}"
    content = HANDOFF_PATH.read_text(encoding="utf-8")
    monitored = get_handoff_monitored_sections(content)

    uncited_lines: list[tuple[int, str]] = []
    for line_no, line in monitored:
        if MM_METRIC_PATTERN.search(line):
            citations = extract_metric_citations(line, start_line_offset=line_no)
            if not citations:
                uncited_lines.append((line_no, line))

    assert not uncited_lines, (
        f"Found {len(uncited_lines)} lines with uncited mm metrics in monitored sections of HANDOFF.md:\n"
        + "\n".join(f"  Line {no}: {line_str}" for no, line_str in uncited_lines)
    )


def test_handoff_numbers_match_receipts() -> None:
    """All quoted mm metrics with receipt citations in HANDOFF.md match within 0.05 mm."""
    assert HANDOFF_PATH.is_file(), f"HANDOFF.md not found at {HANDOFF_PATH}"
    content = HANDOFF_PATH.read_text(encoding="utf-8")

    citations = extract_metric_citations(content)

    assert len(citations) > 0, "Expected at least one metric citation in HANDOFF.md"

    mismatches: list[str] = []
    for cit in citations:
        path = REPO_ROOT / cit.receipt_relpath
        if not path.is_file():
            path = (
                REPO_ROOT
                / "docs"
                / "development"
                / "full_body_models"
                / cit.receipt_relpath
            )
        if not path.is_file():
            mismatches.append(
                f"Line {cit.line_number}: Receipt file not found: {cit.receipt_relpath}"
            )
            continue

        receipt_data = json.loads(path.read_text(encoding="utf-8"))
        try:
            val_m = resolve_json_field(receipt_data, cit.field_path)
        except (KeyError, ValueError) as exc:
            mismatches.append(f"Line {cit.line_number}: {exc}")
            continue

        val_mm = val_m * 1000.0
        delta = abs(cit.quoted_value_mm - val_mm)
        if delta > 0.05:
            mismatches.append(
                f"Line {cit.line_number}: Quoted {cit.quoted_value_mm:.2f} mm != receipt "
                f"{val_mm:.2f} mm (delta={delta:.3f} mm > 0.05 mm) for {cit.receipt_relpath}#{cit.field_path}"
            )

    assert not mismatches, "Metric citation mismatches in HANDOFF.md:\n" + "\n".join(
        f"  - {m}" for m in mismatches
    )
