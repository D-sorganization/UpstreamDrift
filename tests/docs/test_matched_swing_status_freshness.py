"""Tests for matched swing program documentation freshness and stale doc banners (MS-06 #10327).

Enforces:
1. The generated status matrix in docs/development/matched_swing_program/README.md
   matches the rendering from reports/matched_swing_ledger.json byte-for-byte.
2. Every superseded legacy specification or status document carries a dated
   SUPERSEDED banner pointing to matched_swing_program.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_matched_swing_status import (
    FENCE_END,
    FENCE_START,
    render_matched_swing_status,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_PATH = (
    REPO_ROOT / "docs" / "development" / "matched_swing_program" / "README.md"
)
LEDGER_PATH = REPO_ROOT / "reports" / "matched_swing_ledger.json"

STALE_DOCS_TO_CHECK = (
    REPO_ROOT / "docs" / "assessments" / "FEATURE_ENGINE_MATRIX.md",
    REPO_ROOT / "src" / "engines" / "CROSS_ENGINE_PARITY_SPEC.md",
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "OPENSIM_PARITY_SPEC.md",
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "myosuite"
    / "python"
    / "motion_matching"
    / "AUDIT.md",
    REPO_ROOT / "src" / "config" / "models.yaml",
)


def _normalize_markdown(text: str) -> str:
    """Normalize markdown table padding and trailing whitespace for comparison."""
    lines = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(c.replace("-", "").replace(":", "") == "" for c in cells if c):
                lines.append("|" + "|".join("---" for _ in cells) + "|")
            else:
                lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append(line)
    return "\n".join(lines)


def test_matched_swing_status_section_is_fresh() -> None:
    """The generated status section in README.md matches the ledger rendering."""
    if not LEDGER_PATH.is_file():
        pytest.skip(
            f"Ledger file not yet present at {LEDGER_PATH} (pending MS-02 #10323 landing)."
        )

    tracker_text = TRACKER_PATH.read_text(encoding="utf-8")
    assert FENCE_START in tracker_text, f"Missing {FENCE_START} in {TRACKER_PATH}"
    assert FENCE_END in tracker_text, f"Missing {FENCE_END} in {TRACKER_PATH}"

    ledger_data = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    expected_content = render_matched_swing_status(ledger_data)

    start_idx = tracker_text.index(FENCE_START) + len(FENCE_START)
    end_idx = tracker_text.index(FENCE_END)
    actual_content = tracker_text[start_idx:end_idx].strip()

    assert _normalize_markdown(actual_content) == _normalize_markdown(
        expected_content
    ), (
        f"{TRACKER_PATH} status section is stale compared to {LEDGER_PATH}.\n"
        "Run `python scripts/generate_matched_swing_status.py --write` to update."
    )


@pytest.mark.parametrize("doc_path", STALE_DOCS_TO_CHECK)
def test_stale_documents_carry_superseded_banner(doc_path: Path) -> None:
    """Every retired engine claim or greenfield parity doc carries a SUPERSEDED banner."""
    assert doc_path.is_file(), f"Target document not found at {doc_path}"
    content = doc_path.read_text(encoding="utf-8")
    assert "SUPERSEDED" in content, (
        f"{doc_path} is missing required 'SUPERSEDED' banner pointing to matched_swing_program."
    )
    assert "matched_swing_program" in content, (
        f"{doc_path} banner must link or reference 'matched_swing_program'."
    )
