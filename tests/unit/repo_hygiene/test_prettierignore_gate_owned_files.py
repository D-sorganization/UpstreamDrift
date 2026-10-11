"""Files owned by a byte-exact or byte-budget gate stay out of Prettier.

Issue: #11777. Prettier rewrites the generated matched-swing ledger (float
exponents) and pads the DESIGN_DECISIONS.md tables past the doc size budget, so
a commit touching either file could not pass both the hook and CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRETTIERIGNORE = _REPO_ROOT / ".prettierignore"

_GATE_OWNED = (
    # Byte-exact: tests/unit/motion_matching/test_ledger.py::test_ledger_freshness.
    "reports/matched_swing_ledger.json",
    # Byte budget: scripts/check_doc_size_budget.py (51200 bytes, unchanged).
    "docs/development/full_body_models/DESIGN_DECISIONS.md",
)


def _prettierignore_entries() -> set[str]:
    return {
        line.strip()
        for line in _PRETTIERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


@pytest.mark.parametrize("path", _GATE_OWNED)
def test_gate_owned_file_is_prettier_ignored(path: str) -> None:
    """Each gate-owned file is listed verbatim and exists in the repository."""
    assert (_REPO_ROOT / path).is_file(), f"{path} no longer exists"
    assert path in _prettierignore_entries()
