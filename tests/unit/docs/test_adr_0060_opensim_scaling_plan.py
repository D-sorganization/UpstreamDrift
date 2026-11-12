"""ADR-0060 (#13991) plan document is present and traceable."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ADR = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "adr"
    / "0060-opensim-humanoid-scaling-and-fit-fidelity.md"
)


def test_adr_0060_exists() -> None:
    text = _ADR.read_text(encoding="utf-8")
    assert "# ADR-0060:" in text
    assert "Related Issues/PRs: #13991" in text
    assert "## Ordered TDD Slices" in text
