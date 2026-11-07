"""ADR-0061 (#13763) hand licensing decision is present, indexed and gated."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ADR_DIR = Path(__file__).resolve().parents[3] / "docs" / "adr"
_ADR = _ADR_DIR / "0061-hrg-hx6-hand-data-and-asset-licensing.md"


def test_adr_0061_records_decisions_and_sign_off_gate() -> None:
    text = _ADR.read_text(encoding="utf-8")
    assert "# ADR-0061:" in text
    assert "Related Issues/PRs: #13763" in text
    assert "Status: Proposed" in text
    assert "## Sign-Off Gate" in text
    assert "No MANO or SMPL-X meshes" in text


def test_adr_0061_is_indexed() -> None:
    index = (_ADR_DIR / "README.md").read_text(encoding="utf-8")
    assert "(0061-hrg-hx6-hand-data-and-asset-licensing.md)" in index
