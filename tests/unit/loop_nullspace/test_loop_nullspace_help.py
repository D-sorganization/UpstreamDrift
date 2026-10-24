"""NSX-45 help surface gate: loop null-space interpretation doc exists and is wired."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
HELP = ROOT / "docs/help/loop_nullspace.md"


def test_loop_nullspace_help_page_exists_with_title_case() -> None:
    text = HELP.read_text(encoding="utf-8")
    assert "title: Loop Null-Space Exploration" in text
    assert "## Non-Claims" in text
    assert "family-dependent" in text.lower() or "Family-dependent" in text
