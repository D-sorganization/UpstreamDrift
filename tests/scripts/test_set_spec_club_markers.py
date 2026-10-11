"""The committed 7-iron document is current with its club marker file (#12210)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.set_spec_club_markers import set_document_club_markers

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "development" / "full_body_models"
IRON_MARKERS = DOCS / "evidence" / "club_markers" / "iron7.json"


def test_committed_iron_document_is_current(tmp_path: Path) -> None:
    copy = tmp_path / "iron7.json"
    shutil.copy(DOCS / "full_body_spec_anthro_iron7.json", copy)
    before = copy.read_bytes()
    assert set_document_club_markers(IRON_MARKERS, copy) is False
    assert copy.read_bytes() == before


def test_refuses_a_document_for_another_club(tmp_path: Path) -> None:
    copy = tmp_path / "driver.json"
    shutil.copy(DOCS / "full_body_spec_anthro_driver.json", copy)
    with pytest.raises(ValueError, match="driver document, not iron7"):
        set_document_club_markers(IRON_MARKERS, copy)
