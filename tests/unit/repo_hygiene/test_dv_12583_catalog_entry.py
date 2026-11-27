"""NSX-55 (#12583): deferred-validation record DV-12583 stays published in the catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "docs/development/planning/catalog.json"
RECORD = ROOT / "docs/development/planning/DV-12583.md"
SNAPSHOT = ROOT / "docs/development/planning/source-12583.json"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_dv_12583_planning_artifacts_exist() -> None:
    assert CATALOG.is_file(), f"missing catalog: {CATALOG}"
    assert RECORD.is_file(), f"missing deferred record: {RECORD}"
    assert SNAPSHOT.is_file(), f"missing GitHub snapshot: {SNAPSHOT}"


def test_dv_12583_catalog_entry_matches_repo_snapshot() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = next((e for e in catalog["entries"] if e.get("id") == "DV-12583"), None)
    assert entry is not None, "catalog.json must contain id DV-12583"
    assert entry.get("record") == "DV-12583.md"
    assert entry.get("source_snapshot") == "source-12583.json"
    assert entry.get("status") == "deferred"

    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert snapshot.get("number") == 12583
    assert "12583" in entry.get("source_issue", "")
