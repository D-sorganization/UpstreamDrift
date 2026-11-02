"""Committed HRG-D3 (#13174) trunk still evidence (receipt + manifest contract)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/trunk_visuals"
RECEIPT = EVIDENCE / "receipt.json"
EVENTS = ("address", "top", "impact")
VIEWS = ("face_on", "down_the_line")


def test_trunk_visuals_evidence_receipt_lists_three_events_and_hashes() -> None:
    """The GL still receipt records address, top and impact PNGs with spec sha."""
    doc = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert doc["issue"] == 13174
    assert len(doc["spec_sha256"]) == 64
    assert doc["event_frames"]["address"] == 0
    assert doc["event_frames"]["top"] < doc["event_frames"]["impact"]

    manifest = json.loads((EVIDENCE / doc["manifest"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "v2-trunk-stills-manifest/v1"
    assert doc["command"] == manifest["command"]
    assert {row["event"] for row in manifest["stills"]} == set(EVENTS)

    for event in EVENTS:
        for view in VIEWS:
            rel = next(
                row["file"]
                for row in manifest["stills"]
                if row["event"] == event and row["view"] == view
            )
            path = EVIDENCE / "render_out" / rel
            assert path.is_file(), rel
            assert (
                doc["stills_sha256"][event][view]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )
            assert str(doc["spec_sha256"][:12]) in path.name
