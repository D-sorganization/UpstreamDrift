"""OSV-10 (#11759): the club-face fixture regeneration script's pure parts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import regenerate_club_face_fixtures as regen
from src.shared.python.motion_matching.club_face_target import HEAD_TRIAD_LABELS

pytestmark = pytest.mark.unit
FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/club_face"


def test_pipeline_argv_keeps_the_original_run_flags_and_the_weight() -> None:
    argv = regen.pipeline_argv("iron", Path("runs/iron"), 3.0)
    assert argv[argv.index("--capture") + 1] == "iron"
    assert "--zmp-filter" in argv and "--static-seeds" in argv
    assert argv[argv.index("--face-weight") + 1] == "3.0"
    assert argv[argv.index("--spec") + 1].endswith("full_body_spec_anthro_iron7.json")
    assert "--zmp-filter" not in regen.pipeline_argv("driver", Path("d"), 0.0)
    with pytest.raises(ValueError, match="unknown capture"):
        regen.pipeline_argv("owner", Path("o"), 3.0)


def test_triad_offsets_must_sit_on_the_face_frame() -> None:
    labels = ("Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3")
    receipt = {
        "ik": {
            "attachments_m": {
                k: {"body": "Clubhead", "offset_m": [0.0, i, 0.0]}
                for i, k in enumerate(labels)
            }
        }
    }
    assert regen._triad_offsets(receipt)[labels[2]] == [0.0, 2, 0.0]  # noqa: SLF001
    receipt["ik"]["attachments_m"][labels[0]]["body"] = "Club"
    with pytest.raises(ValueError, match="not attached"):
        regen._triad_offsets(receipt)  # noqa: SLF001


def test_committed_provenance_matches_the_committed_fixtures() -> None:
    import hashlib

    doc = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    assert doc["generator"] == "scripts/regenerate_club_face_fixtures.py"
    for club, record in doc["clubs"].items():
        data = (FIXTURES / f"swing_q_{club}.npz").read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["fixture_sha256"]
        assert record["face_weight"] > 0.0
        assert regen.CAPTURE_RUNS[record["capture"]][0] == club
        assert tuple(record["capture_triad_offsets_m"]) == HEAD_TRIAD_LABELS
