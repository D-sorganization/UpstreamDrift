"""Per-club clubhead marker clusters in the anthropometric documents (#12210)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.execution import assets
from src.shared.python.motion_matching.execution.club_markers import (
    CLUB_MARKER_BODY,
    apply_club_markers,
    derive_club_markers,
    load_club_markers,
)
from src.shared.python.motion_matching.execution.spec_builder import marker_seeds

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS = REPO_ROOT / "docs" / "development" / "full_body_models"
CANDIDATE = DOCS / "evidence" / "native_candidates" / "returned81_candidate.json"
IRON_MARKERS = DOCS / "evidence" / "club_markers" / "iron7.json"
DRIVER_MARKERS = DOCS / "evidence" / "club_markers" / "driver.json"
HEAD_TRIAD = ("Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3")
CLUB_FILES = {
    "iron7": (IRON_MARKERS, "full_body_spec_anthro_iron7.json", "iron"),
    "driver": (DRIVER_MARKERS, "full_body_spec_anthro_driver.json", "driver"),
}


def _club_labels(attachments: dict) -> set[str]:
    return {k for k, v in attachments.items() if v["body"] == CLUB_MARKER_BODY}


def _edges_m(offsets: dict) -> list[float]:
    p = [np.asarray(offsets[label], dtype=float) for label in HEAD_TRIAD]
    return sorted(
        float(np.linalg.norm(p[i] - p[j])) for i, j in ((0, 1), (1, 2), (0, 2))
    )


def _document(name: str) -> dict:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("club", sorted(CLUB_FILES))
def test_committed_document_carries_its_club_cluster(club: str) -> None:
    markers_path, document_name, _ = CLUB_FILES[club]
    markers = load_club_markers(markers_path)
    attachments = _document(document_name)["marker_attachments"]
    assert _club_labels(attachments) == set(markers.offsets_m)
    for label, offset in markers.offsets_m.items():
        assert attachments[label]["offset_m"] == pytest.approx(list(offset), abs=0)


def test_iron_head_triad_is_not_the_driver_triad() -> None:
    driver = _document("full_body_spec_anthro_driver.json")["marker_attachments"]
    iron = _document("full_body_spec_anthro_iron7.json")["marker_attachments"]
    driver_edges = _edges_m({k: driver[k]["offset_m"] for k in HEAD_TRIAD})
    iron_edges = _edges_m({k: iron[k]["offset_m"] for k in HEAD_TRIAD})
    # Shapes differ by tens of millimetres (#12210: 52/92/96 vs 40/106/128 mm).
    assert max(abs(a - b) for a, b in zip(driver_edges, iron_edges, strict=True)) > 0.02


def test_driver_cluster_is_the_native_head_triad_re_placed_rigidly() -> None:
    # #12221: the static calibration keeps the native triad's shape and only
    # moves it on the club, so the measured cluster is the same head.
    native = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    native_offsets = {
        label: offset
        for label, body, offset in zip(
            native["marker_labels"],
            native["marker_bodies"],
            native["marker_offsets_m"],
            strict=True,
        )
        if body == CLUB_MARKER_BODY
    }
    driver = load_club_markers(DRIVER_MARKERS)
    assert set(driver.offsets_m) == set(native_offsets)
    assert driver.head_triad_edges_m == pytest.approx(
        _edges_m(native_offsets), abs=5e-4
    )
    moved = max(
        float(np.linalg.norm(np.subtract(driver.offsets_m[k], native_offsets[k])))
        for k in HEAD_TRIAD
    )
    assert moved > 0.01


@pytest.mark.parametrize("club", sorted(CLUB_FILES))
def test_cluster_records_its_measured_source(club: str) -> None:
    markers_path, _, capture = CLUB_FILES[club]
    markers = load_club_markers(markers_path)
    assert markers.club == club
    source = markers.source
    receipt = REPO_ROOT / source["receipt"]
    assert receipt.is_file()
    assert source["method"].startswith("static-seed calibration")
    assert len(source["receipt_sha256"]) == 64
    # The receipt is regenerated on the new document, so the source is pinned
    # to the commit the calibrated offsets were read at.
    assert len(source["receipt_commit"]) == 40
    assert source["capture"] == capture
    assert markers.head_triad_edges_m == pytest.approx(
        _edges_m(markers.offsets_m), abs=1e-9
    )


def test_derive_reads_the_calibrated_club_attachments(tmp_path: Path) -> None:
    receipt = {
        "capture": "iron",
        "capture_sha256": "c" * 64,
        "ik": {
            "attachments_m": {
                "Marker_2:2:1": {"body": "Clubhead", "offset_m": [0.0, 0.0, 0.0]},
                "Marker_2:2:2": {"body": "Clubhead", "offset_m": [0.1, 0.0, 0.0]},
                "Marker_2:2:3": {"body": "Clubhead", "offset_m": [0.0, 0.1, 0.0]},
                "HeadTop": {"body": "Head", "offset_m": [0.0, 0.0, 0.2]},
            }
        },
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    derived = derive_club_markers(path, club="iron7", receipt_label="r.json")
    assert set(derived.offsets_m) == set(HEAD_TRIAD)
    assert derived.source["receipt"] == "r.json"
    assert derived.source["capture_sha256"] == "c" * 64
    assert derived.head_triad_edges_m == pytest.approx(
        sorted([0.1, 0.1, 0.1 * np.sqrt(2)])
    )


def test_derive_rejects_a_receipt_without_club_attachments(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"capture": "iron", "ik": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="attachments_m"):
        derive_club_markers(path, club="iron7", receipt_label="r.json")


def test_apply_is_idempotent_and_reports_changes() -> None:
    document = _document("full_body_spec_anthro_driver.json")
    iron = load_club_markers(IRON_MARKERS)
    assert apply_club_markers(document, iron.offsets_m) is True
    assert apply_club_markers(document, iron.offsets_m) is False
    for label, offset in iron.offsets_m.items():
        assert document["marker_attachments"][label]["offset_m"] == list(offset)


def test_apply_rejects_a_label_that_is_not_on_the_clubhead() -> None:
    document = _document("full_body_spec_anthro_driver.json")
    with pytest.raises(ValueError, match="HeadTop"):
        apply_club_markers(document, {"HeadTop": (0.0, 0.0, 0.0)})


def test_marker_seeds_override_the_native_club_cluster() -> None:
    native = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    iron = load_club_markers(IRON_MARKERS)
    seeds = marker_seeds(0.42, 0.18, 0.3, 0.27, native, club_offsets=iron.offsets_m)
    for label, offset in iron.offsets_m.items():
        assert seeds[label].body == CLUB_MARKER_BODY
        assert seeds[label].offset_m == pytest.approx(offset)


def test_marker_seeds_reject_a_partial_club_cluster() -> None:
    native = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="club marker labels"):
        marker_seeds(
            0.42, 0.18, 0.3, 0.27, native, club_offsets={"Marker_2:2:1": (0, 0, 0)}
        )


def test_assets_resolve_a_cluster_per_club() -> None:
    assert assets.get_club_markers("iron7") == IRON_MARKERS
    # #12221: the driver carries its static-calibrated cluster, not the native one.
    assert assets.get_club_markers("driver") == DRIVER_MARKERS
    assert assets.get_club_markers("wedge") is None
    with pytest.raises(FileNotFoundError):
        assets.get_club_markers("iron7", REPO_ROOT / "missing.json")
