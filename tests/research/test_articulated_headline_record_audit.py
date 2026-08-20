"""Contracts for fail-closed articulated headline-record snapshot audits."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_headline_record_audit import (
    audit_headline_record,
)
from scripts.research.proximal_distal_energy.articulated_headline_uncertainty import (
    HeadlineUncertaintyConfig,
    ROOT,
    SOURCE_PATHS,
    registered_corners,
)

pytestmark = pytest.mark.scientific


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _pathway(status: str = "completed", matched: int = 1) -> dict:
    sources = _source_hashes()
    if status == "not_affected":
        return {
            "status": status,
            "matched_cell_count": None,
            "total_cell_count": 384,
            "all_registered_gates_passed": None,
        }
    if status == "failed_retained":
        return {
            "status": status,
            "failure_class": "ExpectedFailure",
            "failure_message": "retained adverse case",
            "matched_cell_count": None,
            "total_cell_count": 384,
            "all_registered_gates_passed": False,
            "computed_source_sha256": sources,
        }
    return {
        "status": status,
        "failure_class": None,
        "failure_message": None,
        "matched_cell_count": matched,
        "total_cell_count": 384,
        "all_registered_gates_passed": True,
        "computed_source_sha256": sources,
    }


def _record(*, row_count: int = 2, partial_last: bool = True) -> dict:
    config = HeadlineUncertaintyConfig(worker_count=2)
    corners = registered_corners(config)
    rows = []
    for index, corner in enumerate(corners[:row_count]):
        row = json.loads(json.dumps(asdict(corner)))
        row["shaft"] = (
            _pathway(matched=126 if corner.corner_id == "nominal" else 1)
            if "shaft" in corner.pathways or corner.corner_id == "nominal"
            else _pathway("not_affected")
        )
        if not (partial_last and index == row_count - 1):
            row["ground"] = _pathway(matched=0)
        rows.append(row)
    return {
        "schema_version": "articulated-headline-uncertainty/v1",
        "study_id": "articulated-shaft-ground-headline-uncertainty",
        "status": "in_progress",
        "design": {
            "method": "registered_nominal_plus_one_at_a_time_low_high_corners",
            "corner_count": len(corners),
            "axes": json.loads(json.dumps([asdict(axis) for axis in config.axes])),
            "controls": (
                "each full atlas retains both engines, velocity reversal, "
                "timestep refinement, and pathway killswitches"
            ),
        },
        "configuration": json.loads(json.dumps(asdict(config))),
        "corners": rows,
        "results": None,
        "source_sha256": _source_hashes(),
        "limitations": {
            "interaction_order": (
                "one-at-a-time corners do not estimate higher-order parameter "
                "interactions"
            ),
            "calibration": (
                "bounds are engineering ranges, not measured participant or "
                "equipment properties"
            ),
            "human_inference": (
                "survival does not promote any result to a human or coaching claim"
            ),
        },
    }


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _complete(record: dict) -> None:
    record["results"] = {}
    for pathway in ("shaft", "ground"):
        nominal = record["corners"][0][pathway]["matched_cell_count"]
        completed = [
            row for row in record["corners"] if row[pathway]["status"] == "completed"
        ]
        failed = [
            row
            for row in record["corners"]
            if row[pathway]["status"] == "failed_retained"
        ]
        for row in record["corners"]:
            count = row[pathway]["matched_cell_count"]
            row[pathway]["matched_cell_count_change_from_nominal"] = (
                count - nominal if count is not None else None
            )
        counts = [row[pathway]["matched_cell_count"] for row in completed]
        changes = [count - nominal for count in counts]
        record["results"][pathway] = {
            "nominal_matched_cell_count": nominal,
            "evaluated_corner_count": len(completed) + len(failed),
            "completed_corner_count": len(completed),
            "failed_corner_count": len(failed),
            "matched_cell_count_range": [min(counts), max(counts)],
            "matched_cell_count_change_range": [min(changes), max(changes)],
            "nonzero_change_corner_ids": [
                row["corner_id"]
                for row in completed
                if row[pathway]["matched_cell_count"] != nominal
            ],
            "failed_corner_ids": [row["corner_id"] for row in failed],
        }
    record["status"] = "complete"


def test_record_audit_accepts_a_valid_partial_prefix(tmp_path) -> None:
    path = tmp_path / "record.json"
    _write(path, _record(row_count=3))

    report = audit_headline_record(path)

    assert report["status"] == "partial"
    assert report["corner_count"] == 3
    assert report["expected_corner_count"] == 19
    assert report["fully_accounted_corner_count"] == 2
    assert report["terminal_pathway_count"] == 5
    assert report["active_corner_id"] == "grip_stiffness_scale-high"
    assert report["release_evidence"] is False
    assert len(report["record_sha256"]) == 64
    assert len(report["source_set_sha256"]) == 64


def test_record_audit_accepts_only_a_fully_reconciled_complete_record(tmp_path) -> None:
    path = tmp_path / "record.json"
    record = _record(row_count=19, partial_last=False)
    _complete(record)
    _write(path, record)

    report = audit_headline_record(path)

    assert report["status"] == "complete"
    assert report["fully_accounted_corner_count"] == 19
    assert report["terminal_pathway_count"] == 38
    assert report["active_corner_id"] is None
    assert report["release_evidence"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema"),
        ("study", "study identity"),
        ("status", "status"),
        ("design", "design drifted"),
        ("configuration", "configuration drifted"),
        ("limitations", "limitations drifted"),
    ],
)
def test_record_audit_rejects_top_level_governance_drift(
    tmp_path, mutation, message
) -> None:
    path = tmp_path / "record.json"
    record = _record()
    if mutation == "schema":
        record["schema_version"] = "unsupported"
    elif mutation == "study":
        record["study_id"] = "another-study"
    elif mutation == "status":
        record["status"] = "successful"
    elif mutation == "design":
        record["design"]["corner_count"] = 18
    elif mutation == "configuration":
        record["configuration"]["axes"][0]["high"] = 1.5
    else:
        record["limitations"]["human_inference"] = "removed"
    _write(path, record)

    with pytest.raises(RuntimeError, match=message):
        audit_headline_record(path)


@pytest.mark.parametrize("mutation", ["reorder", "duplicate"])
def test_record_audit_rejects_nonprefix_corner_identity(tmp_path, mutation) -> None:
    path = tmp_path / "record.json"
    record = _record(row_count=3, partial_last=False)
    if mutation == "reorder":
        record["corners"][1], record["corners"][2] = (
            record["corners"][2],
            record["corners"][1],
        )
    else:
        record["corners"][2] = record["corners"][1]
    _write(path, record)

    with pytest.raises(RuntimeError, match="registered prefix"):
        audit_headline_record(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source", "source hashes"),
        ("computed_source", "computed source hashes"),
        ("gate", "completed pathway"),
        ("count", "matched cell count"),
    ],
)
def test_record_audit_rejects_completed_pathway_drift(
    tmp_path, mutation, message
) -> None:
    path = tmp_path / "record.json"
    record = _record(row_count=2)
    if mutation == "source":
        key = next(iter(record["source_sha256"]))
        record["source_sha256"][key] = "0" * 64
    elif mutation == "computed_source":
        record["corners"][0]["shaft"]["computed_source_sha256"] = {}
    elif mutation == "gate":
        record["corners"][0]["shaft"]["all_registered_gates_passed"] = False
    else:
        record["corners"][0]["shaft"]["matched_cell_count"] = 385
    _write(path, record)

    with pytest.raises(RuntimeError, match=message):
        audit_headline_record(path)


def test_record_audit_rejects_premature_complete_status(tmp_path) -> None:
    path = tmp_path / "record.json"
    record = _record(row_count=2, partial_last=False)
    record["status"] = "complete"
    record["results"] = {"shaft": {}, "ground": {}}
    _write(path, record)

    with pytest.raises(RuntimeError, match="prematurely complete"):
        audit_headline_record(path)


def test_record_audit_restricts_not_affected_to_registered_pathways(tmp_path) -> None:
    path = tmp_path / "record.json"
    record = _record(row_count=2)
    record["corners"][0]["shaft"] = _pathway("not_affected")
    _write(path, record)

    with pytest.raises(RuntimeError, match="not_affected"):
        audit_headline_record(path)


def test_record_audit_accepts_registered_not_affected_pathway(tmp_path) -> None:
    path = tmp_path / "record.json"
    record = _record(row_count=12, partial_last=False)
    corner = record["corners"][-1]
    assert corner["corner_id"] == "ground_translation_stiffness_scale-low"
    corner["shaft"] = _pathway("not_affected")
    _write(path, record)

    report = audit_headline_record(path)

    assert report["status"] == "partial"
    assert report["fully_accounted_corner_count"] == 12
    assert report["active_corner_id"] is None


def test_record_audit_rejects_torn_json(tmp_path) -> None:
    path = tmp_path / "record.json"
    path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(RuntimeError, match="valid JSON snapshot"):
        audit_headline_record(path)
