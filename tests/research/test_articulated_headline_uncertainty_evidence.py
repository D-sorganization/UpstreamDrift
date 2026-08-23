"""Evidence gates for the committed articulated headline uncertainty record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.research.proximal_distal_energy.articulated_headline_record_audit import (
    audit_headline_record,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
RECORD = DATA / "articulated_headline_uncertainty.json"
PROVENANCE = DATA / "articulated_headline_execution_provenance.json"
pytestmark = pytest.mark.scientific


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _provenance() -> dict[str, object]:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def test_headline_uncertainty_record_is_complete_and_registered() -> None:
    record = _record()

    assert record["schema_version"] == "articulated-headline-uncertainty/v1"
    assert record["status"] == "complete"
    assert record["design"]["method"] == (
        "registered_nominal_plus_one_at_a_time_low_high_corners"
    )
    assert record["design"]["corner_count"] == 19
    assert len(record["design"]["axes"]) == 9
    assert len(record["corners"]) == 19
    expected = {"nominal"}
    for axis in record["design"]["axes"]:
        assert axis["low"] < axis["nominal"] < axis["high"]
        expected.update({f"{axis['name']}-low", f"{axis['name']}-high"})
    assert {corner["corner_id"] for corner in record["corners"]} == expected


def test_headline_counts_reproduce_nominal_and_retain_all_corners() -> None:
    record = _record()
    nominal = next(
        corner for corner in record["corners"] if corner["corner_id"] == "nominal"
    )

    assert nominal["shaft"]["matched_cell_count"] == 126
    assert nominal["ground"]["matched_cell_count"] == 0
    for pathway in ("shaft", "ground"):
        baseline = nominal[pathway]["matched_cell_count"]
        for corner in record["corners"]:
            result = corner[pathway]
            assert result["status"] in {
                "completed",
                "failed_retained",
                "not_affected",
            }
            if result["status"] == "completed":
                assert result["total_cell_count"] == 384
                assert result["all_registered_gates_passed"] is True
                assert result["matched_cell_count_change_from_nominal"] == (
                    result["matched_cell_count"] - baseline
                )
            else:
                assert result["matched_cell_count"] is None
                assert result["matched_cell_count_change_from_nominal"] is None


def test_headline_record_retains_scientific_inference_boundaries() -> None:
    limitations = _record()["limitations"]

    assert "do not estimate" in limitations["interaction_order"]
    assert "not measured participant" in limitations["calibration"]
    assert "does not promote" in limitations["human_inference"]


def test_headline_summary_and_source_hashes_match_authoritative_rows() -> None:
    record = _record()
    provenance = _provenance()

    for relative, expected in record["source_sha256"].items():
        source_blob = subprocess.run(
            ["git", "show", f"{provenance['source_revision']}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        observed = hashlib.sha256(source_blob).hexdigest()
        assert observed == expected, relative
    for corner in record["corners"]:
        for pathway in ("shaft", "ground"):
            result = corner[pathway]
            if result["status"] == "not_affected":
                assert "computed_source_sha256" not in result
                continue
            computed = result["computed_source_sha256"]
            assert computed
            assert all(len(digest) == 64 for digest in computed.values())
    for pathway in ("shaft", "ground"):
        completed = [
            corner
            for corner in record["corners"]
            if corner[pathway]["status"] == "completed"
        ]
        failed = [
            corner
            for corner in record["corners"]
            if corner[pathway]["status"] == "failed_retained"
        ]
        summary = record["results"][pathway]
        counts = [corner[pathway]["matched_cell_count"] for corner in completed]
        assert summary["completed_corner_count"] == len(completed)
        assert summary["failed_corner_count"] == len(failed)
        assert summary["matched_cell_count_range"] == [min(counts), max(counts)]
        assert summary["failed_corner_ids"] == [
            corner["corner_id"] for corner in failed
        ]


def test_headline_execution_provenance_binds_terminal_transfer() -> None:
    provenance = _provenance()

    assert provenance["schema_version"] == (
        "articulated-headline-execution-provenance/v1"
    )
    assert len(provenance["source_revision"]) == 40
    assert (
        provenance["record_sha256"] == hashlib.sha256(RECORD.read_bytes()).hexdigest()
    )
    assert provenance["source_sha256"] == _record()["source_sha256"]
    assert (
        audit_headline_record(
            RECORD,
            expected_sources=provenance["source_sha256"],
        )["status"]
        == "complete"
    )
    assert provenance["terminal_state"] == {"state": "complete", "exit_code": 0}
    assert provenance["checkpoint_file_count"] == 1327
    assert provenance["transfer_file_count"] == 1329
    assert provenance["stale_snapshot_promoted"] is False
