from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.measured_trajectory_source_registry import (
    compute_readiness,
    load_and_validate_registry,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific
REGISTRY = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "measured_trajectory_source_registry.json"
)


def _record() -> dict[str, object]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_committed_registry_is_fail_closed_and_not_human_qualified() -> None:
    result = load_and_validate_registry(REGISTRY)

    assert result["status"] == "blocked_no_qualified_measured_trajectory_authority"
    assert result["source_count"] >= 5
    assert result["pipeline_probe_ready_source_ids"] == []
    assert result["held_out_qualification_ready_source_ids"] == []
    assert result["human_inference_ready"] is False
    assert result["bilateral_wrench_gate_satisfied"] is False


def test_simulation_output_cannot_be_promoted_as_human_measurement() -> None:
    record = _record()
    source = next(
        row for row in record["sources"] if row["source_id"] == "local-simscape-trials"
    )
    source["decision"] = "qualification_candidate"

    with pytest.raises(ValueError, match="simulation output cannot qualify"):
        validate_registry(record)


def test_missing_data_are_not_silently_treated_as_false_measurements() -> None:
    record = _record()
    source = record["sources"][0]
    source["club_kinematics"] = None
    source["blockers"] = []

    with pytest.raises(ValueError, match="unknown fields require a blocker"):
        validate_registry(record)


def test_participant_holdout_requires_at_least_two_grouped_participants() -> None:
    record = _record()
    source = next(
        row for row in record["sources"] if row["source_id"] == "kit-golf-drive-1319"
    )
    source.update(
        {
            "license_status": "explicit_reuse_license",
            "club_kinematics": True,
            "calibration_metadata_available": True,
            "synchronization_metadata_available": True,
            "content_digest_sha256": "a" * 64,
            "blockers": [],
        }
    )

    record["readiness"] = compute_readiness(record["sources"])
    result = validate_registry(record)

    assert result["pipeline_probe_ready_source_ids"] == ["kit-golf-drive-1319"]
    assert result["held_out_qualification_ready_source_ids"] == []


def test_authorized_multisubject_source_can_satisfy_motion_only_gate() -> None:
    record = _record()
    source = next(row for row in record["sources"] if row["source_id"] == "golfpose")
    source.update(
        {
            "access_status": "authorized_download",
            "license_status": "explicit_reuse_license",
            "participant_grouping_available": True,
            "calibration_metadata_available": True,
            "synchronization_metadata_available": True,
            "content_digest_sha256": "b" * 64,
            "blockers": [],
        }
    )

    record["readiness"] = compute_readiness(record["sources"])
    result = validate_registry(record)

    assert result["pipeline_probe_ready_source_ids"] == ["golfpose"]
    assert result["held_out_qualification_ready_source_ids"] == ["golfpose"]
    assert result["human_inference_ready"] is False
    assert result["bilateral_wrench_gate_satisfied"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("landing_url", "http://example.test/data", "HTTPS"),
        ("content_digest_sha256", "abc", "SHA-256"),
        ("participant_count", 0, "participant_count"),
        ("decision", "qualified", "decision"),
    ],
)
def test_registry_rejects_malformed_or_unregistered_values(
    field: str, value: object, message: str
) -> None:
    record = _record()
    record["sources"][0][field] = value

    with pytest.raises(ValueError, match=message):
        validate_registry(record)


def test_registry_summary_is_recomputed_not_trusted() -> None:
    record = _record()
    tampered = copy.deepcopy(record)
    tampered["readiness"]["source_count"] += 1

    with pytest.raises(ValueError, match="readiness summary"):
        validate_registry(tampered)
