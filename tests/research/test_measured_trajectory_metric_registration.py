from __future__ import annotations

import copy
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.proximal_distal_energy.measured_trajectory_metric_registration import (
    REQUIRED_METRIC_IDS,
    REQUIRED_NEGATIVE_CONTROL_IDS,
    load_and_validate_registration,
    validate_participant_split,
    validate_registration,
)


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific
REGISTRATION = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "measured_trajectory_metric_registration.json"
)


def _record() -> dict[str, object]:
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


def test_committed_registration_freezes_every_issue_9004_primary_metric() -> None:
    readiness = load_and_validate_registration(REGISTRATION)
    record = _record()

    assert tuple(row["metric_id"] for row in record["metrics"]) == REQUIRED_METRIC_IDS
    assert readiness["metric_count"] == len(REQUIRED_METRIC_IDS)
    assert readiness["status"] == "blocked_no_qualified_measured_trajectory_authority"
    assert record["registered_before_outcomes"] is True
    assert record["results_status"] == "not_run_no_authority"
    assert readiness["human_inference_ready"] is False
    assert readiness["bilateral_wrench_gate_satisfied"] is False


def test_negative_controls_are_complete_and_only_qualify_discrimination() -> None:
    record = _record()

    assert tuple(row["control_id"] for row in record["negative_controls"]) == (
        REQUIRED_NEGATIVE_CONTROL_IDS
    )
    assert all(
        row["evidence_class"] == "software_discrimination_only"
        for row in record["negative_controls"]
    )
    assert all(row["must_reject_metrics"] for row in record["negative_controls"])


def test_participant_split_rejects_framewise_or_overlapping_assignments() -> None:
    record = _record()
    record["participant_split"]["unit"] = "frame"

    with pytest.raises(ValueError, match="participant"):
        validate_registration(record)

    with pytest.raises(ValueError, match="disjoint"):
        validate_participant_split(("p01", "p02"), ("p02", "p03"))


def test_participant_split_accepts_disjoint_grouped_cohorts() -> None:
    result = validate_participant_split(
        ("p01", "p02", "p03"), ("p04",), adverse=("p05",)
    )

    assert result == {"training": 3, "held_out": 1, "adverse": 1}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["metrics"].pop(), "metric"),
        (
            lambda row: row["metrics"][0].update(
                {"threshold_policy": "select_after_held_out_inspection"}
            ),
            "threshold",
        ),
        (
            lambda row: row["metrics"][0]["required_channels"].append(
                "missing_as_zero"
            ),
            "missing",
        ),
        (lambda row: row.update({"results_status": "qualified"}), "results_status"),
    ],
)
def test_registration_fails_closed_on_scope_or_outcome_drift(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    record = _record()
    mutation(record)

    with pytest.raises(ValueError, match=message):
        validate_registration(record)


def test_readiness_is_recomputed_instead_of_trusted() -> None:
    record = _record()
    tampered = copy.deepcopy(record)
    tampered["readiness"]["metric_count"] += 1

    with pytest.raises(ValueError, match="readiness"):
        validate_registration(tampered)


def test_motion_registration_does_not_claim_force_or_coaching_authority() -> None:
    record = _record()
    text = json.dumps(record).lower()

    assert "bilateral wrench" in text
    assert "cannot" in record["inference_boundary"].lower()
    assert "coaching" in record["inference_boundary"].lower()
    assert all(
        "bilateral_wrench" not in row["required_channels"] for row in record["metrics"]
    )
