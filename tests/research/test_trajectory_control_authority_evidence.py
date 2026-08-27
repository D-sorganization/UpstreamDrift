"""Governed evidence contracts for trajectory control authority."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_trajectory_control_authority import (
    ADDITIVITY_RESIDUAL_GATE,
    ARRAY_PATH,
    DIRECT_PULSE_RESIDUAL_GATE,
    INTEGRATION_REFINEMENT_GATE,
    INPUT_REFINEMENT_GATE,
    REPORT_PATH,
    SOURCE_ARRAY_PATH,
    SOURCE_REPORT_PATH,
    build_report,
    reports_reproducibly_equivalent,
    validate_report,
)

pytestmark = pytest.mark.scientific
ROOT = REPORT_PATH.parents[4]
ARTICLE = REPORT_PATH.parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def registered_report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_registered_report_is_deterministic_and_retains_source_identity(
    registered_report: dict[str, Any],
) -> None:
    assert reports_reproducibly_equivalent(registered_report, build_report())
    assert registered_report["source_identity"] == {
        "phase_event_array_sha256": _sha256(SOURCE_ARRAY_PATH),
        "phase_event_report_schema": "proximal-distal-phase-event-stability/v1",
        "phase_event_report_sha256": _sha256(SOURCE_REPORT_PATH),
        "required_parent_pr": 9117,
        "required_parent_issue": 9116,
    }
    assert validate_report(registered_report) == {
        "channel_case_count": 4,
        "direct_pulse_count": 6,
        "frozen_window_count": 4,
        "integration_refinement_count": 3,
        "input_refinement_count": 3,
    }


def test_reproducibility_comparison_tolerates_only_sub_resolution_residue(
    registered_report: dict[str, Any],
) -> None:
    expected = deepcopy(registered_report)
    expected["falsification_controls"]["channel_additivity"][
        "maximum_abs_residual"
    ] *= 1.5
    assert reports_reproducibly_equivalent(registered_report, expected)

    changed_result = deepcopy(registered_report)
    changed_result["event_conditioned_authority"]["event_tangent"]["rank"] = 2
    assert not reports_reproducibly_equivalent(registered_report, changed_result)

    resolved_residue = deepcopy(registered_report)
    resolved_residue["falsification_controls"]["channel_additivity"][
        "maximum_abs_residual"
    ] = ADDITIVITY_RESIDUAL_GATE * 1e-1
    assert not reports_reproducibly_equivalent(registered_report, resolved_residue)


def test_registered_controls_pass_raw_falsification_gates(
    registered_report: dict[str, Any],
) -> None:
    controls = registered_report["falsification_controls"]

    assert controls["zero_input"]["maximum_abs_gramian_entry"] == 0.0
    assert controls["channel_additivity"]["maximum_abs_residual"] < (
        ADDITIVITY_RESIDUAL_GATE
    )
    assert (
        max(trial["maximum_abs_residual"] for trial in controls["direct_pulses"])
        < DIRECT_PULSE_RESIDUAL_GATE
    )
    assert (
        max(
            trial["relative_event_gramian_residual"]
            for trial in controls["input_step_refinement"]
        )
        < INPUT_REFINEMENT_GATE
    )
    assert (
        max(
            trial["relative_event_gramian_residual"]
            for trial in controls["integration_step_refinement"]
        )
        < INTEGRATION_REFINEMENT_GATE
    )
    assert controls["equivalent_units"]["maximum_abs_residual"] < 1e-12


def test_event_authority_uses_a_three_dimensional_tangent_basis(
    registered_report: dict[str, Any],
) -> None:
    event = registered_report["event_conditioned_authority"]

    assert event["status"] == "transverse"
    assert event["unique_crossing"] is True
    assert event["tangent_dimension"] == 3
    assert event["full_state_dimension"] == 4
    assert event["guard_normal_null_direction_is_actuator_loss"] is False
    assert event["guard_residual"] < 1e-10


def test_frozen_local_countermodel_remains_separate(
    registered_report: dict[str, Any],
) -> None:
    comparisons = registered_report["frozen_local_countermodel"]

    assert len(comparisons) == 4
    assert all(case["same_phase_and_horizon"] for case in comparisons)
    assert any(case["relative_gramian_difference"] > 1e-3 for case in comparisons)
    assert "countermodel" in registered_report["inference_boundary"].lower()


def test_npz_retains_full_precision_arrays() -> None:
    with np.load(ARRAY_PATH) as arrays:
        step_count = arrays["controls"].shape[0]
        assert arrays["state"].shape == (step_count + 1, 4)
        assert arrays["scaled_state_matrices"].shape == (step_count, 4, 4)
        assert arrays["scaled_energy_input_matrices"].shape == (step_count, 4, 2)
        assert arrays["full_gramian_history"].shape == (step_count + 1, 4, 4)
        assert arrays["event_tangent_basis"].shape == (4, 3)
        assert arrays["event_tangent_gramian"].shape == (3, 3)
        assert arrays["direct_pulse_predicted"].shape == (6, 4)
        assert arrays["direct_pulse_observed"].shape == (6, 4)


def test_byte_governed_output_is_excluded_from_prettier() -> None:
    precommit = (REPORT_PATH.parents[4] / ".pre-commit-config.yaml").read_text(
        encoding="utf-8"
    )

    assert "trajectory_control_authority" in precommit


def test_validation_rejects_lost_controls_and_inference_promotion(
    registered_report: dict[str, Any],
) -> None:
    lost_additivity = deepcopy(registered_report)
    lost_additivity["falsification_controls"]["channel_additivity"][
        "maximum_abs_residual"
    ] = ADDITIVITY_RESIDUAL_GATE
    with pytest.raises(ValueError, match="additivity"):
        validate_report(lost_additivity)

    lost_tangent = deepcopy(registered_report)
    lost_tangent["event_conditioned_authority"]["tangent_dimension"] = 4
    with pytest.raises(ValueError, match="tangent"):
        validate_report(lost_tangent)

    promoted = deepcopy(registered_report)
    promoted["inference_boundary"] = "This proves a universal coaching strategy."
    with pytest.raises(ValueError, match="inference boundary"):
        validate_report(promoted)


def test_release_claims_preserve_local_authority_boundary() -> None:
    registry = json.loads(
        (ARTICLE / "data/claim_audit_registry.json").read_text(encoding="utf-8")
    )
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}

    assert {"PD-CLAIM-317", "PD-CLAIM-318"} <= set(claims)
    for claim_id in ("PD-CLAIM-317", "PD-CLAIM-318"):
        claim = claims[claim_id]
        assert claim["published_status"] == (
            "supported_for_declared_local_first_order_analytical_trajectory"
        )
        adjudication = claim["adjudication"].lower()
        assert "bounded nonlinear reachability" in adjudication
        assert "human" in adjudication
        assert "coaching" in adjudication

    review = json.loads(
        (ARTICLE / "data/release_claim_review.json").read_text(encoding="utf-8")
    )
    release = next(
        item
        for item in review["release_claim_reviews"]
        if item["release_claim_key"] == "trajectory_varying_event_control_authority"
    )
    assert release["supporting_claim_ids"] == ["PD-CLAIM-317", "PD-CLAIM-318"]
    assert "participant-held-out" in release["remaining_scientific_gate"]
