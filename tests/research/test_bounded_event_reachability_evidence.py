"""Governed evidence contracts for bounded event reachability."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_bounded_event_reachability import (
    ARRAY_PATH,
    REPORT_PATH,
    build_evidence,
    validate_report,
)

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def regenerated_evidence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the registered serial study once for all reproducibility tests."""

    return build_evidence()


@pytest.fixture(scope="module")
def registered_report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_registered_report_is_reproducible_and_source_current(
    regenerated_evidence: tuple[dict[str, Any], dict[str, np.ndarray]],
    registered_report: dict[str, Any],
) -> None:
    regenerated_report, _ = regenerated_evidence

    assert registered_report == regenerated_report
    assert validate_report(registered_report) == {
        "continuation_trials": 28,
        "control_trials": 10,
        "typed_outcomes": 28,
    }


def test_registered_arrays_exactly_match_recomputed_full_precision_evidence(
    regenerated_evidence: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    _, regenerated_arrays = regenerated_evidence

    with np.load(ARRAY_PATH, allow_pickle=False) as registered_arrays:
        assert set(registered_arrays.files) == set(regenerated_arrays)
        for name, expected in regenerated_arrays.items():
            np.testing.assert_array_equal(registered_arrays[name], expected)


def test_registered_outcomes_preserve_feasibility_and_optimality_boundaries(
    registered_report: dict[str, Any],
) -> None:
    outcomes = registered_report["outcome_counts"]
    qualification = registered_report["qualification"]

    assert outcomes == {
        "event_status": {"transverse": 38},
        "replay_feasibility_status": {"feasible": 32, "infeasible": 6},
        "solver_status": {"converged": 32, "infeasible": 6},
    }
    assert qualification["feasibility_evidence_adequate"] is True
    assert qualification["optimality_evidence_adequate"] is False
    assert qualification["registered_release_adequate"] is True
    assert qualification["multistart_adequate"] is False
    assert qualification["channel_ranking_available"] is False


def test_zero_authority_is_the_only_registered_displaced_target_failure(
    registered_report: dict[str, Any],
) -> None:
    displaced = [
        record
        for record in registered_report["continuation_trials"]
        if record["target_name"] != "zero"
    ]
    infeasible = [
        (record["target_name"], record["channel"])
        for record in displaced
        if record["replay_feasibility_status"] == "infeasible"
    ]

    assert len(displaced) == 24
    assert len(infeasible) == 6
    assert {channel for _, channel in infeasible} == {"zero"}
    assert all(
        record["replay_feasibility_status"] == "feasible"
        for record in displaced
        if record["channel"] != "zero"
    )


def test_rankings_and_human_interpretation_remain_unavailable(
    registered_report: dict[str, Any],
) -> None:
    availability = registered_report["availability"]

    assert availability["registered_model_scenario_feasibility"] == "available"
    assert availability["channel_or_controller_ranking"] == "suppressed"
    for name in (
        "global_nonlinear_reachability",
        "human_actuator_interpretation",
        "passive_torque_interpretation",
        "coaching_recommendation",
    ):
        assert availability[name] == "unavailable"


def test_validation_rejects_optimality_promotion_and_release_semantic_drift(
    registered_report: dict[str, Any],
) -> None:
    promoted = deepcopy(registered_report)
    promoted["qualification"]["channel_ranking_available"] = True
    with pytest.raises(ValueError, match="optimality adequacy"):
        validate_report(promoted)

    conflated = deepcopy(registered_report)
    conflated["qualification"]["registered_release_adequate"] = False
    with pytest.raises(ValueError, match="release adequacy"):
        validate_report(conflated)

    human_promotion = deepcopy(registered_report)
    human_promotion["availability"]["coaching_recommendation"] = "available"
    with pytest.raises(ValueError, match="coaching_recommendation"):
        validate_report(human_promotion)
