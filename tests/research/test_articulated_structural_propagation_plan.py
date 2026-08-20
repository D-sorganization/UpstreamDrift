"""Contracts for fail-closed structural-corner headline propagation planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_propagation_plan import (
    build_structural_propagation_plan,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
COMMITTED = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "articulated_structural_propagation_plan.json"
)


@pytest.fixture(scope="module")
def plan():
    return build_structural_propagation_plan()


def test_plan_binds_all_seven_authority_corners(plan) -> None:
    assert plan["status"] == "ready"
    assert len(plan["corners"]) == 7
    assert len(plan["authority_campaign_sha256"]) == 64
    assert len(plan["design_sha256"]) == 64
    assert plan["design"]["case_indices"] == [0, 8, 9, 17]
    assert plan["design"]["phase_indices"] == [0, 6, 12]


def test_nominal_plan_reproduces_registered_atlas_sizes(plan) -> None:
    nominal = plan["corners"][0]

    assert nominal["corner_id"] == "nominal"
    assert nominal["requested_state_count"] == 12
    assert nominal["feasible_state_count"] == 12
    assert nominal["expected_shaft_trajectory_count"] == 384
    assert nominal["expected_ground_trajectory_count"] == 576
    assert nominal["expected_headline_cell_count_per_atlas"] == 384


def test_low_height_plan_retains_failure_and_runs_other_states(plan) -> None:
    low_height = plan["corners"][1]

    assert low_height["corner_id"] == "height_scale-low"
    assert low_height["status"] == "ready_with_retained_failure"
    assert low_height["requested_state_count"] == 12
    assert low_height["feasible_state_count"] == 11
    assert low_height["retained_failures"] == [
        {"case_index": 0, "phase_index": 12, "failure_class": "ik_nonconvergence"}
    ]
    assert [0, 12] not in low_height["feasible_states"]
    assert low_height["expected_shaft_trajectory_count"] == 352
    assert low_height["expected_ground_trajectory_count"] == 528
    assert low_height["expected_headline_cell_count_per_atlas"] == 352


def test_every_corner_binds_models_and_accounts_for_states(plan) -> None:
    for corner in plan["corners"]:
        assert (
            corner["feasible_state_count"] + len(corner["retained_failures"])
            == corner["requested_state_count"]
        )
        authority = corner["authority"]
        assert len(authority["authority_sha256"]) == 64
        assert set(authority["model_sha256"]) == {"0", "8", "9", "17"}


def test_committed_plan_is_exactly_reproducible(plan) -> None:
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))

    assert committed == plan
