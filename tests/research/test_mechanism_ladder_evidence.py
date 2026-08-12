"""Acceptance tests for recorded higher-order mechanism-ladder evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "mechanism_ladder_study.json"
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_common_schema_closes_transport_and_rotation_power(record: dict) -> None:
    audits = record["frame_and_transport_audits"]

    assert audits["maximum_rotation_power_residual_w"] < 1e-10
    assert audits["maximum_transport_power_residual_w"] < 1e-10
    assert audits["maximum_rotation_force_norm_residual_n"] < 1e-10
    assert audits["maximum_rotation_couple_norm_residual_nm"] < 1e-10


def test_mobile_hub_zero_amplitude_reduces_exactly_to_fixed_hub(record: dict) -> None:
    zero = next(row for row in record["mobile_hub_cases"] if row["amplitude_m"] == 0.0)

    assert zero["maximum_force_shift_n"] == pytest.approx(0.0, abs=1e-12)
    assert zero["maximum_power_difference_w"] == pytest.approx(0.0, abs=1e-12)


def test_closed_loop_constraint_diagnostics_remain_rank_consistent(
    record: dict,
) -> None:
    diagnostics = record["closed_loop_diagnostics"]

    assert diagnostics["minimum_rank"] == 4
    assert diagnostics["maximum_rank"] == 4
    assert diagnostics["minimum_nullspace_dimension"] == 1
    assert diagnostics["maximum_nullspace_dimension"] == 1
    assert diagnostics["maximum_constraint_velocity_residual"] < 1e-12


def test_model_discrepancy_table_distinguishes_executed_spatial_tiers(
    record: dict,
) -> None:
    rows = {row["tier"]: row for row in record["model_discrepancy_table"]}

    assert rows["three_link_planar"]["status"] == "executed"
    assert rows["mobile_hub_inverse_dynamics"]["status"] == "executed"
    assert rows["two_hand_closed_loop_geometry"]["status"] == "executed"
    assert rows["rotated_3d_wrench_audit"]["status"] == "executed"
    assert (
        rows["reduced_full_body_common_state_inverse_dynamics"]["status"] == "executed"
    )
    assert rows["reduced_spatial_forward_cross_engine_contact"]["status"] == "executed"
    assert (
        rows["articulated_full_body_forward_cross_engine_contact"]["status"]
        == "not_executed"
    )
    assert (
        "must not"
        in rows["articulated_full_body_forward_cross_engine_contact"]["boundary"]
    )


def test_headline_values_are_finite_and_bounded(record: dict) -> None:
    reference = record["three_link_reference"]

    assert 0.0 < reference["interface_force_at_delivery_n"] < 1_000.0
    assert abs(reference["interface_total_power_at_delivery_w"]) < 10_000.0
    assert reference["delivery_time_s"] == pytest.approx(0.4272523531, abs=1e-9)
