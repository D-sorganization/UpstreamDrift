"""Acceptance tests for recorded higher-order mechanism-ladder evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.make_mechanism_ladder_figures import (
    _schematic_tiers,
)
from scripts.research.proximal_distal_energy.run_mechanism_ladder_study import (
    _wrist2_velocity,
    build_study,
    write_outputs,
)
from src.shared.python.pendulum_simulator.physics_triple import forward_kinematics
from scripts.research.proximal_distal_energy.flexible_shaft_study import (
    FlexibleShaftParams,
)

DATA = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "mechanism_ladder_study.json"
)
ROOT = Path(__file__).resolve().parents[2]

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
    assert audits["rotation_sample_count"] == 51
    assert audits["reference_translation_m"] == pytest.approx([0.2, -0.1, 0.05])


def test_evidence_uses_content_provenance_and_explicit_path_contract(
    record: dict,
) -> None:
    assert record["schema_version"] == "mechanism-ladder-study-v2"
    assert record["study_id"] == "common-observable-model-ladder"
    assert "git_sha" not in record["provenance"]
    assert len(record["source_sha256"]) == 8
    contract = record["mobile_hub_contract"]
    assert contract["fundamental_frequency_hz"] == pytest.approx(1.25)
    assert contract["second_harmonic_frequency_hz"] == pytest.approx(2.5)
    assert contract["second_harmonic_position_amplitude_ratio"] == pytest.approx(0.5)
    velocity = record["kinematic_velocity_audit"]
    assert velocity["method"].startswith("analytic relative-coordinate")
    assert 0.02 < velocity["maximum_position_gradient_discrepancy_m_s"] < 0.031
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_second_joint_velocity_uses_exact_relative_coordinate_kinematics() -> None:
    params = FlexibleShaftParams.reference()
    state = np.array([[0.4, -0.2, 0.1, 3.0, -0.7, 0.2]])
    velocity = _wrist2_velocity(state, params)[0]
    step = 1e-7
    q_minus = state[0, :3] - step * state[0, 3:]
    q_plus = state[0, :3] + step * state[0, 3:]
    position_minus = np.asarray(forward_kinematics(*q_minus, params.triple())["wrist2"])
    position_plus = np.asarray(forward_kinematics(*q_plus, params.triple())["wrist2"])

    assert velocity == pytest.approx(
        (position_plus - position_minus) / (2.0 * step), abs=1e-8
    )


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
    assert (
        "forward_contact"
        in rows["reduced_spatial_forward_cross_engine_contact"]["capabilities"]
    )
    assert (
        "articulated_contact"
        not in rows["reduced_spatial_forward_cross_engine_contact"]["capabilities"]
    )
    assert rows["articulated_full_body_forward_cross_engine_contact"][
        "capabilities"
    ] == ["articulated_contact"]


def test_schematic_derives_executed_and_open_status_from_evidence(record: dict) -> None:
    tiers = {row["tier"]: row for row in _schematic_tiers(record)}

    assert tiers["reduced_spatial_forward_cross_engine_contact"]["status"] == (
        "executed"
    )
    assert tiers["articulated_full_body_forward_cross_engine_contact"]["status"] == (
        "not_executed"
    )


def test_evidence_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_json, first_npz = write_outputs(first)
    second_json, second_npz = write_outputs(second)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_npz.read_bytes() == second_npz.read_bytes()
    built, _ = build_study()
    assert json.loads(first_json.read_text(encoding="utf-8")) == built


def test_three_link_arrays_retain_the_full_wrench_twist_power_identity() -> None:
    _, arrays = build_study()
    direct = np.einsum(
        "ij,ij->i", arrays["three_link__force"], arrays["three_link__velocity"]
    ) + np.einsum(
        "ij,ij->i",
        arrays["three_link__couple"],
        arrays["three_link__angular_velocity"],
    )

    assert direct == pytest.approx(arrays["three_link__power"], abs=1e-12)


def test_headline_values_are_finite_and_bounded(record: dict) -> None:
    reference = record["three_link_reference"]

    assert 0.0 < reference["interface_force_at_delivery_n"] < 1_000.0
    assert abs(reference["interface_total_power_at_delivery_w"]) < 10_000.0
    assert reference["delivery_time_s"] == pytest.approx(0.4272523531, abs=1e-9)
