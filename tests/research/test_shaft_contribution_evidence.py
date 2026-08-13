"""Regression tests for the recorded shaft-contribution evidence package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_shaft_contribution_study import (
    write_outputs,
)

RECORD = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "shaft_contribution_study.json"
)
ROOT = Path(__file__).resolve().parents[2]
TRACE_RECORD = RECORD.with_name("shaft_contribution_traces.npz")


@pytest.fixture(scope="module")
def evidence() -> dict:
    with RECORD.open(encoding="utf-8") as stream:
        return json.load(stream)


def _variant(evidence: dict, name: str) -> dict:
    return next(row for row in evidence["variant_summaries"] if row["name"] == name)


def test_matched_rigid_and_flexible_reference_cases_both_deliver(
    evidence: dict,
) -> None:
    flexible = _variant(evidence, "flexible_reference")
    rigid = _variant(evidence, "rigid_matched")

    assert flexible["impact"]["impact_found"]
    assert rigid["impact"]["impact_found"]
    assert flexible["impact"]["impact_speed_m_s"] - rigid["impact"][
        "impact_speed_m_s"
    ] == pytest.approx(0.10833967558328794)
    assert flexible["impact"]["shaft_flex_at_impact_deg"] == pytest.approx(
        -3.581146177759325
    )


def test_energy_and_dissipation_contracts_close(evidence: dict) -> None:
    for name in (
        "flexible_reference",
        "rigid_matched",
        "gravity_disabled",
        "joint_damping_disabled",
        "shaft_damping_disabled",
    ):
        row = _variant(evidence, name)
        assert row["energy"]["maximum_abs_energy_closure_error_j"] < 0.07
        assert row["energy"]["shaft_damping_dissipation_j"] <= 1e-12
        assert row["energy"]["joint_damping_dissipation_j"] <= 1e-12


def test_ablations_remove_only_their_declared_terms(evidence: dict) -> None:
    gravity_off = _variant(evidence, "gravity_disabled")
    joint_damping_off = _variant(evidence, "joint_damping_disabled")
    shaft_damping_off = _variant(evidence, "shaft_damping_disabled")

    assert gravity_off["terms"]["gravity"]["peak_abs_generalized_torque_nm"] == 0.0
    assert (
        joint_damping_off["terms"]["joint_damping"]["peak_abs_generalized_torque_nm"]
        == 0.0
    )
    assert (
        shaft_damping_off["terms"]["shaft_damping"]["peak_abs_generalized_torque_nm"]
        == 0.0
    )


def test_timestep_refinement_converges_impact_metrics(evidence: dict) -> None:
    rows = {row["dt_s"]: row for row in evidence["timestep_rows"]}
    fine = rows[0.00025]["impact"]

    assert abs(rows[0.0005]["impact"]["impact_time_s"] - fine["impact_time_s"]) < 5e-5
    assert (
        abs(rows[0.0005]["impact"]["impact_speed_m_s"] - fine["impact_speed_m_s"])
        < 0.006
    )
    assert (
        abs(rows[0.001]["impact"]["impact_speed_m_s"] - fine["impact_speed_m_s"]) < 0.02
    )


def test_robustness_grid_covers_every_declared_combination(evidence: dict) -> None:
    grid = evidence["robustness_grid"]
    expected = (
        len(grid["stiffness_values_nm_rad"])
        * len(grid["damping_values_nms_rad"])
        * len(grid["cut_times_s"])
    )

    assert expected == 120
    assert len(grid["rows"]) == expected
    assert all(row["impact_found"] for row in grid["rows"])


def test_provenance_uses_content_hashes_and_exact_kinematics(evidence: dict) -> None:
    assert evidence["schema_version"] == "shaft-contribution-study-v2"
    assert evidence["study_id"] == "matched-rigid-flexible-shaft-contribution-study"
    assert "git_sha" not in evidence["provenance"]
    assert len(evidence["source_sha256"]) == 3
    for relative, expected in evidence["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for row in evidence["variant_summaries"]:
        impact = row["impact"]
        assert impact["tip_velocity_method"].startswith("analytic")
        assert impact["maximum_tip_position_gradient_discrepancy_m_s"] > 0.0
        assert row["energy"]["velocity_bias_power_identity_verified"] is True
        assert row["energy"]["velocity_bias_power_identity_tolerance_w"] == 1e-6


def test_compliant_interface_power_is_two_sided_and_closed() -> None:
    with np.load(TRACE_RECORD) as arrays:
        for name in ("flexible_reference", "rigid_matched"):
            prefix = f"{name}__shaft_port_"
            force_sum = (
                arrays[prefix + "distal_force_power"]
                + arrays[prefix + "proximal_force_power"]
            )
            np.testing.assert_allclose(force_sum, 0.0, atol=1e-12)
            np.testing.assert_allclose(
                arrays[prefix + "adjacent_body_power"],
                arrays[prefix + "relative_coordinate_power"],
                atol=1e-12,
            )


def test_balanced_grid_quantifies_main_effects_without_causal_promotion(
    evidence: dict,
) -> None:
    attribution = evidence["robustness_grid"][
        "balanced_main_effect_fraction_of_total_ss"
    ]
    speed = attribution["impact_speed_m_s"]
    strain = attribution["peak_shaft_strain_energy_j"]
    assert speed["torque_cut_time_s"] > 0.99
    assert speed["shaft_stiffness_nm_rad"] < 0.002
    assert strain["shaft_damping_nms_rad"] > strain["shaft_stiffness_nm_rad"]
    assert (
        "not sampling uncertainty"
        in evidence["robustness_grid"]["attribution_boundary"]
    )


@pytest.mark.timeout(360)
def test_outputs_replay_with_declared_numerical_tolerance(tmp_path: Path) -> None:
    replay_json, replay_npz = write_outputs(tmp_path / "replay")

    assert replay_json.read_bytes() == RECORD.read_bytes()
    with np.load(replay_npz) as replay, np.load(TRACE_RECORD) as authority:
        assert replay.files == authority.files
        for name in authority.files:
            np.testing.assert_allclose(
                replay[name],
                authority[name],
                rtol=0.0,
                atol=1.0e-6,
                err_msg=f"semantic replay mismatch for {name}",
            )
