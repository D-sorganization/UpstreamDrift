"""Regression contracts for moving-base flexible-club evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_moving_base_flexible_study import (
    run_study,
    write_study,
)

pytestmark = [pytest.mark.scientific, pytest.mark.timeout(180)]
DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
)


@pytest.fixture(scope="module")
def evidence() -> tuple[dict, dict[str, np.ndarray]]:
    return run_study()


def test_evidence_is_one_coupled_forward_system(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    assert record["schema_version"] == "moving-base-flexible-club-evidence-v1"
    assert record["trajectory_kind"] == "forward_constrained_dynamics"
    assert record["base_motion_prescribed"] is False
    assert record["shaft_flex_prescribed"] is False
    assert record["physiological_evidence"] is False
    assert all(len(value) == 64 for value in record["source_files"].values())
    assert arrays["baseline_q"].shape[1] == 10
    assert arrays["baseline_contact_force_on_club_n"].shape[1:] == (2, 2)
    audit = record["constraint_acceleration_bias_audit"]
    assert audit["maximum_residual_m_s2"] < audit["tolerance_m_s2"] == 1e-7


def test_baseline_closes_constraints_power_and_energy(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    closure = evidence[0]["baseline"]["closure"]
    assert closure["constraint_rank_min"] == 4
    assert closure["position_constraint_max_m"] < 1e-8
    assert closure["velocity_constraint_max_m_s"] < 1e-8
    assert closure["kkt_residual_max"] < 1e-8
    assert closure["acceleration_constraint_residual_max"] < 1e-8
    assert closure["contact_power_identity_max_w"] < 1e-9
    assert closure["constraint_two_sided_power_residual_max_w"] < 1e-9
    assert closure["work_energy_residual_abs_j"] < 0.1
    assert closure["projection_energy_change_absolute_sum_j"] >= abs(
        closure["projection_energy_change_sum_j"]
    )


def test_base_motion_flex_and_negative_couple_are_model_outputs(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    baseline = evidence[0]["baseline"]
    assert 0.005 < baseline["maximum_base_displacement_m"] < 0.08
    assert baseline["maximum_abs_shaft_flex_deg"] > 1.0
    assert baseline["peak_shaft_strain_energy_j"] > 0.01
    assert baseline["force_generated_couple"]["minimum_nm"] < -1.0
    assert baseline["direct_wrist_torque"]["minimum_nm"] >= -1.01


def test_same_state_zero_command_branch_and_geometric_control(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    branch = record["zero_command_branch"]
    assert branch["cut_time_s"] == pytest.approx(0.2)
    assert branch["initial_state_exactly_inherited"] is True
    assert branch["minimum_force_generated_couple_nm"] < -1.0
    assert branch["negative_persistence_s"] >= 0.045
    np.testing.assert_array_equal(
        arrays["branch_q"][0], arrays["baseline_q"][branch["cut_index"]]
    )
    np.testing.assert_array_equal(
        arrays["branch_qdot"][0], arrays["baseline_qdot"][branch["cut_index"]]
    )
    assert (
        record["coincident_grip_negative_control"][
            "maximum_abs_force_generated_couple_nm"
        ]
        < 1e-12
    )


def test_sensitivity_and_convergence_expose_falsifiers(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    assert {row["parameter"] for row in record["mechanism_sensitivity"]} == {
        "base_stiffness_n_m",
        "shaft_stiffness_nm_rad",
        "shaft_damping_nms_rad",
    }
    convergence = record["timestep_convergence"]
    assert [row["step_s"] for row in convergence] == [0.002, 0.001, 0.0005]
    residuals = [row["work_energy_residual_abs_j"] for row in convergence]
    assert residuals[0] > residuals[1] > residuals[2]
    assert len(record["falsification_tests"]) >= 5


def test_committed_artifacts_match_executable_study(
    tmp_path: Path,
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    json_path, npz_path = write_study(tmp_path, record=record, arrays=arrays)
    assert json.loads(json_path.read_text(encoding="utf-8")) == record
    with np.load(npz_path) as stored:
        assert set(stored.files) == set(arrays)
        for name, values in arrays.items():
            np.testing.assert_array_equal(stored[name], values)
    assert (
        json.loads(
            (DATA_DIR / "moving_base_flexible_study.json").read_text(encoding="utf-8")
        )
        == record
    )
    with np.load(DATA_DIR / "moving_base_flexible_study.npz") as stored:
        assert set(stored.files) == set(arrays)
        for name, values in arrays.items():
            np.testing.assert_array_equal(stored[name], values)
