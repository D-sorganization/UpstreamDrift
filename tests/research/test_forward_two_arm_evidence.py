"""Evidence contracts for the forward constrained two-hand study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_forward_two_arm_study import (
    run_study,
    write_study,
)

pytestmark = [pytest.mark.scientific, pytest.mark.timeout(180)]
ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
)


@pytest.fixture(scope="module")
def evidence() -> tuple[dict, dict[str, np.ndarray]]:
    return run_study()


def test_evidence_executes_forward_model_and_declares_boundaries(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    assert record["schema_version"] == "forward-two-arm-evidence-v1"
    assert record["study_id"] == "forward-constrained-two-hand-v1"
    assert record["trajectory_kind"] == "forward_constrained_dynamics"
    assert record["prescribed_kinematics"] is False
    assert record["physiological_evidence"] is False
    assert record["human_validation"] is False
    assert set(record["source_files"]) == {
        "scripts/research/proximal_distal_energy/forward_two_arm.py",
        "scripts/research/proximal_distal_energy/run_forward_two_arm_study.py",
        "scripts/research/proximal_distal_energy/two_arm_closed_loop.py",
    }
    assert all(len(digest) == 64 for digest in record["source_files"].values())
    assert arrays["baseline_q"].shape[1] == 7
    assert arrays["baseline_contact_force_on_club_n"].shape[1:] == (2, 2)
    audit = record["constraint_acceleration_bias_audit"]
    assert audit["maximum_residual_m_s2"] < audit["tolerance_m_s2"] == 1e-7


def test_forward_constraints_and_energy_residuals_close(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    closure = record["baseline"]["closure"]
    assert closure["constraint_rank_min"] == 4
    assert closure["position_constraint_max_m"] < 1e-8
    assert closure["velocity_constraint_max_m_s"] < 1e-8
    assert closure["kkt_residual_max"] < 1e-9
    assert closure["acceleration_constraint_residual_max"] < 1e-9
    assert closure["work_energy_residual_abs_j"] < 0.1
    assert closure["contact_wrench_power_equivalence_max_w"] < 1e-9
    assert closure["constraint_two_sided_power_residual_max_w"] < 1e-9
    assert closure["projection_energy_change_absolute_sum_j"] >= abs(
        closure["projection_energy_change_sum_j"]
    )


def test_force_generated_negative_couple_persists_after_killswitch(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    baseline = record["baseline"]["force_generated_couple"]
    assert baseline["minimum_nm"] < -10.0
    assert 0.18 < baseline["first_negative_time_s"] < 0.24
    representative = record["representative_killswitch"]
    assert representative["cut_time_s"] == pytest.approx(0.2)
    assert representative["initial_force_generated_couple_nm"] < 0.0
    assert representative["negative_persistence_s"] >= 0.045
    assert representative["minimum_force_generated_couple_nm"] < -3.0
    np.testing.assert_array_equal(
        arrays["branch_q"][0],
        arrays["baseline_q"][representative["cut_index"]],
    )
    np.testing.assert_array_equal(
        arrays["branch_qdot"][0],
        arrays["baseline_qdot"][representative["cut_index"]],
    )
    assert representative["initial_state_exactly_inherited"] is True


def test_negative_control_and_numerical_sensitivity_are_recorded(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    negative_control = record["zero_contact_moment_arm_control"]
    assert negative_control["maximum_abs_force_generated_couple_nm"] < 1e-12
    convergence = record["timestep_convergence"]
    assert [row["step_s"] for row in convergence] == [0.002, 0.001, 0.0005]
    onset = [row["first_negative_time_s"] for row in convergence]
    assert max(onset) - min(onset) <= 0.004
    energy_residual = [row["work_energy_residual_abs_j"] for row in convergence]
    assert energy_residual[0] > energy_residual[1] > energy_residual[2]
    assert record["projection_sensitivity"][0]["projection_tolerance_m"] == 1e-08


def test_writer_is_deterministic_and_emits_portable_records(
    tmp_path,
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    json_path, npz_path = write_study(tmp_path, record=record, arrays=arrays)
    with json_path.open(encoding="utf-8") as stream:
        loaded = json.load(stream)
    with np.load(npz_path) as stored:
        np.testing.assert_array_equal(stored["baseline_q"], arrays["baseline_q"])
    assert loaded == record
    repeated, repeated_arrays = run_study()
    assert repeated == record
    for name, values in arrays.items():
        np.testing.assert_array_equal(repeated_arrays[name], values)


def test_committed_evidence_matches_executable_study(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    with (ARTIFACT_DIR / "forward_two_arm_study.json").open(encoding="utf-8") as stream:
        assert json.load(stream) == record
    with np.load(ARTIFACT_DIR / "forward_two_arm_study.npz") as stored:
        assert set(stored.files) == set(arrays)
        for name, values in arrays.items():
            np.testing.assert_array_equal(stored[name], values)
