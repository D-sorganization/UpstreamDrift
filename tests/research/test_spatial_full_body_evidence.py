"""Regression contracts for the committed spatial evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.scientific

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"


def _record() -> dict:
    return json.loads((DATA_DIR / "spatial_full_body_study.json").read_text())


def test_spatial_evidence_is_traceable_and_fail_closed() -> None:
    record = _record()

    assert record["schema_version"] == "spatial-full-body-common-state-evidence-v1"
    assert record["trajectory_kind"] == "prescribed_common_state_inverse_dynamics"
    assert record["formulations"] == [
        "lagrange_christoffel",
        "mujoco_native_inverse_dynamics",
    ]
    assert len(record["common_model_sha256"]) == 64
    assert record["claim_status"]["H3_passive_late_negative_couple"].startswith(
        "inconclusive"
    )
    assert record["claim_status"]["full_body_forward_closed_contact"] == "untested"
    assert record["claim_status"]["human_or_coaching_inference"] == "unsupported"
    for relative_path, expected_hash in record["source_sha256"].items():
        actual = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_hash


def test_spatial_interventions_and_cross_formulation_result_satisfy_contract() -> None:
    record = _record()
    intervention = record["intervention"]
    comparison = record["cross_formulation"]

    assert intervention["force_generated_couple_minimum_nm"] < -1.0
    assert 0.17 <= intervention["first_negative_time_s"] <= 0.18
    assert intervention["reversed_geometry_couple_maximum_nm"] > 1.0
    assert intervention["sign_reversal_residual_max_nm"] < 1e-12
    assert intervention["coincident_hands_couple_max_abs_nm"] == 0.0
    assert intervention["direct_club_torque_command_nm"] == 0.0
    assert comparison["classification"] == "equivalent"
    assert (
        comparison["maximum_absolute_generalized_force_error"]
        <= comparison["tolerance"]["absolute_generalized_force"]
    )
    assert (
        comparison["maximum_relative_inverse_dynamics_error"]
        <= comparison["tolerance"]["relative"]
    )
    assert comparison["maximum_absolute_mass_matrix_error"] < 1e-10
    assert comparison["maximum_relative_mass_matrix_error"] < 1e-10
    assert comparison["maximum_absolute_bias_force_error"] < 1e-7
    assert comparison["maximum_relative_bias_force_error"] < 1e-8
    assert comparison["external_load_convention_mismatch_relative_error"] > 0.20
    checks = record["spatial_checks"]
    assert checks["maximum_abs_generalized_load_power_residual_w"] < 1e-10
    assert checks["maximum_abs_reference_transport_power_residual_w"] < 1e-10


def test_spatial_array_bundle_preserves_common_observables() -> None:
    with np.load(DATA_DIR / "spatial_full_body_study.npz") as arrays:
        time = arrays["time_s"]
        baseline = arrays["force_generated_couple_nm"]
        reverse = arrays["reverse_geometry_couple_nm"]
        coincident = arrays["coincident_hands_couple_nm"]
        lagrange = arrays["inverse_dynamics_lagrange"]
        mujoco = arrays["inverse_dynamics_mujoco"]

        assert time.ndim == 1 and np.all(np.diff(time) > 0)
        assert lagrange.shape == mujoco.shape == (time.size, 20)
        np.testing.assert_allclose(reverse, -baseline, atol=1e-12)
        np.testing.assert_allclose(coincident, 0.0, atol=1e-12)
        np.testing.assert_allclose(lagrange, mujoco, atol=1e-8, rtol=1e-8)
        assert np.max(np.abs(arrays["action_reaction_power_residual_w"])) < 1e-10
        assert np.max(np.abs(arrays["generalized_load_power_residual_w"])) < 1e-10
        assert np.max(np.abs(arrays["reference_transport_power_residual_w"])) < 1e-10
