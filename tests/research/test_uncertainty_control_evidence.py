"""Regression contracts for the Phase 4 committed evidence bundle."""

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
    return json.loads((DATA_DIR / "uncertainty_control_study.json").read_text())


def test_evidence_is_traceable_and_declares_nonphysiological_boundaries() -> None:
    record = _record()

    assert record["schema_version"] == "proximal-distal-uncertainty-control-v1"
    assert record["registered_before_preferred_result"] is True
    assert record["actuator_contract"]["physiological_interpretation"] == "unsupported"
    assert record["control_comparison"]["universal_optimum_claim"] == "unsupported"
    assert record["claim_status"]["human_or_coaching_inference"] == "unsupported"
    for relative_path, expected_hash in record["source_sha256"].items():
        actual = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_hash


def test_designs_are_independent_and_evidence_arrays_are_complete() -> None:
    record = _record()
    with np.load(DATA_DIR / record["array_artifact"]) as arrays:
        assert arrays["global_physical_design"].shape == (24, 12)
        assert arrays["global_outputs"].shape == (24, 6)
        assert arrays["training_outputs"].shape == (8, 6, 6)
        assert arrays["held_out_outputs"].shape == (8, 6, 6)
        assert arrays["prcc"].shape == (12, 6)
        assert np.max(np.abs(arrays["prcc"])) <= 1.0 + 1e-12
        assert not np.array_equal(
            arrays["training_unit_design"], arrays["held_out_unit_design"]
        )
        assert np.all(np.isfinite(arrays["global_outputs"]))


def test_identifiability_fails_closed_for_unresolvable_parameters() -> None:
    identifiability = _record()["identifiability"]
    hand = identifiability["individual_hand_force_from_net_planar_wrench"]
    coupled = identifiability["coupled_parameter_screen"]

    assert hand["mapping_rank"] == 3
    assert hand["unknown_force_components"] == 4
    assert hand["nullity"] == 1
    assert hand["status"].startswith("structurally_nonidentifiable")
    assert coupled["parameter_count"] == 12
    assert coupled["observable_count"] == 6
    assert coupled["nullity_lower_bound"] >= 6
    assert coupled["status"].startswith("practically_nonidentifiable")


def test_anticipatory_restrain_strategy_has_better_held_out_lower_tail_speed() -> None:
    candidates = {
        row["name"]: row for row in _record()["control_comparison"]["candidates"]
    }
    early = candidates["early_restrain"]["held_out"]
    reactive = candidates["late_drive"]["held_out"]

    assert early["delivery_speed_q10_m_s"] > reactive["delivery_speed_q10_m_s"]
    # The benefit is conditional: the face/path proxy is worse.
    assert early["face_path_error_mean_deg"] > reactive["face_path_error_mean_deg"]


def test_all_rollouts_close_constraints_and_contact_power() -> None:
    closure = _record()["closure"]

    assert closure["maximum_position_constraint_m"] < 1.1e-10
    assert closure["maximum_velocity_constraint_m_s"] < 1e-10
    assert closure["maximum_kkt_residual"] < 1e-9
    assert closure["maximum_contact_power_residual_w"] < 1e-9
