"""Evidence gates for the coupled higher-order shaft experiment."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
JSON_PATH = DATA / "moving_base_modal_shaft_study.json"
NPZ_PATH = DATA / "moving_base_modal_shaft_study.npz"


def _record() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def test_evidence_sources_and_arrays_are_hash_bound() -> None:
    record = _record()
    for relative, expected in record["source_sha256"].items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert (
        sha256(NPZ_PATH.read_bytes()).hexdigest() == record["array_artifact"]["sha256"]
    )
    with np.load(NPZ_PATH) as archive:
        assert sorted(archive.files) == record["array_artifact"]["array_names"]
        assert all(np.all(np.isfinite(archive[name])) for name in archive.files)
        assert archive["baseline_q"].shape[1] == 12
        assert archive["baseline_modal_coordinates"].shape[1] == 3


def test_modal_transport_matches_the_finite_element_reference() -> None:
    transport = _record()["modal_transport"]
    assert transport["maximum_frequency_discrepancy_relative"] < 2e-3
    assert transport["modal_mass_identity_max_abs"] < 2e-10
    assert len(transport["finite_element_frequencies_hz"]) == 3


def test_killswitch_geometry_and_numerical_controls_bound_the_claim() -> None:
    record = _record()
    branch = record["same_state_killswitch"]
    assert branch["prebranch_state_max_abs_difference"] == 0.0
    assert branch["complete_distal_command_removed"] is True
    assert branch["negative_couple_persistence_s"] >= 0.025
    assert branch["minimum_force_generated_couple_nm"] < -20.0
    assert branch["closure"]["contact_power_identity_max_w"] < 1e-9
    geometry = record["geometry_controls"]
    assert geometry["same_achieved_forces"] is True
    assert geometry["coincident_maximum_abs_nm"] == 0.0
    assert geometry["reversed_arm_sign_residual_max_abs_nm"] == 0.0
    refinement = record["timestep_refinement"]
    residuals = [row["closure"]["work_energy_residual_abs_j"] for row in refinement]
    assert residuals[2] < residuals[1] < residuals[0]
    persistence = np.array([row["negative_couple_persistence_s"] for row in refinement])
    assert np.ptp(persistence) <= 0.0005


def test_short_pulse_exposes_truncation_without_invalidating_three_modes() -> None:
    record = _record()
    smooth = {row["mode_count"]: row for row in record["smooth_mode_comparison"]}
    pulse = {row["mode_count"]: row for row in record["short_pulse_mode_comparison"]}
    assert (
        pulse[1]["tip_deflection_rms_difference_m"]
        > 2.5 * smooth[1]["tip_deflection_rms_difference_m"]
    )
    assert pulse[3]["tip_deflection_rms_difference_m"] < 1e-5
    assert pulse[3]["clubhead_position_max_difference_m"] < 2e-6
    assert pulse[6]["clubhead_position_max_difference_m"] == 0.0


def test_claim_boundary_does_not_promote_synthetic_properties() -> None:
    record = _record()
    status = record["claim_status"]
    assert status["distributed_modal_shaft_coupled_forward"].startswith("supported")
    assert status["equipment_calibration"].startswith("untested")
    assert status["human_strategy"] == "untested"
    assert any("not equipment calibration" in item for item in record["limitations"])
