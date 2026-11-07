"""Tests for SHOW-A9 (#14734): refit of best-match MuJoCo driver on A7 grip + A8 trunk.

Validates the A9 candidate refit against the 11-04 baseline (fa7a9b7075):
1. Side-by-side comparison on marker RMS, joint RMS, closure, weight fraction,
   peak torque, and in-support fraction.
2. A7 address grip roll holds 15 deg +/- 1 deg face-on; wrist coordinate
   violations are explicitly cataloged (preserving the A6 lesson).
3. Best-match registry update is only earned if the candidate beats the baseline;
   otherwise recorded as a candidate with trade-offs.
4. Input and output SHA-256 hashes are pinned; no bundle or mocap committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.show_a9_refit import (
    BASELINE_METRICS,
    CANDIDATE_METRICS,
    RefitComparison,
    compare_against_baseline,
    evaluate_a9_receipt_gates,
    is_registry_update_earned,
    load_a9_candidate_receipt,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = (
    REPO_ROOT / "docs" / "development" / "full_body_models" / "evidence" / "show_a9"
)
MODELS_DIR = REPO_ROOT / "docs" / "development" / "full_body_models"
V1_SPEC_PATH = MODELS_DIR / "full_body_spec_anthro_driver.json"
V2_SPEC_PATH = MODELS_DIR / "full_body_spec_anthro_driver_v2.json"

EXPECTED_V1_SHA256 = "fd72d8be7a6c5c557a10886a7d2717a4eb4619805a24cc1be644215eea21dcb3"
EXPECTED_V2_SHA256 = "323c2182ed1592552de700c02ab0f297e620660f2ac5a9c7f0f01e6af8f25b8c"
EXPECTED_CAPTURE_SHA256 = (
    "cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d"
)


def test_a9_candidate_receipt_exists_and_pins_hashes() -> None:
    receipt = load_a9_candidate_receipt()
    assert receipt["issue"] == 14734
    assert receipt["model_id"] == "mujoco_driver_candidate_a9"
    assert receipt["engine"] == "mujoco"

    inputs = receipt["input_hashes"]
    assert inputs["v1_spec_sha256"] == EXPECTED_V1_SHA256
    assert inputs["v2_spec_sha256"] == EXPECTED_V2_SHA256
    assert inputs["capture_sha256"] == EXPECTED_CAPTURE_SHA256
    assert inputs["baseline_commit"] == "fa7a9b7075"

    assert len(receipt["output_hashes"]["receipt_sha256"]) == 64

    # Ensure no binary mocap, npz or bundle files are committed in evidence
    forbidden_suffixes = {".npz", ".c3d", ".bundle", ".mp4"}
    for f in EVIDENCE_DIR.glob("**/*"):
        if f.is_file():
            assert f.suffix not in forbidden_suffixes, f"Forbidden file in git: {f}"


def test_a9_address_grip_roll_holds_a7_convention_15_deg() -> None:
    receipt = load_a9_candidate_receipt()
    grip = receipt["address_grip"]
    assert grip["target_deg"] == pytest.approx(15.0)
    assert abs(grip["lead_roll_deg"] - 15.0) <= 1.0
    assert grip["held_within_band"] is True
    assert grip["modelling_choice"] is True
    assert "12 o'clock" in grip["convention"]


def test_a9_wrist_coordinate_violations_cataloged_preserving_a6_lesson() -> None:
    receipt = load_a9_candidate_receipt()
    violations = receipt["wrist_coordinate_violations"]
    # The A6 lesson: grip roll must not absorb wrist error; violations cataloged
    for joint in (
        "RWInputX",
        "RWInputY",
        "LWInputX",
        "LWInputY",
        "RScapInputX",
        "RScapInputY",
    ):
        assert joint in violations, f"Missing cataloged violation for {joint}"
        entry = violations[joint]
        assert entry["frames"] > 0
        assert len(entry["range"]) == 2
        assert entry["min"] < entry["range"][0] or entry["max"] > entry["range"][1]


def test_a9_side_by_side_comparison_matches_11_04_baseline() -> None:
    comparison = compare_against_baseline(CANDIDATE_METRICS, BASELINE_METRICS)
    assert isinstance(comparison, RefitComparison)

    # 1. marker RMS
    assert comparison.baseline.marker_rms_mm == pytest.approx(50.9)
    assert comparison.candidate.marker_rms_mm == pytest.approx(55.3)

    # 2. joint RMS
    assert comparison.baseline.joint_rms_deg == pytest.approx(2.4)
    assert comparison.candidate.joint_rms_deg == pytest.approx(2.4)

    # 3. closure residual
    assert comparison.baseline.closure_residual_mm == pytest.approx(62.2)
    assert comparison.candidate.closure_residual_mm == pytest.approx(76.8)

    # 4. weight fraction range
    assert comparison.baseline.weight_fraction_range == (0.04, 2.82)
    assert comparison.candidate.weight_fraction_range == (0.00, 2.82)

    # 5. peak joint torque
    assert comparison.baseline.peak_joint_torque_nm == pytest.approx(537.0)
    assert comparison.candidate.peak_joint_torque_nm == pytest.approx(537.0)

    # 6. in-support fraction
    assert comparison.baseline.in_support_fraction == pytest.approx(0.637)
    assert comparison.candidate.in_support_fraction == pytest.approx(0.637)


def test_a9_gate_evaluation_on_receipt_is_rejected_g3() -> None:
    receipt = load_a9_candidate_receipt()
    gates_eval = evaluate_a9_receipt_gates(receipt)
    assert gates_eval["status"] == "REJECTED"
    assert gates_eval["is_physically_accepted"] is False
    assert gates_eval["horizon"] == "G3"

    # closure residual and weight fraction min fail G3
    gates = {g["name"]: g for g in gates_eval["gates"]}
    assert gates["max_closure_residual_m"]["status"] == "failed"
    assert gates["weight_fraction"]["status"] == "failed"
    assert gates["whole_marker_rmse_m"]["status"] == "passed"


def test_a9_registry_update_is_not_earned_and_recorded_as_candidate() -> None:
    earned = is_registry_update_earned(CANDIDATE_METRICS, BASELINE_METRICS)
    assert earned is False

    receipt = load_a9_candidate_receipt()
    reg = receipt["registry_update"]
    assert reg["earned"] is False
    assert reg["action"] == "record_as_candidate"
    assert "does not beat baseline" in reg["reason"].lower()


def test_a9_spec_carries_v2_trunk_segments() -> None:
    v2_spec = json.loads(V2_SPEC_PATH.read_text(encoding="utf-8"))
    order = v2_spec["coordinate_order"]
    for trunk_joint in (
        "lumbar_flexion",
        "lumbar_lateral_bending",
        "lumbar_axial_rotation",
        "thoracic_flexion",
        "thoracic_lateral_bending",
        "thoracic_axial_rotation",
    ):
        assert trunk_joint in order
