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
EVIDENCE_DIR = REPO_ROOT / "docs/development/full_body_models/evidence/show_a9"
MODELS_DIR = REPO_ROOT / "docs/development/full_body_models"
V1_SPEC_PATH = MODELS_DIR / "full_body_spec_anthro_driver.json"
V2_SPEC_PATH = MODELS_DIR / "full_body_spec_anthro_driver_v2.json"
EXPECTED_V1 = "fd72d8be7a6c5c557a10886a7d2717a4eb4619805a24cc1be644215eea21dcb3"
EXPECTED_V2 = "323c2182ed1592552de700c02ab0f297e620660f2ac5a9c7f0f01e6af8f25b8c"
EXPECTED_CAP = "cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d"


def test_a9_candidate_receipt_exists_and_pins_hashes() -> None:
    receipt = load_a9_candidate_receipt()
    assert receipt["issue"] == 14734 and receipt["engine"] == "mujoco"
    inputs = receipt["input_hashes"]
    assert inputs["v1_spec_sha256"] == EXPECTED_V1
    assert inputs["v2_spec_sha256"] == EXPECTED_V2
    assert inputs["capture_sha256"] == EXPECTED_CAP
    assert inputs["baseline_commit"] == "fa7a9b7075"
    assert len(receipt["output_hashes"]["receipt_sha256"]) == 64
    for f in EVIDENCE_DIR.glob("**/*"):
        if f.is_file():
            assert f.suffix not in {".npz", ".c3d", ".bundle", ".mp4"}


def test_a9_address_grip_roll_holds_a7_convention_15_deg() -> None:
    receipt = load_a9_candidate_receipt()
    grip = receipt["address_grip"]
    assert grip["target_deg"] == pytest.approx(15.0)
    assert abs(grip["lead_roll_deg"] - 15.0) <= 1.0
    assert grip["held_within_band"] and grip["modelling_choice"]
    assert "12 o'clock" in grip["convention"]


def test_a9_wrist_coordinate_violations_cataloged_preserving_a6_lesson() -> None:
    receipt = load_a9_candidate_receipt()
    viol = receipt["wrist_coordinate_violations"]
    for j in (
        "RWInputX",
        "RWInputY",
        "LWInputX",
        "LWInputY",
        "RScapInputX",
        "RScapInputY",
    ):
        assert j in viol
        assert viol[j]["frames"] > 0 and len(viol[j]["range"]) == 2


def test_a9_side_by_side_comparison_matches_11_04_baseline() -> None:
    comp = compare_against_baseline(CANDIDATE_METRICS, BASELINE_METRICS)
    assert isinstance(comp, RefitComparison)
    assert comp.baseline.marker_rms_mm == pytest.approx(50.9)
    assert comp.candidate.marker_rms_mm == pytest.approx(55.3)
    assert comp.baseline.joint_rms_deg == pytest.approx(2.4)
    assert comp.candidate.joint_rms_deg == pytest.approx(2.4)
    assert comp.baseline.closure_residual_mm == pytest.approx(62.2)
    assert comp.candidate.closure_residual_mm == pytest.approx(76.8)
    assert comp.baseline.weight_fraction_range == (0.04, 2.82)
    assert comp.candidate.weight_fraction_range == (0.00, 2.82)
    assert comp.baseline.peak_joint_torque_nm == pytest.approx(537.0)
    assert comp.candidate.peak_joint_torque_nm == pytest.approx(537.0)
    assert comp.baseline.in_support_fraction == pytest.approx(0.637)
    assert comp.candidate.in_support_fraction == pytest.approx(0.637)


def test_a9_gate_evaluation_on_receipt_is_rejected_g3() -> None:
    eval_res = evaluate_a9_receipt_gates(load_a9_candidate_receipt())
    assert eval_res["status"] == "REJECTED" and not eval_res["is_physically_accepted"]
    assert eval_res["horizon"] == "G3"
    gates = {g["name"]: g for g in eval_res["gates"]}
    assert gates["max_closure_residual_m"]["status"] == "failed"
    assert gates["weight_fraction"]["status"] == "failed"
    assert gates["whole_marker_rmse_m"]["status"] == "passed"


def test_a9_registry_update_is_not_earned_and_recorded_as_candidate() -> None:
    assert not is_registry_update_earned(CANDIDATE_METRICS, BASELINE_METRICS)
    reg = load_a9_candidate_receipt()["registry_update"]
    assert not reg["earned"] and reg["action"] == "record_as_candidate"
    assert "does not beat baseline" in reg["reason"].lower()


def test_a9_spec_carries_v2_trunk_segments() -> None:
    order = json.loads(V2_SPEC_PATH.read_text(encoding="utf-8"))["coordinate_order"]
    for j in (
        "lumbar_flexion",
        "lumbar_lateral_bending",
        "lumbar_axial_rotation",
        "thoracic_flexion",
        "thoracic_lateral_bending",
        "thoracic_axial_rotation",
    ):
        assert j in order
