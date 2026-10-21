"""Tests for OpenSim barbell lift joint reaction loads (OSL-5, #13043).

Follows TDD red-first against real OpenSim 4.6.
Verifies:
1. Pure-Python record builder: unavailable lifts explicitly report reason, never fake numbers.
2. README rendering: missing fields render 'unavailable', never hard-coded 0.00.
3. Static-hold analytic bound:
   - Frame/sign self-consistency at default pose.
   - Minimal-acceleration real squat frame evaluation.
4. Real OpenSim JointReaction on squat:
   - L5/S1 (back joint, inferior pelvis ISB frame and torso frame).
   - L5/S1 qualification (lower bound, ideal coordinate actuator, missing erector spinae).
   - Peak lumbar actuator moment recorded.
   - Inherited OSL-4 qualification classification and reserve shares.
   - Knee: tibiofemoral resultant, and patellofemoral load (patella joint documented).
   - Shoulder: glenohumeral (acromial joint, articulated in model documented).
   - Per-phase mean and peak in N and body weights.
5. Movement Optimizer spine_loads.py L5/S1 aligned cross-check in torso frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.engines.physics_engines.opensim.python.msk_lift_joint_reaction_report import (
    unavailable_joint_reaction_record,
)
from src.engines.physics_engines.opensim.python.msk_lift_joint_reactions import (
    evaluate_lift_joint_reactions,
    static_hold_analytic_bound,
)
from src.shared.python.lifting.pack_audit.names import LIFTS

pytestmark = pytest.mark.unit


def _real_opensim() -> None:
    from tests.helpers.opensim_support import require_opensim

    osim = require_opensim()
    if not hasattr(osim, "__file__"):
        pytest.skip("opensim in sys.modules is a test double, not the real package")


def _require_body() -> None:
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )

    try:
        resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(f"base musculoskeletal model not available: {exc}")


# -------------------------------------------------------------------------
# Pure-Python Unit Tests (No OpenSim Needed)
# -------------------------------------------------------------------------


@pytest.mark.parametrize("lift", [name for name in LIFTS if name != "squat"])
def test_unavailable_lifts_reported_with_reason_never_zero(lift: str) -> None:
    """Non-squat lifts must report status='unavailable' with explicit reason, never fake values."""
    receipt = evaluate_lift_joint_reactions(lift)
    assert receipt["lift"] == lift
    assert receipt["status"] == "unavailable"
    assert "no OSL-4 muscle solve" in receipt["reason"]
    assert receipt["classification"] == "unqualified"
    assert receipt["l5_s1"] is None
    assert receipt["knee_tibiofemoral"] is None
    assert receipt["knee_patellofemoral"] is None
    assert receipt["shoulder_glenohumeral"] is None
    assert receipt["cross_check_movement_optimizer"] is None
    assert receipt["phases"] is None


def test_unavailable_record_structure() -> None:
    """Unavailable record has null metrics and required provenance keys."""
    rec = unavailable_joint_reaction_record(
        "deadlift", "no OSL-4 muscle solve (needs OSL-3 #13041 bundles)"
    )
    assert rec["lift"] == "deadlift"
    assert rec["status"] == "unavailable"
    assert rec["reason"] == "no OSL-4 muscle solve (needs OSL-3 #13041 bundles)"
    assert rec["classification"] == "unqualified"
    for field in (
        "l5_s1",
        "knee_tibiofemoral",
        "knee_patellofemoral",
        "shoulder_glenohumeral",
        "cross_check_movement_optimizer",
        "phases",
    ):
        assert rec[field] is None


def test_unknown_lift_raises() -> None:
    """Unknown lift name raises ValueError."""
    with pytest.raises(ValueError, match="unknown lift"):
        evaluate_lift_joint_reactions("overhead_press")


def test_render_evidence_readme_missing_fields() -> None:
    """README rendering with missing fields prints 'unavailable' and never hard-coded 0.00."""
    from scripts.measure_opensim_lift_joint_reactions import render_evidence_readme

    readme = render_evidence_readme({}, commit="test_commit_hash")
    assert "unavailable" in readme
    assert "0.00" not in readme
    assert "test_commit_hash" in readme

    partial = {"squat": {"lift": "squat"}}
    readme_partial = render_evidence_readme(partial, commit="test_commit_hash")
    assert "unavailable" in readme_partial
    assert "0.00" not in readme_partial


# -------------------------------------------------------------------------
# Integration Tests (With Real OpenSim 4.6)
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def squat_reaction_solution() -> dict[str, Any]:
    _real_opensim()
    _require_body()
    receipt = evaluate_lift_joint_reactions("squat", stride=5)
    assert receipt["status"] == "complete", receipt.get("reason", "")
    return receipt


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_static_hold_analytic_bound() -> None:
    """Static-hold analytic bound verifies default pose self-consistency and real dynamic frame."""
    _real_opensim()
    _require_body()
    bound = static_hold_analytic_bound("squat")
    assert bound["passed"] is True

    # Default pose check (frame and sign self-consistency only)
    def_pose = bound["frame_sign_self_consistency_at_default_pose"]
    assert def_pose["passed"] is True
    w_above = def_pose["w_above_n"]
    comp = def_pose["l5_s1_compression_n"]
    assert math.isclose(comp, w_above, rel_tol=1e-5, abs_tol=1e-4)
    assert def_pose["muscles_crossing_l5s1_count"] == 0
    assert def_pose["sum_muscle_forces_crossing_l5s1_n"] == 0.0

    # Without a solved squat frame the dynamic check is not invented.
    assert bound["squat_min_acceleration_frame"]["passed"] is None
    assert "not evaluated" in bound["squat_min_acceleration_frame"]["reason"]


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_min_acceleration_frame_bound(
    squat_reaction_solution: dict[str, Any],
) -> None:
    """The dynamic bound uses the solved squat frame with the least |qdd|."""
    bound = squat_reaction_solution["static_hold_analytic_bound"]
    min_acc = bound["squat_min_acceleration_frame"]
    lo = squat_reaction_solution["window_start_s"]
    hi = squat_reaction_solution["window_end_s"]
    assert lo <= min_acc["time_s"] <= hi
    assert min_acc["passed"] == (
        min_acc["lower_bound_satisfied"] and min_acc["upper_bound_satisfied"]
    )
    assert min_acc["qdd_norm_rad_s2"] > 0.0
    w_above = float(min_acc["w_above_n"])
    crossing = float(min_acc["sum_muscle_forces_crossing_l5s1_n"])
    c_pelvis = float(min_acc["l5_s1_compression_pelvis_n"])
    c_torso = float(min_acc["l5_s1_compression_torso_n"])
    # Independent oracle: same pelvis-frame inequalities as build_real_squat_bound_record
    # and lift_models.tex OSL-5 (W_above <= Comp <= W_above + sum F_m on the pelvis axis).
    assert min_acc["lower_bound_satisfied"] == (c_pelvis >= w_above - 1e-4)
    assert min_acc["upper_bound_satisfied"] == (
        c_pelvis <= w_above + crossing + 1e-4
    )
    # Torso frame: with Rajagopal, sum F_m = 0 at L5/S1, so Comp_torso <= W_above is the
    # binding static upper bound (documented dynamic squat frame ~1030 N < 1338.59 N).
    assert c_torso <= w_above + crossing + 1e-4
    assert c_torso > 0.0
    # Dynamic bound honestly reports failure when pelvis and torso frames disagree.
    assert min_acc["passed"] is False


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_joint_reactions_computed(
    squat_reaction_solution: dict[str, Any],
) -> None:
    """Squat joint reactions are computed for L5/S1, knee, and shoulder with qualifications."""
    sol = squat_reaction_solution
    assert sol["status"] == "complete"
    assert sol["lift"] == "squat"
    assert sol["acceleration_error_max"] < 5e-6

    # Inherited OSL-4 qualification
    assert sol["classification"] == "unqualified"
    assert sol["mean_reserve_share"] is not None
    assert sol["mean_reserve_share"] > 0.05
    assert sol["peak_reserve_share"] is not None
    assert sol["peak_reserve_share"] > 0.05
    assert sol["reserve_share_threshold"] == 0.05

    # L5/S1
    l5s1 = sol["l5_s1"]
    assert l5s1["joint"] == "back"
    assert "pelvis" in l5s1["frame"].lower()
    assert (
        l5s1["qualification"]
        == "lower bound: no trunk muscles; the lumbar moment is an ideal actuator"
    )
    assert l5s1["compression_mean_n"] > 1000.0
    assert l5s1["compression_peak_n"] > 1000.0
    assert l5s1["compression_mean_bw"] > 1.5
    assert l5s1["ap_shear_mean_n"] != 0.0
    assert l5s1["compression_torso_frame_mean_n"] > 1000.0
    assert l5s1["ap_shear_torso_frame_mean_n"] != 0.0
    assert l5s1["peak_lumbar_actuator_moment_nm"] is not None
    assert l5s1["peak_lumbar_actuator_moment_nm"] > 500.0
    assert "#12735" in l5s1["segmental_spine_reference"]

    # Knee Tibiofemoral
    tf = sol["knee_tibiofemoral"]
    assert tf["joint_r"] == "walker_knee_r"
    assert tf["joint_l"] == "walker_knee_l"
    assert tf["resultant_mean_n"] > 5000.0
    assert tf["resultant_peak_n"] > 8000.0
    assert tf["resultant_mean_bw"] > 10.0

    # Knee Patellofemoral (model has patella joint)
    pf = sol["knee_patellofemoral"]
    assert pf["model_has_patella_joint"] is True
    assert pf["joint_r"] == "patellofemoral_r"
    assert pf["joint_l"] == "patellofemoral_l"
    assert pf["resultant_mean_n"] > 5000.0
    assert pf["resultant_peak_n"] > 8000.0

    # Shoulder Glenohumeral (acromial joint, articulated)
    gh = sol["shoulder_glenohumeral"]
    assert gh["status"] == "complete"
    assert gh["is_articulated"] is True
    assert gh["joint_r"] == "acromial_r"
    assert gh["joint_l"] == "acromial_l"
    assert gh["resultant_mean_n"] > 500.0
    assert gh["resultant_mean_bw"] > 0.5


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_movement_optimizer_crosscheck(
    squat_reaction_solution: dict[str, Any],
) -> None:
    """Movement Optimizer spine_loads.py L5/S1 comparison is evaluated in aligned torso frame."""
    sol = squat_reaction_solution
    assert "cross_check_movement_optimizer" in sol
    cc = sol["cross_check_movement_optimizer"]
    assert cc["model"] == "planar_ergonomic_l5s1_estimate"
    assert "torso frame" in cc["comparison_frame"]
    assert cc["pre_evaluation_tolerance_n"] == 50.0
    assert cc["tolerance_satisfied"] is False
    assert cc["opensim_m_above_kg"] > 100.0
    assert cc["movement_optimizer_m_above_aligned_kg"] > 100.0
    assert cc["compression_mean_n"] > 500.0
    assert cc["shear_mean_n"] > 500.0
    assert cc["aligned_residual_compression_mean_n"] is not None
    assert cc["aligned_residual_compression_mean_n"] > 50.0
    assert cc["aligned_residual_ap_shear_mean_n"] is not None
    assert abs(cc["aligned_residual_ap_shear_mean_n"]) > 50.0
    assert "model_difference_explanation" in cc
    assert len(cc["model_difference_explanation"]) > 20


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_per_phase_metrics(squat_reaction_solution: dict[str, Any]) -> None:
    """Per-phase metrics match OSL-4 phases."""
    sol = squat_reaction_solution
    assert "phases" in sol
    phases = sol["phases"]
    assert len(phases) == 1
    phase_name = list(phases.keys())[0]
    assert "bottom" in phase_name
    bottom = phases[phase_name]
    assert "l5_s1_compression_mean_n" in bottom
    assert "l5_s1_compression_peak_n" in bottom
    assert "knee_tf_resultant_mean_n" in bottom
    assert "knee_pf_resultant_mean_n" in bottom
    assert "shoulder_gh_resultant_mean_n" in bottom


def test_osl4_qualification_is_read_never_defaulted(tmp_path: Path) -> None:
    """The inherited qualification comes from the OSL-4 receipt or is ``None``."""
    from src.engines.physics_engines.opensim.python.msk_lift_joint_reactions import (
        OSL4_RECEIPT,
        _read_osl4_qualification,
    )

    assert OSL4_RECEIPT.is_file()
    missing = _read_osl4_qualification(tmp_path / "absent.json")
    assert missing["mean_reserve_share"] is None
    assert "unreadable" in missing["reason"]
    real = _read_osl4_qualification()
    squat = json.loads(OSL4_RECEIPT.read_text(encoding="utf-8"))["lifts"]["squat"]
    assert real["mean_reserve_share"] == squat["mean_reserve_share"]
    assert real["peak_reserve_share"] == squat["peak_reserve_share"]
