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


_RECEIPT = Path(
    "docs/development/lifting/evidence/opensim_joint_reactions/receipts.json"
)
#: The contact equilibrium solves the vertical root to 1e-12 m (OSL-3), so the
#: supports carry the weight to well under 1e-6 of it.
_SUPPORT_REL_TOL = 1e-6


def _committed_bench() -> dict[str, Any]:
    return json.loads(_RECEIPT.read_text(encoding="utf-8"))["lifts"]["bench_press"]


def test_bench_press_joint_reactions_use_bench_contact() -> None:
    """The bench receipt is complete and loaded by the shared bench contact."""
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        BENCH_EXTERNAL_LOAD_SOURCE,
    )

    bench = _committed_bench()
    assert bench["status"] == "complete", bench.get("reason")
    assert bench["external_load_source"]["bench"] == BENCH_EXTERNAL_LOAD_SOURCE
    assert "back-sphere contact" in BENCH_EXTERNAL_LOAD_SOURCE
    for value in (
        bench["l5_s1"]["compression_mean_n"],
        bench["l5_s1"]["compression_peak_n"],
        bench["knee_tibiofemoral"]["resultant_mean_n"],
        bench["shoulder_glenohumeral"]["resultant_mean_n"],
    ):
        assert math.isfinite(value)
    assert bench["phases"]
    assert bench["timeseries"]["time_s"]
    assert bench["classification"] == _osl4_bench_classification()


def _osl4_bench_classification() -> str:
    from src.engines.physics_engines.opensim.python.msk_lift_joint_reactions import (
        OSL4_RECEIPT,
    )

    lifts = json.loads(OSL4_RECEIPT.read_text(encoding="utf-8"))["lifts"]
    return str(lifts["bench_press"]["classification"])


def test_bench_static_hold_supports_carry_body_and_bar() -> None:
    """Feet plus bench carry (body + bar) x g; the standing bound is not applied."""
    bound = _committed_bench()["static_hold_analytic_bound"]
    support = bound["bench_static_hold_support"]
    assert set(support["support_vertical_n"]) >= {"bench"}
    carried = sum(support["support_vertical_n"].values())
    assert math.isclose(carried, support["expected_weight_n"], rel_tol=_SUPPORT_REL_TOL)
    assert abs(support["relative_error"]) <= _SUPPORT_REL_TOL
    assert support["passed"] is True
    supine = bound["bench_press_min_acceleration_frame"]
    assert supine["passed"] is None
    assert "supine" in supine["reason"]
    assert bound["passed"] is True


def test_bench_press_is_not_hard_coded_unavailable() -> None:
    """A complete OSL-4 bench solve with a bundle runs; no bench special case."""
    from unittest import mock

    from src.engines.physics_engines.opensim.python import (
        msk_lift_joint_reactions as jr,
    )

    qual = jr._read_osl4_qualification("bench_press")
    with mock.patch.object(jr, "is_engine_available", return_value=True):
        assert jr._unavailable_reason("bench_press", qual) is None


@pytest.mark.requires_opensim
def test_source_fingerprint_has_no_host_paths() -> None:
    """Committed receipts carry repo-relative module paths, not host paths."""
    from src.engines.physics_engines.opensim.python.msk_lift_joint_reaction_diagnostics import (
        joint_reaction_source_fingerprint,
    )

    _real_opensim()
    fingerprint = joint_reaction_source_fingerprint()
    for key in (
        "joint_reactions_module",
        "joint_reaction_frame_module",
        "simbody_kinematics_module",
    ):
        assert fingerprint[key].startswith("src/engines/"), fingerprint[key]
    assert not any(str(Path.home()) in str(v) for v in fingerprint.values())


def test_osl4_qualification_reads_all_lifts() -> None:
    """_read_osl4_qualification reads each lift from receipt without hard-coding."""
    from src.engines.physics_engines.opensim.python.msk_lift_joint_reactions import (
        OSL4_RECEIPT,
        _read_osl4_qualification,
    )

    raw = json.loads(OSL4_RECEIPT.read_text(encoding="utf-8"))["lifts"]
    for lift in ("squat", "deadlift", "bench_press", "snatch", "clean_and_jerk"):
        qual = _read_osl4_qualification(lift)
        assert qual["status"] == raw[lift]["status"]
        assert qual["classification"] == raw[lift]["classification"]
        assert qual["mean_reserve_share"] == raw[lift]["mean_reserve_share"]
        assert qual["peak_reserve_share"] == raw[lift]["peak_reserve_share"]
        assert qual["reserve_share_threshold"] == 0.05


def test_unavailable_record_structure() -> None:
    """Unavailable record has null metrics and required provenance keys."""
    rec = unavailable_joint_reaction_record(
        "bench_press", "bench press IK does not pose a supine lifter"
    )
    assert rec["lift"] == "bench_press"
    assert rec["status"] == "unavailable"
    assert rec["reason"] == "bench press IK does not pose a supine lifter"
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


# -------------------------------------------------------------------------
# Integration Tests (With Real OpenSim 4.6)
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lift_reaction_solutions() -> dict[str, dict[str, Any]]:
    _real_opensim()
    _require_body()
    solutions: dict[str, dict[str, Any]] = {}
    for lift in LIFTS:
        receipt = evaluate_lift_joint_reactions(lift, stride=5)
        assert receipt["status"] == "complete", f"{lift}: {receipt.get('reason', '')}"
        solutions[lift] = receipt
    return solutions


@pytest.fixture(scope="module")
def squat_reaction_solution(
    lift_reaction_solutions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return lift_reaction_solutions["squat"]


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", ["squat", "deadlift", "snatch", "clean_and_jerk"])
def test_complete_lifts_joint_reactions_computed(
    lift_reaction_solutions: dict[str, dict[str, Any]],
    lift: str,
) -> None:
    """All lifts with complete OSL-4 records get complete joint reaction loads."""
    sol = lift_reaction_solutions[lift]
    assert sol["status"] == "complete"
    assert sol["lift"] == lift
    # Replayed |udot - qdd| in rad/s^2. Measured 1.14e-5 (squat), 1.23e-5
    # (deadlift), 3.50e-5 (snatch), 2.31e-4 (clean and jerk) against IK
    # accelerations of about 20 rad/s^2: a relative error below 1.2e-5.
    assert sol["acceleration_error_max"] < 5e-4
    assert sol["window_start_s"] == 0.0
    assert sol["window_end_s"] == 0.2
    assert sol["duration_s"] == 0.2
    assert len(sol["timeseries"]["time_s"]) == 41

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
    # Loaded lifts compress L5/S1 by more than one body weight; the squat,
    # with the most trunk lean under the bar, by more than 1.5.
    assert l5s1["compression_mean_n"] > 1000.0
    assert l5s1["compression_peak_n"] > 1000.0
    assert l5s1["compression_mean_bw"] > (1.5 if lift == "squat" else 1.0)
    assert l5s1["ap_shear_mean_n"] != 0.0
    assert l5s1["compression_torso_frame_mean_n"] > 500.0
    assert sol["acceleration_error_max"] is not None
    assert l5s1["ap_shear_torso_frame_mean_n"] != 0.0
    assert l5s1["peak_lumbar_actuator_moment_nm"] is not None
    assert "#12735" in l5s1["segmental_spine_reference"]

    # Knee Tibiofemoral
    tf = sol["knee_tibiofemoral"]
    assert tf["joint_r"] == "walker_knee_r"
    assert tf["joint_l"] == "walker_knee_l"
    assert tf["resultant_mean_n"] > 1000.0
    assert tf["resultant_peak_n"] > 1000.0

    # Knee Patellofemoral (model has patella joint)
    pf = sol["knee_patellofemoral"]
    assert pf["model_has_patella_joint"] is True
    assert pf["joint_r"] == "patellofemoral_r"
    assert pf["joint_l"] == "patellofemoral_l"
    assert pf["resultant_mean_n"] > 500.0
    assert pf["resultant_peak_n"] > 1000.0

    # Shoulder Glenohumeral (acromial joint, articulated)
    gh = sol["shoulder_glenohumeral"]
    assert gh["status"] == "complete"
    assert gh["is_articulated"] is True
    assert gh["joint_r"] == "acromial_r"
    assert gh["joint_l"] == "acromial_l"
    assert gh["resultant_mean_n"] > 100.0


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_bench_press_live_solve_uses_bench_contact(
    lift_reaction_solutions: dict[str, dict[str, Any]],
) -> None:
    """The live bench solve applies the bench contact and balances its supports."""
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        BENCH_EXTERNAL_LOAD_SOURCE,
    )

    bench = lift_reaction_solutions["bench_press"]
    assert bench["external_load_source"]["bench"] == BENCH_EXTERNAL_LOAD_SOURCE
    support = bench["static_hold_analytic_bound"]["bench_static_hold_support"]
    assert abs(support["relative_error"]) <= _SUPPORT_REL_TOL
    assert math.isfinite(bench["l5_s1"]["compression_peak_n"])


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_static_hold_analytic_bound() -> None:
    """Static-hold analytic bound verifies default pose self-consistency."""
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

    # Without a solved frame the dynamic check is not invented.
    assert bound["squat_min_acceleration_frame"]["passed"] is None
    assert "not evaluated" in bound["squat_min_acceleration_frame"]["reason"]


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", ["squat", "deadlift", "snatch", "clean_and_jerk"])
def test_min_acceleration_frame_bound_all_lifts(
    lift_reaction_solutions: dict[str, dict[str, Any]],
    lift: str,
) -> None:
    """The dynamic bound uses the solved frame with the least |qdd|."""
    sol = lift_reaction_solutions[lift]
    bound = sol["static_hold_analytic_bound"]
    frame_key = f"{lift}_min_acceleration_frame"
    assert frame_key in bound
    min_acc = bound[frame_key]
    lo = sol["window_start_s"]
    hi = sol["window_end_s"]
    assert lo <= min_acc["time_s"] <= hi
    assert min_acc["passed"] == (
        min_acc["lower_bound_satisfied"] and min_acc["upper_bound_satisfied"]
    )
    assert min_acc["qdd_norm_rad_s2"] > 0.0
    w_above = float(min_acc["w_above_n"])
    crossing = float(min_acc["sum_muscle_forces_crossing_l5s1_n"])
    c_pelvis = float(min_acc["l5_s1_compression_pelvis_n"])
    c_torso = float(min_acc["l5_s1_compression_torso_n"])
    assert w_above > 1000.0
    assert min_acc["lower_bound_satisfied"] == (c_pelvis >= w_above - 1e-4)
    assert min_acc["upper_bound_satisfied"] == (c_pelvis <= w_above + crossing + 1e-4)
    assert c_torso <= w_above + crossing + 1e-4
    assert c_torso > 0.0
    if lift == "squat":
        # The least-accelerated squat frame still fails the upper bound.
        assert min_acc["passed"] is False


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", ["squat", "deadlift"])
def test_movement_optimizer_crosscheck(
    lift_reaction_solutions: dict[str, dict[str, Any]],
    lift: str,
) -> None:
    """Movement Optimizer spine_loads.py L5/S1 comparison in aligned torso frame for squat and deadlift."""
    sol = lift_reaction_solutions[lift]
    assert "cross_check_movement_optimizer" in sol
    cc = sol["cross_check_movement_optimizer"]
    assert cc["model"] == "planar_ergonomic_l5s1_estimate"
    assert "torso frame" in cc["comparison_frame"]
    assert cc["pre_evaluation_tolerance_n"] == 50.0
    assert cc["opensim_m_above_kg"] > 50.0
    assert cc["movement_optimizer_m_above_aligned_kg"] > 50.0
    assert cc["compression_mean_n"] > 100.0
    assert cc["shear_mean_n"] > 100.0
    assert cc["aligned_residual_compression_mean_n"] is not None
    assert cc["aligned_residual_ap_shear_mean_n"] is not None
    assert "model_difference_explanation" in cc
    assert len(cc["model_difference_explanation"]) > 20


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_per_phase_metrics(
    squat_reaction_solution: dict[str, Any],
) -> None:
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
