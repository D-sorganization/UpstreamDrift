"""Tests for OpenSim muscle forces and activations for barbell lifts (OSL-4, #13042).

Follows TDD red-first against real OpenSim 4.6.
Verifies:
1. Classification logic: qualified only if reserve share < 5% (0.05).
2. Major muscle groups table and Rajagopal model citation.
3. Unavailable lifts are explicitly reported with reason, never zero.
4. Squat: muscle moment + reserve actuator moment matches ID joint moment.
5. Reserve share reporting and qualification for squat.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
RECEIPTS_PATH = (
    _REPO / "docs/development/lifting/evidence/opensim_muscles/receipts.json"
)
IK_ID_RECEIPTS_PATH = (
    _REPO / "docs/development/lifting/evidence/opensim_ik_id/receipts.json"
)

from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
    GROUP_TO_RAJAGOPAL_MUSCLES,
    MAJOR_MUSCLE_GROUPS,
    RAJAGOPAL_MODEL_CITATION,
    classify_lift_reserves,
    detect_motion_phases,
    evaluate_lift_muscles,
    reserve_share,
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
        pytest.skip(f"muscle-bearing body not available: {exc}")


# -------------------------------------------------------------------------
# Unit Tests (Without OpenSim)
# -------------------------------------------------------------------------


def test_reserve_share_zero_demand() -> None:
    """Zero demand and zero reserves give 0.0 reserve share."""
    tau = np.zeros(5)
    reserves = np.zeros(5)
    assert reserve_share(tau, reserves) == 0.0


def test_reserve_share_zero_reserves() -> None:
    """Non-zero demand with zero reserves gives 0.0 reserve share."""
    tau = np.array([50.0, -100.0, 75.0])
    reserves = np.zeros(3)
    assert reserve_share(tau, reserves) == 0.0


def test_reserve_share_positive() -> None:
    """Calculates sum(|r|) / sum(|tau|)."""
    tau = np.array([100.0, -100.0])  # sum |tau| = 200
    reserves = np.array([2.0, -3.0])  # sum |r| = 5
    assert reserve_share(tau, reserves) == pytest.approx(5.0 / 200.0)


def test_reserve_share_infinite_when_demand_zero_and_reserves_present() -> None:
    """Reserves with zero demand returns infinity."""
    tau = np.zeros(3)
    reserves = np.array([1.0, 0.0, 0.0])
    assert math.isinf(reserve_share(tau, reserves))


def test_classification_logic() -> None:
    """Qualified only when reserve share < 5% (0.05) AND residual gate passes."""
    assert classify_lift_reserves(0.0) == "qualified"
    assert classify_lift_reserves(0.049) == "qualified"
    assert classify_lift_reserves(0.05) == "unqualified"
    assert classify_lift_reserves(0.051) == "unqualified"
    assert classify_lift_reserves(0.568) == "unqualified"
    # Even with low reserve share, failing residual gate disqualifies the lift
    assert classify_lift_reserves(0.01, residual_gate_passed=False) == "unqualified"
    assert classify_lift_reserves(0.01, residual_gate_passed=True) == "qualified"


def test_major_muscle_groups_table() -> None:
    """All 7 required major muscle groups are mapped and documented with citation."""
    expected_groups = (
        "quadriceps",
        "gluteals",
        "hamstrings",
        "erector spinae",
        "deltoids",
        "trapezius",
        "triceps",
    )
    assert set(MAJOR_MUSCLE_GROUPS) == set(expected_groups)
    assert set(GROUP_TO_RAJAGOPAL_MUSCLES.keys()) == set(expected_groups)

    # Lower limb groups in Rajagopal 80-muscle model
    quads = GROUP_TO_RAJAGOPAL_MUSCLES["quadriceps"]
    assert len(quads) == 8  # 4 bilateral pairs (recfem, vasint, vaslat, vasmed)
    assert any("vaslat" in m for m in quads)

    glutes = GROUP_TO_RAJAGOPAL_MUSCLES["gluteals"]
    assert len(glutes) == 18  # 9 bilateral pairs (glmax1-3, glmed1-3, glmin1-3)
    assert any("glmax1" in m for m in glutes)

    hams = GROUP_TO_RAJAGOPAL_MUSCLES["hamstrings"]
    assert len(hams) == 8  # 4 bilateral pairs (bflh, bfsh, semimem, semiten)
    assert any("semimem" in m for m in hams)

    # Upper body / trunk groups have no muscles in Rajagopal 80-muscle lower-limb model
    for trunk_arm_group in ("erector spinae", "deltoids", "trapezius", "triceps"):
        assert GROUP_TO_RAJAGOPAL_MUSCLES[trunk_arm_group] == ()

    assert "Rajagopal" in RAJAGOPAL_MODEL_CITATION
    assert "2016" in RAJAGOPAL_MODEL_CITATION


def test_deep_squat_window_never_labeled_standing() -> None:
    """A partial engine window covering deep squat (>45 deg knee flexion) is never labeled standing.

    Verifies OSL-4 finding 1: deep squat bottom (knee ~110 deg) must report
    'bottom (engine window ...)' and never falsely label standing, descent, or lockout.
    """
    times = np.linspace(0.0, 0.2, 41)
    coord_names = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]
    # Deep squat: hip at 120 deg (2.094 rad), knees at ~110 deg (1.92 rad)
    q = np.zeros((41, 4))
    q[:, 0] = np.radians(120.0)
    q[:, 1] = np.radians(120.0)
    q[:, 2] = np.radians(110.0)
    q[:, 3] = np.radians(109.5)

    segments = detect_motion_phases("squat", times, coord_names, q)
    assert len(segments) == 1
    seg_name, t0, t1 = segments[0]
    assert "bottom" in seg_name
    assert "engine window 0.0-0.2 s" in seg_name
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(0.2)

    # Ensure nonexistent full-lift phases are not present
    all_names = [s[0].lower() for s in segments]
    for disallowed in ("standing", "descent", "lockout"):
        assert not any(disallowed in name for name in all_names)


def test_bench_press_unavailable_due_to_supine_ik_reason() -> None:
    """Bench press IK does not pose a supine lifter, so it stays unavailable with that reason."""
    receipt = evaluate_lift_muscles("bench_press")
    assert receipt["status"] == "unavailable"
    assert "supine" in receipt["reason"].lower()
    assert receipt["forces"] is None
    assert receipt["activations"] is None
    assert receipt["reserve_share"] is None
    assert receipt["classification"] == "unqualified"


def test_deadlift_window_never_labeled_with_nonexistent_phases() -> None:
    """A partial engine window covering deadlift pull never falsely labels setup or lockout."""
    times = np.linspace(0.0, 0.2, 41)
    coord_names = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]
    q = np.zeros((41, 4))
    q[:, 0] = np.radians(90.0)
    q[:, 1] = np.radians(90.0)
    q[:, 2] = np.radians(70.0)
    q[:, 3] = np.radians(70.0)

    segments = detect_motion_phases("deadlift", times, coord_names, q)
    assert len(segments) == 1
    seg_name, t0, t1 = segments[0]
    assert "pull" in seg_name
    assert "engine window 0.0-0.2 s" in seg_name
    all_names = [s[0].lower() for s in segments]
    for disallowed in ("setup", "lockout", "descent"):
        assert not any(disallowed in name for name in all_names)


def test_snatch_window_never_labeled_with_nonexistent_phases() -> None:
    """A partial engine window covering snatch pull never falsely labels turnover or catch."""
    times = np.linspace(0.0, 0.2, 41)
    coord_names = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]
    q = np.zeros((41, 4))
    q[:, 0] = np.radians(85.0)
    q[:, 1] = np.radians(85.0)
    q[:, 2] = np.radians(50.0)
    q[:, 3] = np.radians(50.0)

    segments = detect_motion_phases("snatch", times, coord_names, q)
    assert len(segments) == 1
    seg_name, t0, t1 = segments[0]
    assert "pull" in seg_name
    assert "engine window 0.0-0.2 s" in seg_name
    all_names = [s[0].lower() for s in segments]
    for disallowed in ("turnover", "catch", "recovery"):
        assert not any(disallowed in name for name in all_names)


def test_clean_and_jerk_window_never_labeled_with_nonexistent_phases() -> None:
    """A partial engine window covering clean pull never falsely labels jerk phases."""
    times = np.linspace(0.0, 0.2, 41)
    coord_names = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]
    q = np.zeros((41, 4))
    q[:, 0] = np.radians(95.0)
    q[:, 1] = np.radians(95.0)
    q[:, 2] = np.radians(50.0)
    q[:, 3] = np.radians(50.0)

    segments = detect_motion_phases("clean_and_jerk", times, coord_names, q)
    assert len(segments) == 1
    seg_name, t0, t1 = segments[0]
    assert "pull" in seg_name
    assert "engine window 0.0-0.2 s" in seg_name
    all_names = [s[0].lower() for s in segments]
    for disallowed in (
        "rack_catch",
        "rack_recovery",
        "jerk_dip_drive",
        "overhead_catch",
        "overhead_recovery",
    ):
        assert not any(disallowed in name for name in all_names)


def test_muscle_receipts_recorded_for_all_lifts() -> None:
    """Verify docs/development/lifting/evidence/opensim_muscles/receipts.json."""
    if not RECEIPTS_PATH.is_file():
        pytest.skip(f"receipts file not found: {RECEIPTS_PATH}")
    data = json.loads(RECEIPTS_PATH.read_text(encoding="utf-8"))
    lifts = data["lifts"]
    assert set(lifts.keys()) == set(LIFTS)
    assert lifts["bench_press"]["status"] == "unavailable"
    assert "supine" in lifts["bench_press"]["reason"].lower()
    for lift in ("squat", "deadlift", "snatch", "clean_and_jerk"):
        entry = lifts[lift]
        assert entry["status"] == "complete", f"{lift}: {entry.get('reason')}"
        assert entry["classification"] == "unqualified"
        assert entry["reserve_share"] is not None and entry["reserve_share"] > 0.05
        assert "residual_status" in entry


def test_muscle_receipts_carry_measured_ik_id_residuals() -> None:
    """Every complete record must carry IK/ID residual status matching IK/ID receipts."""
    if not RECEIPTS_PATH.is_file() or not IK_ID_RECEIPTS_PATH.is_file():
        pytest.skip("receipts files not found")
    m_data = json.loads(RECEIPTS_PATH.read_text(encoding="utf-8"))["lifts"]
    ik_data = json.loads(IK_ID_RECEIPTS_PATH.read_text(encoding="utf-8"))["lifts"]
    for lift in ("squat", "deadlift", "snatch", "clean_and_jerk"):
        m_rec = m_data[lift]
        assert m_rec["status"] == "complete"
        res_status = m_rec["residual_status"]
        ik_id_id = ik_data[lift]["ik_id"]["id"]
        gate = ik_id_id["residual_gate"]
        residuals = ik_id_id["residuals"]
        peak_f = max(
            residuals["pelvis_tx"]["max_abs"],
            residuals["pelvis_ty"]["max_abs"],
            residuals["pelvis_tz"]["max_abs"],
        )
        peak_m = max(
            residuals["pelvis_tilt"]["max_abs"],
            residuals["pelvis_list"]["max_abs"],
            residuals["pelvis_rotation"]["max_abs"],
        )
        assert res_status["forces_within"] == gate["forces_within"]
        assert res_status["moments_within"] == gate["moments_within"]
        assert res_status["measured_pelvis_residual_force_n"] == pytest.approx(
            peak_f, abs=1e-3
        )
        assert res_status["hicks_force_limit_n"] == pytest.approx(
            gate["max_force_n"], abs=1e-3
        )
        assert res_status["measured_pelvis_residual_moment_nm"] == pytest.approx(
            peak_m, abs=1e-3
        )
        assert res_status["hicks_moment_limit_nm"] == pytest.approx(
            gate["max_moment_nm"], abs=1e-3
        )
        if lift != "squat":
            assert res_status["forces_within"] is False
            assert res_status["moments_within"] is False
        else:
            assert res_status["forces_within"] is True
            assert res_status["moments_within"] is True


# -------------------------------------------------------------------------
# Integration Tests (With Real OpenSim 4.6)
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def squat_muscle_solution() -> dict:
    _real_opensim()
    _require_body()
    receipt = evaluate_lift_muscles("squat", stride=5, run_moco=False)
    assert receipt["status"] == "complete", receipt.get("reason", "")
    return receipt


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_muscle_moments_match_joint_moments_within_reserve(
    squat_muscle_solution: dict,
) -> None:
    """Muscle moment sum plus reserve moments equals ID joint moments.

    For each coordinate: tau_muscle + tau_reserve == tau_id.
    """
    sol = squat_muscle_solution
    assert sol["status"] == "complete"
    assert "moment_balance_max_error_nm" in sol
    assert sol["moment_balance_max_error_nm"] < 1e-4

    # Verify per-group summary
    groups = sol["major_groups"]
    for group_name in ("quadriceps", "gluteals", "hamstrings"):
        assert group_name in groups
        assert groups[group_name]["peak_activation"] > 0.0
        assert groups[group_name]["mean_activation"] > 0.0
        assert groups[group_name]["peak_force_n"] > 0.0

    # Verify trunk / upper body groups are documented as absent in Rajagopal
    for group_name in ("erector spinae", "deltoids", "trapezius", "triceps"):
        assert group_name in groups
        assert groups[group_name]["n_muscles"] == 0
        assert groups[group_name]["peak_activation"] is None

    # Reserve share and classification
    share = sol["reserve_share"]
    assert share is not None
    assert math.isfinite(share)
    assert share > 0.05
    assert sol["classification"] == "unqualified"

    # Verify honest phase labeling
    assert "phases" in sol
    phase_names = list(sol["phases"].keys())
    assert len(phase_names) == 1
    assert "bottom" in phase_names[0]
    assert "standing" not in sol["phases"]
    assert "descent" not in sol["phases"]
    assert "lockout" not in sol["phases"]

    # Verify window metadata and start pose
    assert sol["window_start_s"] == pytest.approx(0.0)
    assert sol["window_end_s"] == pytest.approx(0.2)
    assert sol["duration_s"] == pytest.approx(0.2)
    start_pose = sol["start_pose"]
    assert start_pose["hip_flexion_r_deg"] == pytest.approx(120.0, abs=0.1)
    assert start_pose["hip_flexion_l_deg"] == pytest.approx(120.0, abs=0.1)
    assert start_pose["knee_angle_r_deg"] > 100.0

    # Verify coordinate breakdown and asymmetry
    assert "coordinate_breakdown" in sol
    assert "hip_flexion_r" in sol["coordinate_breakdown"]
    assert "primary_reserve_drivers" in sol
    assert "hip_flexion_l" in sol["primary_reserve_drivers"]

    # Verify bound pinning flags
    assert "bound_violations_or_pinned" in sol
    pinned = sol["bound_violations_or_pinned"]
    assert pinned["has_pinned_coordinates"] is True
    assert "hip_flexion_r" in pinned["pinned_coordinates"]
    assert "hip_flexion_l" in pinned["pinned_coordinates"]

    # Verify asymmetry explanation
    assert "hip_demand_asymmetry" in sol
    asym = sol["hip_demand_asymmetry"]
    assert asym["shear_differential_n"] > 300.0
    assert "cause" in asym

    # Moco status is honestly recorded
    moco = sol["moco"]
    assert "solver" in moco
    assert isinstance(moco["converged"], bool)


def test_hip_demand_asymmetry_cause_quotes_the_measured_values() -> None:
    """The cause text is built from this run's numbers, never a stale literal."""
    from types import SimpleNamespace

    from src.engines.physics_engines.opensim.python.msk_lift_muscle_report import (
        compute_hip_demand_asymmetry,
    )

    coords = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]
    tau = np.array([[-111.11, -222.22, -33.33, -44.44]])
    targets = SimpleNamespace(
        grf_force_n={
            "left": np.array([[12.5, 700.25, 0.0]]),
            "right": np.array([[-7.5, 650.75, 0.0]]),
        }
    )
    record = compute_hip_demand_asymmetry(tau, coords, targets)
    assert record["shear_differential_n"] == pytest.approx(20.0)
    for value in ("-111.11", "-222.22", "-33.33", "-44.44", "+12.50", "-7.50"):
        assert value in record["cause"]
    assert "-434.68" not in record["cause"]


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", ["deadlift", "snatch", "clean_and_jerk"])
def test_live_remaining_lifts_muscle_solutions(lift: str) -> None:
    """Live static optimization solve for deadlift, snatch, and clean & jerk."""
    _real_opensim()
    _require_body()
    receipt = evaluate_lift_muscles(lift, stride=5, run_moco=False)
    assert receipt["status"] == "complete", receipt.get("reason", "")
    assert receipt["classification"] == "unqualified"
    assert receipt["reserve_share"] is not None and receipt["reserve_share"] > 0.05
    assert receipt["moment_balance_max_error_nm"] < 1e-4
    assert "residual_status" in receipt
    assert receipt["residual_status"]["forces_within"] is False
    assert receipt["residual_status"]["moments_within"] is False
    phase_names = list(receipt["phases"].keys())
    assert len(phase_names) == 1
    assert "pull" in phase_names[0]
