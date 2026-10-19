"""Tests for OpenSim barbell lift joint reaction loads (OSL-5, #13043).

Follows TDD red-first against real OpenSim 4.6.
Verifies:
1. Pure-Python record builder: unavailable lifts explicitly report reason, never fake numbers.
2. Static-hold analytic bound: W_above <= L5/S1 compression <= W_above + sum(muscle forces crossing L5/S1).
3. Real OpenSim JointReaction on squat:
   - L5/S1 (back joint, inferior pelvis ISB frame): compression and AP shear.
   - Knee: tibiofemoral resultant, and patellofemoral load (patella joint documented).
   - Shoulder: glenohumeral (acromial joint, articulated in model documented).
   - Per-phase mean and peak in N and body weights.
4. Movement Optimizer spine_loads.py L5/S1 cross-check against squat kinematics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
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
    """Static-hold analytic bound: W_above <= L5/S1 compression <= W_above + sum(F_muscles).

    In the upright static hold, acceleration is zero and trunk is vertical.
    W_above is the gravitational weight of all bodies above L5/S1 (torso, arms, barbell).
    In Rajagopal 80-muscle lower-limb model, sum(F_muscles crossing L5/S1) = 0.
    Thus L5/S1 compression equals W_above within machine precision.
    """
    _real_opensim()
    _require_body()
    bound = static_hold_analytic_bound("squat")
    assert bound["passed"] is True
    w_above = bound["w_above_n"]
    comp = bound["l5_s1_compression_n"]
    assert math.isclose(comp, w_above, rel_tol=1e-5, abs_tol=1e-4)
    assert comp >= w_above - 1e-4
    assert comp <= w_above + bound["sum_muscle_forces_crossing_l5s1_n"] + 1e-4
    assert bound["muscles_crossing_l5s1_count"] == 0


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_joint_reactions_computed(
    squat_reaction_solution: dict[str, Any],
) -> None:
    """Squat joint reactions are computed for L5/S1, knee, and shoulder."""
    sol = squat_reaction_solution
    assert sol["status"] == "complete"
    assert sol["lift"] == "squat"
    assert sol["acceleration_error_max"] < 5e-6

    # L5/S1
    l5s1 = sol["l5_s1"]
    assert l5s1["joint"] == "back"
    assert "pelvis" in l5s1["frame"].lower()
    assert l5s1["compression_mean_n"] > 1000.0
    assert l5s1["compression_peak_n"] > 1000.0
    assert l5s1["compression_mean_bw"] > 1.5
    assert l5s1["ap_shear_mean_n"] != 0.0

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
def test_movement_optimizer_crosscheck(squat_reaction_solution: dict[str, Any]) -> None:
    """Movement Optimizer spine_loads.py L5/S1 comparison is evaluated and explained."""
    sol = squat_reaction_solution
    assert "cross_check_movement_optimizer" in sol
    cc = sol["cross_check_movement_optimizer"]
    assert cc["model"] == "planar_ergonomic_l5s1_estimate"
    assert cc["compression_mean_n"] > 500.0
    assert cc["shear_mean_n"] > 500.0
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
