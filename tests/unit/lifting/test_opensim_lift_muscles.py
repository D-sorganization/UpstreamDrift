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

import math
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
    GROUP_TO_RAJAGOPAL_MUSCLES,
    MAJOR_MUSCLE_GROUPS,
    RAJAGOPAL_MODEL_CITATION,
    classify_lift_reserves,
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
    """Qualified only when reserve share < 5% (0.05)."""
    assert classify_lift_reserves(0.0) == "qualified"
    assert classify_lift_reserves(0.049) == "qualified"
    assert classify_lift_reserves(0.05) == "unqualified"
    assert classify_lift_reserves(0.051) == "unqualified"
    assert classify_lift_reserves(0.568) == "unqualified"


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


@pytest.mark.parametrize("lift", [lift for lift in LIFTS if lift != "squat"])
def test_unavailable_lifts_reported_with_reason_never_zero(lift: str) -> None:
    """Non-squat lifts must report status='unavailable' with reason, never fake numbers."""
    receipt = evaluate_lift_muscles(lift)
    assert receipt["status"] == "unavailable"
    assert "reason" in receipt and receipt["reason"]
    assert receipt["forces"] is None
    assert receipt["activations"] is None
    assert receipt["reserve_share"] is None
    assert receipt["classification"] == "unqualified"


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

    # Moco status is honestly recorded
    moco = sol["moco"]
    assert "solver" in moco
    assert isinstance(moco["converged"], bool)
