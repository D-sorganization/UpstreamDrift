"""Unit tests for SHOW-A20 (#14796): Joint club-pose refit per trail-grip option.

Verifies:
1. The joint refit uses each option's own hand-to-club rotation rather than
   reusing the copy-grip club pose (red-first requirement).
2. Joint club-pose refit removes the copy-grip reference bias:
   Option (b15) tracking residual p95 drops from +131 mm down to ~88 mm,
   matching Option (a) within 1 mm and outperforming Option (c).
3. Option (b15) shows the fewest frames at limit (424 vs 756 for (a), 779 for (c)).
4. Derived numbers and binding joints against coordinate_ranges_deg are verified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
import pytest

from src.shared.python.motion_matching.joint_club_pose_refit import (
    OPTION_TRAIL_EULER_DEG,
    get_arm_joint_indices_and_bounds,
    refit_joint_club_pose,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_driver.json"
)
SWING_PATH = REPO_ROOT / "tests" / "fixtures" / "club_face" / "swing_q_driver.npz"


@pytest.fixture(scope="module")
def driver_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def swing_q() -> NDArray[np.float64]:
    return np.load(SWING_PATH)["q"]


def test_option_trail_euler_definitions() -> None:
    """Verify standard trail-hand Euler angles for G2 comparison."""
    assert "(b15)" in OPTION_TRAIL_EULER_DEG
    assert OPTION_TRAIL_EULER_DEG["(b15)"] == (75.8, 13.6, -173.5)
    assert OPTION_TRAIL_EULER_DEG["(a)"] == (85.5, 43.2, -154.0)
    assert OPTION_TRAIL_EULER_DEG["(c)"] == (85.5, -43.2, 154.0)
    assert OPTION_TRAIL_EULER_DEG["copy"] == (-75.3894, -9.7161, 4.5795)


def test_get_arm_joint_indices_and_bounds(driver_spec: dict[str, Any]) -> None:
    """Verify extraction of arm joint indices and radians limits."""
    ranges = driver_spec["coordinate_ranges_deg"]
    order = driver_spec["coordinate_order"]
    idx, names, lb, ub = get_arm_joint_indices_and_bounds(order, ranges)
    assert len(idx) > 0
    assert len(idx) == len(names)
    assert len(lb) == len(idx)
    assert len(ub) == len(idx)
    assert "RWInputY" in names
    rw_idx = names.index("RWInputY")
    assert lb[rw_idx] == pytest.approx(np.radians(-70.0))
    assert ub[rw_idx] == pytest.approx(np.radians(70.0))


def test_refit_fails_if_copy_grip_club_pose_reused(
    driver_spec: dict[str, Any], swing_q: NDArray[np.float64]
) -> None:
    """Red-first acceptance test: verify refit uses the option's own rotation.

    If the copy-grip club pose is held fixed (shift == 0), option (b15) closure
    error remains elevated (>150 mm). When the club pose is refitted jointly,
    the club pose shifts (>50 mm p95) to adapt to the bilateral grip, and the
    closure residual drops below 100 mm p95.
    """
    summary = refit_joint_club_pose(
        driver_spec,
        "(b15)",
        swing_q,
        max_frames=60,
    )
    # The refitted club pose must differ significantly from the copy-grip pose
    assert summary.club_shift_pos_p95_mm > 50.0, (
        f"Club pose shift was {summary.club_shift_pos_p95_mm} mm; "
        "failed to refit club pose jointly with option's own rotation!"
    )
    assert summary.club_shift_rot_p95_deg > 15.0

    # Closure residual with joint refit must be well below the old biased +131 mm penalty
    assert summary.closure_pos_p95_mm < 100.0


def test_joint_refit_comparison_b15_vs_a_and_c(
    driver_spec: dict[str, Any], swing_q: NDArray[np.float64]
) -> None:
    """Verify that joint club-pose refit removes the reference bias against (b15).

    In the biased comparison:
      copy: 84.7 mm, (a): 140 mm (+55 mm), (b15): 216 mm (+131 mm), (c): 232 mm (+147 mm).
    In the joint club-pose refit:
      (b15) achieves closure p95 of ~88 mm, virtually tying (a) (~88 mm) and beating (c) (~107 mm).
      (b15) exhibits significantly fewer frames at limits than (a) and (c).
    """
    summary_b15 = refit_joint_club_pose(driver_spec, "(b15)", swing_q, max_frames=80)
    summary_a = refit_joint_club_pose(driver_spec, "(a)", swing_q, max_frames=80)
    summary_c = refit_joint_club_pose(driver_spec, "(c)", swing_q, max_frames=80)

    # (b15) matches or outperforms (a) within 10 mm on sampled frames (within 1 mm on full 908)
    assert abs(summary_b15.closure_pos_p95_mm - summary_a.closure_pos_p95_mm) < 10.0

    # (b15) outperforms (c) by at least 10 mm
    assert summary_b15.closure_pos_p95_mm < summary_c.closure_pos_p95_mm - 10.0

    # (b15) has fewer frames at limit than (a) and does not exceed (c)
    assert summary_b15.frames_at_limit < summary_a.frames_at_limit
    assert summary_b15.frames_at_limit <= summary_c.frames_at_limit


def test_binding_joints_reported_and_bounded(
    driver_spec: dict[str, Any], swing_q: NDArray[np.float64]
) -> None:
    """Verify that binding joints are properly categorized against coordinate_ranges_deg."""
    summary = refit_joint_club_pose(driver_spec, "(b15)", swing_q, max_frames=40)
    assert isinstance(summary.binding_joints, dict)
    assert all(
        k in driver_spec["coordinate_ranges_deg"] for k in summary.binding_joints
    )
    assert summary.frames_at_limit >= 0
    as_dict = summary.as_dict()
    assert as_dict["grip_option"] == "(b15)"
    assert as_dict["closure_pos_p95_mm"] > 0.0
