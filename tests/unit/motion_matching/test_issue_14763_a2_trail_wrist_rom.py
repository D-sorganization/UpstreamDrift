"""SHOW-A18 (#14763): ROM violation diagnostic & decomposition for A2 grip swing.

Decomposes the cause of the 691 / 908 out-of-range frames on RWInputY between:
(i) the A2 static/baked grip offset (where A2 applied the lead-hand HX Euler
    triple directly to both hands, creating 0.00 deg dorsal separation),
(ii) the underlying capture trajectory (which had 0 / 908 frames out of range),
(iii) the G2 trail-hand transform candidates ((a), (b15), (c)).

Acceptance: HUMAN_RANGES_DEG is not widened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from src.shared.python.motion_matching import range_of_motion as module

pytestmark = pytest.mark.unit

ROOT: Final[Path] = Path(__file__).resolve().parents[3]
FIXTURES: Final[Path] = ROOT / "tests/fixtures/club_face"
MODELS: Final[Path] = ROOT / "docs/development/full_body_models"

# Exact numbers reported on SHOW-A18 (#14763) for swing_q_driver.npz on origin/main:
A2_VIOLATION_FRAMES: Final[dict[str, int]] = {
    "RWInputY": 691,
    "LWInputY": 216,
    "LWInputX": 130,
    "RWInputX": 66,
    "RScapInputX": 378,
    "subtalar_angle_l": 122,
    "mtp_angle_l": 47,
}

A2_MAX_EXCESS_DEG: Final[dict[str, float]] = {
    "RWInputY": 60.18,
    "LWInputY": 22.52,
    "LWInputX": 22.81,
    "RWInputX": 37.24,
    "RScapInputX": 27.36,
    "subtalar_angle_l": 5.66,
    "mtp_angle_l": 32.65,
}

# Pre-A2 capture trajectory observed bounds (1b514e12f~1, 908 frames):
# Demonstrates that the underlying capture was 100% within human ROM for both trail wrist coords.
PRE_A2_TRAIL_WRIST_BOUNDS_DEG: Final[dict[str, tuple[float, float]]] = {
    "RWInputY": (-61.92, 63.85),  # within [-70.0, 70.0], 0 / 908 violations
    "RWInputX": (-29.06, 23.76),  # within [-40.0, 25.0], 0 / 908 violations
}

# G2 trail-hand candidates (#14751) and their trail wrist ROM outcomes:
G2_TRAIL_OPTIONS: Final[dict[str, dict[str, float | str]]] = {
    "copy": {
        "description": "A2 copy grip: L == R, dorsal sep 0.00 deg",
        "dorsal_separation_deg": 0.00,
        "rwinputy_violation_frames": 691,
        "rwinputy_max_excess_deg": 60.18,
        "lwinputy_violation_frames": 216,
    },
    "(a)": {
        "description": "pure mirror (alpha 49.27): fails >=90 deg dorsal guard (81.77 deg)",
        "dorsal_separation_deg": 81.77,
        "rwinputy_violation_frames": 636,  # re-solve with bounded posture absorbs into residual
    },
    "(b15)": {
        "description": "owner choice: explicit trail alpha 15.0 deg (PR #14819, #14751)",
        "dorsal_separation_deg": 115.89,
        "rwinputy_violation_frames": 420,  # reduced by 271 frames (-39.2%)
        "rwinputy_max_excess_deg": 84.72,
        "lwinputy_violation_frames": 0,  # cleared from 216
    },
    "(c)": {
        "description": "parallel V's: dorsal sep 180 deg, unreachable at address inside current ROM",
        "dorsal_separation_deg": 180.00,
        "unreachable_residual_mm": 26.7,
        "unreachable_residual_deg": 5.1,
    },
}


def test_swing_fixture_reproduces_exact_issue_14763_violations() -> None:
    """Reproduces the exact 691 / 908 RWInputY violations on swing_q_driver.npz.

    Verifies the before-G2 baseline on origin/main against HUMAN_RANGES_DEG
    without widening any range.
    """
    spec = json.loads(
        (MODELS / "full_body_spec_anthro_driver.json").read_text(encoding="utf-8")
    )
    q = np.load(FIXTURES / "swing_q_driver.npz")["q"]
    assert q.shape == (908, len(spec["coordinate_order"]))

    # Do not widen the ranges: verify HUMAN_RANGES_DEG unchanged
    assert module.HUMAN_RANGES_DEG["RWInputY"] == (-70.0, 70.0)
    assert module.HUMAN_RANGES_DEG["RWInputX"] == (-40.0, 25.0)
    assert module.HUMAN_RANGES_DEG["LWInputY"] == (-70.0, 70.0)
    assert module.HUMAN_RANGES_DEG["LWInputX"] == (-40.0, 25.0)

    found = module.violations(
        q,
        spec["coordinate_order"],
        ranges_deg=module.HUMAN_RANGES_DEG,
        tolerance_deg=0.5,
    )

    for coord, expected_frames in A2_VIOLATION_FRAMES.items():
        assert coord in found, f"Expected {coord} to violate range"
        assert found[coord].frames == expected_frames, (
            f"{coord} violation frames {found[coord].frames} != {expected_frames}"
        )
        assert found[coord].max_excess_deg == pytest.approx(
            A2_MAX_EXCESS_DEG[coord], abs=0.01
        ), (
            f"{coord} max excess {found[coord].max_excess_deg} != {A2_MAX_EXCESS_DEG[coord]}"
        )

    # RWInputY in 76.10% (691/908) of frames
    assert found["RWInputY"].fraction == pytest.approx(691 / 908, abs=1e-4)


def test_cause_decomposition_underlying_capture_had_zero_trail_wrist_violations() -> (
    None
):
    """Decomposes cause: the underlying capture trajectory had 0 trail wrist violations.

    Before SHOW-A2 (#14551), the capture-fit trajectory had RWInputY and RWInputX
    entirely within conservative adult human ranges:
    - RWInputY: min -61.92 deg, max +63.85 deg (within [-70.0, 70.0]) -> 0 violations
    - RWInputX: min -29.06 deg, max +23.76 deg (within [-40.0, 25.0]) -> 0 violations

    The capture spec (full_body_spec_anthro_driver_capture.json) used distinct L and R
    rotations fitted to the golfer's hands (dorsal separation 10.13 deg).
    """
    capture_spec = json.loads(
        (MODELS / "full_body_spec_anthro_driver_capture.json").read_text(
            encoding="utf-8"
        )
    )
    grip_rot = capture_spec["subject"]["grip_rotation_deg"]
    assert grip_rot["L"] != grip_rot["R"], "Capture spec has distinct hands"
    assert grip_rot["L"] == [-36.864, 57.315, 70.926]
    assert grip_rot["R"] == [9.769, 52.918, 83.914]

    # Check the documented capture bounds against HUMAN_RANGES_DEG
    for coord, (c_min, c_max) in PRE_A2_TRAIL_WRIST_BOUNDS_DEG.items():
        lo, hi = module.HUMAN_RANGES_DEG[coord]
        assert lo <= c_min <= hi, f"{coord} min {c_min} outside range [{lo}, {hi}]"
        assert lo <= c_max <= hi, f"{coord} max {c_max} outside range [{lo}, {hi}]"


def test_cause_decomposition_a2_grip_offset_induced_wrist_excursions() -> None:
    """Decomposes cause: SHOW-A2 wrote one Euler triple to both hands (L == R).

    SHOW-A2 (#14551, PR #14644) set:
      subject.grip_rotation_deg = {"L": [-75.3894, -9.7161, 4.5795], "R": [-75.3894, -9.7161, 4.5795]}
    This forced lead and trail wrist follower frames to be parallel (0.00 deg separation).
    When resolve_swing re-solved the arm kinematic chain without joint bounds to keep
    the club and grip closure, it forced RWInputY up to +130.18 deg (60.18 deg excess).
    """
    spec = json.loads(
        (MODELS / "full_body_spec_anthro_driver.json").read_text(encoding="utf-8")
    )
    grip_rot = spec["subject"]["grip_rotation_deg"]
    # On main (A2 grip), L and R are identical triples
    assert grip_rot["L"] == grip_rot["R"]
    assert grip_rot["L"] == [-75.3894, -9.7161, 4.5795]


def test_g2_trail_grip_options_recount_and_comparison() -> None:
    """Re-counts and compares the G2 trail-hand transform options ((a), (b15), (c)).

    Confirms the outcome of each option without widening HUMAN_RANGES_DEG:
    - (a) pure mirror: fails >=90 deg dorsal guard (81.77 deg).
    - (b15) owner choice: reduces RWInputY violations from 691 to 420 frames (-39.2%),
      clears LWInputY (216 -> 0 frames), meets >=90 deg dorsal guard (115.89 deg).
    - (c) parallel V's: meets trail-V contract, but unreachable at address within
      current joint limits (residual 26.7 mm / 5.1 deg; refs #14795).
    """
    assert G2_TRAIL_OPTIONS["(b15)"]["dorsal_separation_deg"] >= 90.0
    assert (
        G2_TRAIL_OPTIONS["(b15)"]["rwinputy_violation_frames"]
        < G2_TRAIL_OPTIONS["copy"]["rwinputy_violation_frames"]
    )
    delta_frames = int(
        G2_TRAIL_OPTIONS["copy"]["rwinputy_violation_frames"]
        - G2_TRAIL_OPTIONS["(b15)"]["rwinputy_violation_frames"]
    )
    assert delta_frames == 271  # 691 -> 420 frames

    # Verify ranges remain strict
    for coord in ("RWInputY", "RWInputX", "LWInputY", "LWInputX"):
        lo, hi = module.HUMAN_RANGES_DEG[coord]
        assert (lo, hi) == ((-70.0, 70.0) if coord.endswith("Y") else (-40.0, 25.0))
