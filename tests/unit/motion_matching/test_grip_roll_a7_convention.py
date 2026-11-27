"""Contract test for SHOW-A7 grip roll convention (#14731).

Pins the owner-defined grip roll convention from Dieter (00:25Z):
- 0 deg = back of wrist (dorsal axis) points along e_v (up, on top of shaft at 12 o'clock).
- + = counterclockwise on the face-on clock toward target along e_h.
- The pre-A7 flagship driver model (which read 15 deg under the old palm-vertical zero)
  reads 75 deg (+- 1 deg) under this convention, confirming the owner's measurement
  that the wrists were ~75 deg from vertical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.shared.python.motion_matching.grip_roll import (
    GripPoseState,
    address_grip_roll_deg,
    grip_pose_state_from_spec,
    reference_axes,
)

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
FLAGSHIP_CLUB_ROTATION = Rotation.from_matrix(
    [[-0.5417, -0.7177, -0.4376], [0.6055, 0.0279, -0.7953], [0.5831, -0.6958, 0.4195]]
).as_matrix()


def test_a7_zero_is_back_of_wrist_up_on_top_of_shaft() -> None:
    """A dorsal axis pointing along e_v (up) must read exactly 0 deg."""
    shaft = FLAGSHIP_CLUB_ROTATION @ np.array([0.0, 1.0, 0.0])
    e_h, e_v = reference_axes(shaft, TargetFrame())

    # Build a state where lead normal in world aligns with e_v (12 o'clock)
    # normal = -(rot @ dorsal_club) => dorsal_club = -(rot.T @ e_v)
    dorsal_lead = -(FLAGSHIP_CLUB_ROTATION.T @ e_v)
    state = GripPoseState(
        club_rotation=FLAGSHIP_CLUB_ROTATION,
        lead_dorsal_club=dorsal_lead,
        trail_dorsal_club=dorsal_lead,
    )
    lead_roll, _ = address_grip_roll_deg(state)
    assert lead_roll == pytest.approx(0.0, abs=1e-5)


def test_a7_sign_is_positive_counterclockwise_toward_target() -> None:
    """Rotating from 12 o'clock toward target (+e_h) must produce positive roll."""
    shaft = FLAGSHIP_CLUB_ROTATION @ np.array([0.0, 1.0, 0.0])
    e_h, e_v = reference_axes(shaft, TargetFrame())

    # 15 deg CCW from e_v toward e_h:
    angle_rad = np.radians(15.0)
    normal = np.cos(angle_rad) * e_v + np.sin(angle_rad) * e_h
    dorsal_lead = -(FLAGSHIP_CLUB_ROTATION.T @ normal)
    state = GripPoseState(
        club_rotation=FLAGSHIP_CLUB_ROTATION,
        lead_dorsal_club=dorsal_lead,
        trail_dorsal_club=dorsal_lead,
    )
    lead_roll, _ = address_grip_roll_deg(state)
    assert lead_roll == pytest.approx(15.0, abs=1e-5)


def test_owner_check_pre_a7_driver_reads_75_degrees() -> None:
    """The pre-A7 flagship driver model must read ~75 deg on the A7 clock."""
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    state = grip_pose_state_from_spec(spec, FLAGSHIP_CLUB_ROTATION)
    lead_roll, _ = address_grip_roll_deg(state)
    assert lead_roll == pytest.approx(75.0, abs=1.5)
