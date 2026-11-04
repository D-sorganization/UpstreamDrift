"""Golf-language address grip roll (SHOW-A1, #14550).

0 deg = hand vertical face-on, + = strong. Synthetic hands are built at a known
roll about a tilted shaft, so the recovered value is checked against the input,
not against the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.shared.python.motion_matching.grip_roll import (
    GripPoseState,
    address_grip_roll_deg,
    reference_axes,
)

# A realistic address shaft: from the hands down and toward the ball, in the
# face-on plane (x-z), golfer facing -x, target along -y (UD default frame).
SHAFT = np.array([-0.718, 0.028, -0.696])
SHAFT = SHAFT / np.linalg.norm(SHAFT)


def _club_rotation(shaft: np.ndarray = SHAFT, spin_deg: float = 0.0) -> np.ndarray:
    """World <- club frame with column 1 (club y) along ``shaft``."""
    y = shaft / np.linalg.norm(shaft)
    x = np.cross(y, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    base = np.column_stack([x, y, z])
    spin = Rotation.from_rotvec(np.radians(spin_deg) * y).as_matrix()
    return spin @ base


def _state(lead_deg: float, trail_deg: float, spin_deg: float = 0.0) -> GripPoseState:
    """Hands whose back-of-hand (lead) / palm (trail) normals sit at the roll."""
    rot = _club_rotation(spin_deg=spin_deg)
    e_h, e_v = reference_axes(SHAFT, TargetFrame())

    def dorsal_club(roll_deg: float) -> np.ndarray:
        r = np.radians(roll_deg)
        normal = np.cos(r) * e_h + np.sin(r) * e_v  # world, perpendicular to shaft
        return -(rot.T @ normal)  # repo convention: normal = -dorsal axis

    return GripPoseState(
        club_rotation=rot,
        lead_dorsal_club=dorsal_club(lead_deg),
        trail_dorsal_club=dorsal_club(trail_deg),
    )


@pytest.mark.parametrize("deg", [0.0, 15.0, 45.0, -15.0, -45.0, 90.0, 99.4])
def test_recovers_synthetic_roll_within_half_degree(deg: float) -> None:
    lead, trail = address_grip_roll_deg(_state(deg, -deg / 2.0))
    assert lead == pytest.approx(deg, abs=0.5)
    assert trail == pytest.approx(-deg / 2.0, abs=0.5)


def test_strong_is_positive_back_of_lead_hand_toward_the_sky() -> None:
    state = _state(30.0, 30.0)
    e_h, e_v = reference_axes(SHAFT, TargetFrame())
    normal_world = -(state.club_rotation @ state.lead_dorsal_club)
    assert normal_world @ e_v > 0.0  # up
    assert address_grip_roll_deg(state)[0] > 0.0


def test_reference_axes_are_horizontal_and_orthonormal() -> None:
    e_h, e_v = reference_axes(SHAFT, TargetFrame())
    assert abs(e_h[2]) < 1e-12  # horizontal
    assert e_h @ SHAFT == pytest.approx(0.0, abs=1e-12)
    assert e_v @ SHAFT == pytest.approx(0.0, abs=1e-12)
    assert e_h @ e_v == pytest.approx(0.0, abs=1e-12)
    assert e_h @ np.array(TargetFrame().target_dir) > 0.0  # points at the target
    assert e_v[2] > 0.0  # points up


def test_roll_is_invariant_to_spin_of_hand_and_club_together() -> None:
    # Rotating the club frame about the shaft while the hand follows rigidly
    # leaves the world-space roll unchanged only if the dorsal axes are
    # re-expressed; here the dorsal axes stay fixed in the club, so a spin of
    # d degrees changes the roll by exactly d (alpha is a rotation about the shaft).
    base = address_grip_roll_deg(_state(10.0, 10.0))[0]
    spun = address_grip_roll_deg(_state(10.0, 10.0, spin_deg=7.0))[0]
    assert abs(((spun - base) + 180.0) % 360.0 - 180.0) == pytest.approx(7.0, abs=0.5)


def test_rejects_a_non_orthonormal_club_frame() -> None:
    state = _state(10.0, 10.0)
    bad = GripPoseState.__new__(GripPoseState)
    object.__setattr__(bad, "club_rotation", state.club_rotation * 1.1)
    object.__setattr__(bad, "lead_dorsal_club", state.lead_dorsal_club)
    object.__setattr__(bad, "trail_dorsal_club", state.trail_dorsal_club)
    object.__setattr__(bad, "target", TargetFrame())
    with pytest.raises(ContractViolationError):
        address_grip_roll_deg(bad)


def test_rejects_a_dorsal_axis_along_the_shaft() -> None:
    rot = _club_rotation()
    with pytest.raises(ContractViolationError):
        address_grip_roll_deg(
            GripPoseState(
                club_rotation=rot,
                lead_dorsal_club=np.array([0.0, 1.0, 0.0]),
                trail_dorsal_club=np.array([1.0, 0.0, 0.0]),
            )
        )


def test_rejects_a_vertical_shaft() -> None:
    with pytest.raises(ContractViolationError):
        reference_axes(np.array([0.0, 0.0, -1.0]), TargetFrame())


def test_rejects_left_handed_frames_until_mirroring_is_specified() -> None:
    with pytest.raises(ContractViolationError):
        address_grip_roll_deg(
            GripPoseState(
                club_rotation=_club_rotation(),
                lead_dorsal_club=np.array([1.0, 0.0, 0.0]),
                trail_dorsal_club=np.array([1.0, 0.0, 0.0]),
                target=TargetFrame(handedness="left"),
            )
        )
