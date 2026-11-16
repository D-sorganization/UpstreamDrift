"""Per-hand shaft roll about the grip axis (HRG-GB1b, #15138).

Sign convention
----------------
Rotation follows the right-hand rule about the shaft axis pointing **butt→head**
(+x in the shared hand–club grip frame used by :mod:`grip_contact.pad_layout` and
engine bushings). Positive roll rotates the hand frame accordingly; zero recreates
today's attachment. Left-handed golfers mirror the sign (same rule as thumb azimuth
in :mod:`grip_spec`).
"""

from __future__ import annotations

import math
from typing import Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.spatial_algebra.pose6dof.rotations import (
    axis_angle_to_rotation_matrix,
)

if TYPE_CHECKING:
    from src.shared.python.grip_spec.types import GripSpec

SHAFT_ROLL_RANGE_DEG: tuple[float, float] = (-180.0, 180.0)

#: Unit shaft axis in the shared grip frame (butt → head).
SHARED_GRIP_SHAFT_AXIS: NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=float)

#: Same axis expressed in the appearance hand-grip frame (z = butt → head).
APPEARANCE_GRIP_SHAFT_AXIS: NDArray[np.float64] = np.array([0.0, 0.0, 1.0], dtype=float)


def _normalize_axis(axis: np.ndarray, name: str) -> NDArray[np.float64]:
    vector = np.asarray(axis, dtype=float)
    require(
        vector.shape == (3,) and np.isfinite(vector).all(), f"{name} must be finite"
    )
    norm = float(np.linalg.norm(vector))
    require(norm > 1e-12, f"{name} must be non-zero")
    return vector / norm


def apply_shaft_roll(
    hand_frame: np.ndarray,
    shaft_axis: np.ndarray,
    roll_deg: float,
) -> np.ndarray:
    """Rotate ``hand_frame`` about ``shaft_axis`` by ``roll_deg`` (degrees).

    Accepts a proper 3×3 rotation or a 4×4 rigid transform whose rotation block
    is proper. The translation is unchanged; only the orientation is updated.
    """
    if not math.isfinite(roll_deg):
        raise ValueError(f"roll_deg must be finite, got {roll_deg!r}")
    if abs(roll_deg) <= 1e-15:
        return np.asarray(hand_frame, dtype=float).copy()
    axis = _normalize_axis(shaft_axis, "shaft_axis")
    arr = np.asarray(hand_frame, dtype=float)
    delta = axis_angle_to_rotation_matrix(axis, math.radians(roll_deg))
    if arr.shape == (3, 3):
        return delta @ arr
    if arr.shape == (4, 4):
        out = arr.copy()
        out[:3, :3] = delta @ out[:3, :3]
        return out
    raise ValueError(f"hand_frame must be shape (3, 3) or (4, 4), got {arr.shape}")


def resolved_shaft_roll_deg(
    spec: GripSpec,
    role: Literal["lead", "trail"],
) -> float:
    """Shaft roll for ``role`` with left-handed mirroring."""
    raw = spec.lead.shaft_roll_deg if role == "lead" else spec.trail.shaft_roll_deg
    if spec.handedness == "left":
        return -raw
    return raw


def pose_shaft_roll_deg(pose: GripPose, hand: Hand) -> float:
    """Resolved per-hand shaft roll carried on a :class:`GripPose`."""
    if not isinstance(hand, Hand):
        raise TypeError(f"hand must be a Hand, got {hand!r}")
    roll = pose.lead_shaft_roll_deg if hand is Hand.LEAD else pose.trail_shaft_roll_deg
    if pose.handedness == "left":
        return -roll
    return roll


def hand_to_club_relative_rotation(
    pose: GripPose,
    hand: Hand,
    *,
    shaft_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Hand-body to shared-grip rotation with shaft roll (identity at zero)."""
    axis = SHARED_GRIP_SHAFT_AXIS if shaft_axis is None else shaft_axis
    return apply_shaft_roll(np.eye(3), axis, pose_shaft_roll_deg(pose, hand))[:3, :3]


def attachment_base_transform(
    pose: GripPose,
    hand: Hand,
    *,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """4×4 hand-to-grip attachment including shaft roll (shared grip frame)."""
    t = np.eye(4) if base is None else np.asarray(base, dtype=float).copy()
    return apply_shaft_roll(t, SHARED_GRIP_SHAFT_AXIS, pose_shaft_roll_deg(pose, hand))


def appearance_frame_with_shaft_roll(
    frame: np.ndarray,
    pose: GripPose,
    hand: Hand,
) -> np.ndarray:
    """Apply shaft roll in the appearance hand-grip frame (z = shaft)."""
    roll_deg = pose_shaft_roll_deg(pose, hand)
    if not math.isfinite(roll_deg) or abs(roll_deg) <= 1e-15:
        return np.asarray(frame, dtype=float).copy()
    arr = np.asarray(frame, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"frame must be shape (4, 4), got {arr.shape}")
    axis = _normalize_axis(APPEARANCE_GRIP_SHAFT_AXIS, "shaft_axis")
    delta = axis_angle_to_rotation_matrix(axis, math.radians(roll_deg))
    out = arr.copy()
    out[:3, :3] = out[:3, :3] @ delta
    return out


def grip_interface_offsets_with_shaft_roll(
    interface,
    pose: GripPose,
) -> dict[str, np.ndarray]:
    """Per-side club grip frames with shaft roll applied (#15138)."""
    return {
        "L": attachment_base_transform(
            pose, Hand.LEAD, base=interface.frame("L").matrix()
        ),
        "R": attachment_base_transform(
            pose, Hand.TRAIL, base=interface.frame("R").matrix()
        ),
    }


__all__ = [
    "APPEARANCE_GRIP_SHAFT_AXIS",
    "SHAFT_ROLL_RANGE_DEG",
    "SHARED_GRIP_SHAFT_AXIS",
    "appearance_frame_with_shaft_roll",
    "apply_shaft_roll",
    "attachment_base_transform",
    "grip_interface_offsets_with_shaft_roll",
    "hand_to_club_relative_rotation",
    "pose_shaft_roll_deg",
    "resolved_shaft_roll_deg",
]
