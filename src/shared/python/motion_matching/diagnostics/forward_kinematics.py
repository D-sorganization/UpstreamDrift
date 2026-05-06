"""Minimal forward-kinematics evaluator for a coarse golfer skeleton.

This is intentionally lightweight: it exists to answer the question
"does this joint-angle vector produce a recognisable golfer shape?",
not to replicate the Simscape multibody dynamics. Segments are treated
as rigid links arranged through pelvis -> spine -> torso -> shoulders
-> elbows -> wrists -> hands. Joint angles are interpreted as
intrinsic (body-fixed) Euler rotations applied in the order X, Y, Z.

All angles are in DEGREES on input; this matches the convention used
by the GolfSwing3D model workspace (verified against
``trial_001_*.csv`` rows where e.g. ``model_LEStartPosition = 5.78``
is degrees, not radians).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SegmentLengths:
    """Coarse anthropometric segment lengths (metres).

    Default values are de-Leva-style defaults for an adult male, paired
    with a ~1.10 m driver shaft. They are NOT pulled from the Simscape
    model — diagnostic accuracy at the 'is this a golfer' level only.
    """

    pelvis_to_spine: float = 0.20
    spine_to_torso: float = 0.20
    torso_to_shoulder: float = 0.18  # half-width of shoulder girdle
    upper_arm: float = 0.30
    forearm: float = 0.27
    hand: float = 0.10
    club_shaft: float = 1.10


@dataclass(frozen=True)
class SkeletonPose:
    """Cartesian positions (m) of named landmarks in world frame."""

    points: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.points[key]


def _rx(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _euler_xyz(x: float, y: float, z: float) -> np.ndarray:
    """Intrinsic XYZ Euler -> rotation matrix."""
    return _rx(x) @ _ry(y) @ _rz(z)


def forward_kinematics(
    angles: Mapping[str, float],
    lengths: SegmentLengths | None = None,
) -> SkeletonPose:
    """Compute landmark positions for a coarse golfer skeleton.

    Parameters
    ----------
    angles
        Mapping of joint-angle field name -> degrees. Recognised fields
        match the Simulink.Parameter names in ``3DModelInputs*.mat``:
        Hip{X,Y,Z}, Spine{X,Y}, Torso, L/RScap{X,Y}, L/RS{X,Y,Z},
        L/RE, L/RF, L/RW{X,Y}, plus optional Translation{X,Y,Z}.
        Missing fields default to 0.
    lengths
        Segment lengths, default :class:`SegmentLengths`.

    Returns
    -------
    SkeletonPose
        Named landmark positions in world frame (metres).

    Raises
    ------
    TypeError
        If ``angles`` is not a Mapping.
    """
    if not isinstance(angles, Mapping):
        raise TypeError(f"angles must be a Mapping, got {type(angles).__name__}")
    if lengths is None:
        lengths = SegmentLengths()

    def a(name: str) -> float:
        return float(angles.get(name, 0.0))

    pelvis = np.array(
        [
            a("TranslationStartPositionX"),
            a("TranslationStartPositionY"),
            a("TranslationStartPositionZ"),
        ]
    )

    R_hip = _euler_xyz(
        a("HipStartPositionX"), a("HipStartPositionY"), a("HipStartPositionZ")
    )
    # Spine joins pelvis to torso. Spine X = forward tilt, Y = side-bend.
    R_spine = R_hip @ _rx(a("SpineStartPositionX")) @ _ry(a("SpineStartPositionY"))
    spine_top = pelvis + R_spine @ np.array([0, 0, lengths.pelvis_to_spine])

    # Torso = axial rotation (Z) about spine.
    R_torso = R_spine @ _rz(a("TorsoStartPosition"))
    torso_top = spine_top + R_torso @ np.array([0, 0, lengths.spine_to_torso])

    # Shoulders located laterally from torso top.
    R_lscap = R_torso @ _euler_xyz(
        a("LScapStartPositionX"), a("LScapStartPositionY"), 0.0
    )
    R_rscap = R_torso @ _euler_xyz(
        a("RScapStartPositionX"), a("RScapStartPositionY"), 0.0
    )
    l_shoulder = torso_top + R_lscap @ np.array([0, lengths.torso_to_shoulder, 0])
    r_shoulder = torso_top + R_rscap @ np.array([0, -lengths.torso_to_shoulder, 0])

    R_ls = R_lscap @ _euler_xyz(
        a("LSStartPositionX"), a("LSStartPositionY"), a("LSStartPositionZ")
    )
    R_rs = R_rscap @ _euler_xyz(
        a("RSStartPositionX"), a("RSStartPositionY"), a("RSStartPositionZ")
    )

    # Upper arms point along the shoulder's local +X by default (T-pose
    # extension). With all angles zero this reproduces a T-pose.
    l_elbow = l_shoulder + R_ls @ np.array([0, lengths.upper_arm, 0])
    r_elbow = r_shoulder + R_rs @ np.array([0, -lengths.upper_arm, 0])

    R_le = R_ls @ _rx(a("LEStartPosition"))
    R_re = R_rs @ _rx(a("REStartPosition"))
    l_wrist = l_elbow + R_le @ np.array([0, lengths.forearm, 0])
    r_wrist = r_elbow + R_re @ np.array([0, -lengths.forearm, 0])

    R_lw = (
        R_le
        @ _ry(a("LFStartPosition"))
        @ _euler_xyz(a("LWStartPositionX"), a("LWStartPositionY"), 0.0)
    )
    R_rw = (
        R_re
        @ _ry(a("RFStartPosition"))
        @ _euler_xyz(a("RWStartPositionX"), a("RWStartPositionY"), 0.0)
    )
    l_hand = l_wrist + R_lw @ np.array([0, lengths.hand, 0])
    r_hand = r_wrist + R_rw @ np.array([0, -lengths.hand, 0])

    butt = 0.5 * (l_hand + r_hand)
    # Club extends from butt along the average lead-hand axis.
    club_dir = R_lw @ np.array([1.0, 0.0, 0.0])
    norm = np.linalg.norm(club_dir)
    club_dir = club_dir / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    clubhead = butt + lengths.club_shaft * club_dir

    points = {
        "pelvis": pelvis,
        "spine_top": spine_top,
        "torso_top": torso_top,
        "l_shoulder": l_shoulder,
        "r_shoulder": r_shoulder,
        "l_elbow": l_elbow,
        "r_elbow": r_elbow,
        "l_wrist": l_wrist,
        "r_wrist": r_wrist,
        "l_hand": l_hand,
        "r_hand": r_hand,
        "butt": butt,
        "clubhead": clubhead,
    }
    return SkeletonPose(points=points)
