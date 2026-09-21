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
    pelvis_to_hip: float = 0.10
    thigh: float = 0.44
    shin: float = 0.42
    foot: float = 0.18


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
    *,
    include_lower_body: bool = False,
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
    include_lower_body
        Whether to compute lower-body landmarks (hips, knees, ankles, feet).
        Defaults to False to preserve 13-landmark upper-body kinematics compatibility.

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
    points = _compute_upper_body(pelvis, R_hip, a, lengths)
    if include_lower_body:
        points.update(_compute_lower_body(pelvis, R_hip, a, lengths))
    return SkeletonPose(points=points)


def _compute_upper_body(
    pelvis: np.ndarray,
    R_hip: np.ndarray,
    a: object,
    lengths: SegmentLengths,
) -> dict[str, np.ndarray]:
    getter = a if callable(a) else (lambda _: 0.0)
    R_spine = (
        R_hip @ _rx(getter("SpineStartPositionX")) @ _ry(getter("SpineStartPositionY"))
    )
    spine_top = pelvis + R_spine @ np.array([0, 0, lengths.pelvis_to_spine])

    R_torso = R_spine @ _rz(getter("TorsoStartPosition"))
    torso_top = spine_top + R_torso @ np.array([0, 0, lengths.spine_to_torso])

    R_lscap = R_torso @ _euler_xyz(
        getter("LScapStartPositionX"), getter("LScapStartPositionY"), 0.0
    )
    R_rscap = R_torso @ _euler_xyz(
        getter("RScapStartPositionX"), getter("RScapStartPositionY"), 0.0
    )
    l_shoulder = torso_top + R_lscap @ np.array([0, lengths.torso_to_shoulder, 0])
    r_shoulder = torso_top + R_rscap @ np.array([0, -lengths.torso_to_shoulder, 0])

    R_ls = R_lscap @ _euler_xyz(
        getter("LSStartPositionX"),
        getter("LSStartPositionY"),
        getter("LSStartPositionZ"),
    )
    R_rs = R_rscap @ _euler_xyz(
        getter("RSStartPositionX"),
        getter("RSStartPositionY"),
        getter("RSStartPositionZ"),
    )

    l_elbow = l_shoulder + R_ls @ np.array([0, lengths.upper_arm, 0])
    r_elbow = r_shoulder + R_rs @ np.array([0, -lengths.upper_arm, 0])

    R_le = R_ls @ _rx(getter("LEStartPosition"))
    R_re = R_rs @ _rx(getter("REStartPosition"))
    l_wrist = l_elbow + R_le @ np.array([0, lengths.forearm, 0])
    r_wrist = r_elbow + R_re @ np.array([0, -lengths.forearm, 0])

    R_lw = (
        R_le
        @ _ry(getter("LFStartPosition"))
        @ _euler_xyz(getter("LWStartPositionX"), getter("LWStartPositionY"), 0.0)
    )
    R_rw = (
        R_re
        @ _ry(getter("RFStartPosition"))
        @ _euler_xyz(getter("RWStartPositionX"), getter("RWStartPositionY"), 0.0)
    )
    l_hand = l_wrist + R_lw @ np.array([0, lengths.hand, 0])
    r_hand = r_wrist + R_rw @ np.array([0, -lengths.hand, 0])

    butt = 0.5 * (l_hand + r_hand)
    club_dir = R_lw @ np.array([1.0, 0.0, 0.0])
    norm = np.linalg.norm(club_dir)
    club_dir = club_dir / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    clubhead = butt + lengths.club_shaft * club_dir

    return {
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


def _compute_lower_body(
    pelvis: np.ndarray,
    R_hip: np.ndarray,
    a: object,
    lengths: SegmentLengths,
) -> dict[str, np.ndarray]:
    getter = a if callable(a) else (lambda _: 0.0)
    l_hip = pelvis + R_hip @ np.array([0.0, lengths.pelvis_to_hip, 0.0])
    r_hip = pelvis + R_hip @ np.array([0.0, -lengths.pelvis_to_hip, 0.0])

    R_lhip = R_hip @ _euler_xyz(
        getter("LHipStartPositionX"),
        getter("LHipStartPositionY"),
        getter("LHipStartPositionZ"),
    )
    R_rhip = R_hip @ _euler_xyz(
        getter("RHipStartPositionX"),
        getter("RHipStartPositionY"),
        getter("RHipStartPositionZ"),
    )

    l_knee = l_hip + R_lhip @ np.array([0.0, 0.0, -lengths.thigh])
    r_knee = r_hip + R_rhip @ np.array([0.0, 0.0, -lengths.thigh])

    R_lknee = R_lhip @ _rx(getter("LKneeStartPosition"))
    R_rknee = R_rhip @ _rx(getter("RKneeStartPosition"))

    l_ankle = l_knee + R_lknee @ np.array([0.0, 0.0, -lengths.shin])
    r_ankle = r_knee + R_rknee @ np.array([0.0, 0.0, -lengths.shin])

    R_lankle = R_lknee @ _rx(getter("LAnkleStartPosition"))
    R_rankle = R_rknee @ _rx(getter("RAnkleStartPosition"))

    l_foot = l_ankle + R_lankle @ np.array([lengths.foot, 0.0, 0.0])
    r_foot = r_ankle + R_rankle @ np.array([lengths.foot, 0.0, 0.0])

    return {
        "l_hip": l_hip,
        "r_hip": r_hip,
        "l_knee": l_knee,
        "r_knee": r_knee,
        "l_ankle": l_ankle,
        "r_ankle": r_ankle,
        "l_foot": l_foot,
        "r_foot": r_foot,
    }
