"""Swing comparison metrics module (Issue #11164).

Provides (SwingMotion and the capture-A marker schema live in ``motion``):
- Pure metric calculation functions with DbC:
  * compute_tempo
  * compute_segment_rotations
  * compute_kinematic_sequence
  * compute_lead_arm_metrics
  * compute_wrist_metrics
  * compute_hand_path_metrics
  * compute_club_metrics
  * compute_all_metrics
  * compare -> ComparisonReport
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.swing_comparison.events import (
    SwingEvents,
    _fill_nans_1d,
    _fill_nans_3d,
    detect_events,
)

from src.shared.python.swing_comparison.turn import (
    LineTurn,
    TurnMetrics,
    marker_turn_lines,
)
from src.shared.python.swing_comparison.motion import (
    CAPTURE_A_PELVIS_LEFT_LABELS,
    CAPTURE_A_PELVIS_RIGHT_LABELS,
    CAPTURE_A_SHOULDER_LEFT_LABELS,
    CAPTURE_A_SHOULDER_RIGHT_LABELS,
    CAPTURE_A_TRUNK_LEFT_LABELS,
    CAPTURE_A_TRUNK_RIGHT_LABELS,
    SwingMotion,
    swing_motion_from_markers,
)


# ---------------------------------------------------------------------------
# Metric Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TempoMetrics:
    """Swing tempo metrics.

    Attributes:
        backswing_time: Duration from address to top of backswing in seconds.
        downswing_time: Duration from top of backswing to impact in seconds.
        tempo_ratio: Ratio of backswing duration to downswing duration (dimensionless, ~3:1).
    """

    backswing_time: float
    downswing_time: float
    tempo_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "backswing_time_s": float(self.backswing_time),
            "downswing_time_s": float(self.downswing_time),
            "tempo_ratio": float(self.tempo_ratio),
        }


@dataclass(frozen=True)
class SegmentRotationMetrics:
    """Pelvis, upper-trunk and shoulder-girdle turn, and X-factor separation.

    All angles in degrees, relative to the same line at address, positive in the
    backswing (see ``swing_comparison.turn`` for the frame convention).  Values
    are NaN where the markers are unavailable; unavailable never means zero.

    The former "thorax yaw" was the shoulder-marker line, which rides the
    scapula.  It is now the ``shoulder_girdle`` line; the BackLeft/BackRight
    ``upper_trunk`` line is separate.  ``thorax_yaw*`` remain as deprecated
    aliases of the shoulder-girdle line.

    Attributes:
        pelvis_yaw / shoulder_girdle_yaw / upper_trunk_yaw: Turn time series (N,).
        x_factor: Upper-trunk minus pelvis turn (N,).
        x_factor_shoulder_girdle: Shoulder-girdle minus pelvis turn (N,).
        *_address / *_top / *_impact: Values at the three swing events.
        x_factor_stretch: Maximum absolute X-factor during the swing (NaN if none).
        turn: The full ``TurnMetrics`` (status, reasons, max backswing).
    """

    pelvis_yaw: np.ndarray
    shoulder_girdle_yaw: np.ndarray
    upper_trunk_yaw: np.ndarray
    x_factor: np.ndarray
    x_factor_shoulder_girdle: np.ndarray
    pelvis_yaw_address: float
    pelvis_yaw_top: float
    pelvis_yaw_impact: float
    shoulder_girdle_yaw_address: float
    shoulder_girdle_yaw_top: float
    shoulder_girdle_yaw_impact: float
    upper_trunk_yaw_address: float
    upper_trunk_yaw_top: float
    upper_trunk_yaw_impact: float
    x_factor_address: float
    x_factor_top: float
    x_factor_impact: float
    x_factor_stretch: float
    turn: TurnMetrics | None = None

    def _deprecated(self, name: str, replacement: str) -> None:
        warnings.warn(
            f"SegmentRotationMetrics.{name} is deprecated: it was the shoulder-"
            f"marker (scapular) line, now named {replacement}; the rib-cage line is "
            "upper_trunk_yaw",
            DeprecationWarning,
            stacklevel=3,
        )

    @property
    def thorax_yaw(self) -> np.ndarray:
        """Deprecated alias of ``shoulder_girdle_yaw``."""
        self._deprecated("thorax_yaw", "shoulder_girdle_yaw")
        return self.shoulder_girdle_yaw

    @property
    def thorax_yaw_address(self) -> float:
        """Deprecated alias of ``shoulder_girdle_yaw_address``."""
        self._deprecated("thorax_yaw_address", "shoulder_girdle_yaw_address")
        return self.shoulder_girdle_yaw_address

    @property
    def thorax_yaw_top(self) -> float:
        """Deprecated alias of ``shoulder_girdle_yaw_top``."""
        self._deprecated("thorax_yaw_top", "shoulder_girdle_yaw_top")
        return self.shoulder_girdle_yaw_top

    @property
    def thorax_yaw_impact(self) -> float:
        """Deprecated alias of ``shoulder_girdle_yaw_impact``."""
        self._deprecated("thorax_yaw_impact", "shoulder_girdle_yaw_impact")
        return self.shoulder_girdle_yaw_impact

    def to_dict(self) -> dict[str, float]:
        gird = {
            f"shoulder_girdle_yaw_{k}_deg": float(
                getattr(self, f"shoulder_girdle_yaw_{k}")
            )
            for k in ("address", "top", "impact")
        }
        return {
            "pelvis_yaw_address_deg": float(self.pelvis_yaw_address),
            "pelvis_yaw_top_deg": float(self.pelvis_yaw_top),
            "pelvis_yaw_impact_deg": float(self.pelvis_yaw_impact),
            **gird,
            "upper_trunk_yaw_address_deg": float(self.upper_trunk_yaw_address),
            "upper_trunk_yaw_top_deg": float(self.upper_trunk_yaw_top),
            "upper_trunk_yaw_impact_deg": float(self.upper_trunk_yaw_impact),
            # Deprecated keys kept for existing consumers; shoulder-girdle values.
            "thorax_yaw_address_deg": gird["shoulder_girdle_yaw_address_deg"],
            "thorax_yaw_top_deg": gird["shoulder_girdle_yaw_top_deg"],
            "thorax_yaw_impact_deg": gird["shoulder_girdle_yaw_impact_deg"],
            "x_factor_address_deg": float(self.x_factor_address),
            "x_factor_top_deg": float(self.x_factor_top),
            "x_factor_impact_deg": float(self.x_factor_impact),
            "x_factor_stretch_deg": float(self.x_factor_stretch),
        }


@dataclass(frozen=True)
class SegmentKinematicPeak:
    """Peak angular speed and timing for a single segment."""

    name: str
    peak_speed: float
    peak_time: float
    peak_frame: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "peak_speed_deg_s": float(self.peak_speed),
            "peak_time_s": float(self.peak_time),
            "peak_frame": int(self.peak_frame),
        }


KINEMATIC_SEGMENTS: tuple[str, ...] = ("pelvis", "thorax", "lead_arm", "club")
"""Kinematic-sequence segments, proximal to distal."""


POST_IMPACT_MARGIN_FRACTION = 0.15
"""Post-impact search margin as a fraction of the downswing duration."""

POST_IMPACT_MARGIN_S = 0.015
"""Fixed post-impact search margin in seconds (about 5 frames at 360 Hz)."""


@dataclass(frozen=True)
class KinematicSequenceMetrics:
    """Kinematic sequence peak angular speeds and temporal ordering.

    Attributes:
        pelvis: Pelvis kinematic peak info.
        thorax: Thorax kinematic peak info.
        lead_arm: Lead arm kinematic peak info.
        club: Club kinematic peak info.
        order: Tuple of segment names ordered chronologically by peak speed time.
        is_proximal_to_distal: True if order matches standard proximal-to-distal sequence.
        thorax_proxy: Which markers measured thorax yaw: "trunk_back_markers"
            (BackLeft/BackRight), "shoulder_line" (acromion fallback, includes
            scapular motion) or "unavailable".
    """

    pelvis: SegmentKinematicPeak
    thorax: SegmentKinematicPeak
    lead_arm: SegmentKinematicPeak
    club: SegmentKinematicPeak
    order: tuple[str, ...]
    is_proximal_to_distal: bool
    thorax_proxy: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pelvis": self.pelvis.to_dict(),
            "thorax": self.thorax.to_dict(),
            "lead_arm": self.lead_arm.to_dict(),
            "club": self.club.to_dict(),
            "order": list(self.order),
            "is_proximal_to_distal": bool(self.is_proximal_to_distal),
            "thorax_proxy": self.thorax_proxy,
        }


@dataclass(frozen=True)
class LeadArmMetrics:
    """Lead arm elbow angle metrics.

    Included angle at the elbow is between upper arm (elbow -> shoulder)
    and forearm (elbow -> wrist). Straight arm = 180 degrees.
    Flexion angle is (180 - included_angle) in degrees.

    Attributes:
        elbow_angle_deg: Time series of elbow included angle (N,) in degrees.
        elbow_angle_address: Included elbow angle at address in degrees.
        elbow_angle_top: Included elbow angle at top of backswing in degrees.
        elbow_angle_impact: Included elbow angle at impact in degrees.
        min_included_angle: Minimum included elbow angle from address to impact in degrees.
        max_flexion_deg: Maximum elbow flexion (180 - min_included_angle) in degrees.
    """

    elbow_angle_deg: np.ndarray
    elbow_angle_address: float
    elbow_angle_top: float
    elbow_angle_impact: float
    min_included_angle: float
    max_flexion_deg: float

    def to_dict(self) -> dict[str, float]:
        return {
            "elbow_angle_address_deg": float(self.elbow_angle_address),
            "elbow_angle_top_deg": float(self.elbow_angle_top),
            "elbow_angle_impact_deg": float(self.elbow_angle_impact),
            "min_included_angle_deg": float(self.min_included_angle),
            "max_flexion_deg": float(self.max_flexion_deg),
        }


@dataclass(frozen=True)
class WristMetrics:
    """Lead wrist hinge angle metrics.

    Angle between forearm vector (elbow -> wrist) and club shaft vector (grip -> clubhead).

    Attributes:
        hinge_angle_deg: Time series of wrist hinge angle (N,) in degrees.
        hinge_angle_top: Wrist hinge angle at top of backswing in degrees.
        hinge_angle_impact: Wrist hinge angle at impact in degrees.
    """

    hinge_angle_deg: np.ndarray
    hinge_angle_top: float
    hinge_angle_impact: float

    def to_dict(self) -> dict[str, float]:
        return {
            "hinge_angle_top_deg": float(self.hinge_angle_top),
            "hinge_angle_impact_deg": float(self.hinge_angle_impact),
        }


@dataclass(frozen=True)
class HandPathMetrics:
    """Hand / grip path length and elevation metrics.

    Trajectory in meters, Z up.

    Attributes:
        path_length_total_m: Total 3D distance traveled by grip over full motion in meters.
        path_length_swing_m: 3D distance traveled by grip from address to finish in meters.
        max_height_top_m: Maximum Z elevation reached by grip at / near top of backswing in meters.
        height_address_m: Grip Z elevation at address in meters.
        height_impact_m: Grip Z elevation at impact in meters.
    """

    path_length_total_m: float
    path_length_swing_m: float
    max_height_top_m: float
    height_address_m: float
    height_impact_m: float

    def to_dict(self) -> dict[str, float]:
        return {
            "path_length_total_m": float(self.path_length_total_m),
            "path_length_swing_m": float(self.path_length_swing_m),
            "max_height_top_m": float(self.max_height_top_m),
            "height_address_m": float(self.height_address_m),
            "height_impact_m": float(self.height_impact_m),
        }


@dataclass(frozen=True)
class ClubMetrics:
    """Club delivery kinematics at impact.

    Target line is oriented along +X in a right-handed frame with Z up.

    Attributes:
        impact_club_head_speed_m_s: Clubhead speed at impact in m/s.
        peak_club_head_speed_m_s: Maximum clubhead speed during the swing in m/s.
        impact_shaft_lean_deg: Forward shaft lean in degrees at impact (positive = grip ahead of head).
        impact_face_angle_deg: Face angle relative to target line in degrees (positive = open), or None.
    """

    impact_club_head_speed_m_s: float
    peak_club_head_speed_m_s: float
    impact_shaft_lean_deg: float
    impact_face_angle_deg: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "impact_club_head_speed_m_s": float(self.impact_club_head_speed_m_s),
            "peak_club_head_speed_m_s": float(self.peak_club_head_speed_m_s),
            "impact_shaft_lean_deg": float(self.impact_shaft_lean_deg),
            "impact_face_angle_deg": float(self.impact_face_angle_deg)
            if self.impact_face_angle_deg is not None
            else None,
        }


@dataclass(frozen=True)
class SwingMetrics:
    """Comprehensive composite swing metrics container."""

    tempo: TempoMetrics
    segment_rotation: SegmentRotationMetrics
    kinematic_sequence: KinematicSequenceMetrics
    lead_arm: LeadArmMetrics
    wrist: WristMetrics
    hand_path: HandPathMetrics
    club: ClubMetrics

    def _segment_peak(self, segment: str) -> SegmentKinematicPeak:
        if segment not in KINEMATIC_SEGMENTS:
            raise ValueError(
                f"segment must be one of {KINEMATIC_SEGMENTS}, got {segment!r}"
            )
        peak: SegmentKinematicPeak = getattr(self.kinematic_sequence, segment)
        return peak

    def peak_speed(self, segment: str) -> float:
        """Peak angular speed (deg/s) of a kinematic-sequence ``segment``."""
        return self._segment_peak(segment).peak_speed

    def peak_time(self, segment: str) -> float:
        """Time (s) of the peak angular speed of a kinematic-sequence ``segment``."""
        return self._segment_peak(segment).peak_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo": self.tempo.to_dict(),
            "segment_rotation": self.segment_rotation.to_dict(),
            "kinematic_sequence": self.kinematic_sequence.to_dict(),
            "lead_arm": self.lead_arm.to_dict(),
            "wrist": self.wrist.to_dict(),
            "hand_path": self.hand_path.to_dict(),
            "club": self.club.to_dict(),
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison report comparing two swings."""

    motion_a_events: SwingEvents
    motion_b_events: SwingEvents
    metrics_a: SwingMetrics
    metrics_b: SwingMetrics
    differences: dict[str, float | None]
    shared_marker_rms: dict[str, float]
    mean_marker_rms: float

    def to_dict(self) -> dict[str, Any]:
        from src.shared.python.swing_comparison.report import comparison_to_dict

        return comparison_to_dict(self)

    def to_markdown(self) -> str:
        from src.shared.python.swing_comparison.report import comparison_to_markdown

        return comparison_to_markdown(self)


# ---------------------------------------------------------------------------
# Metric Calculation Functions with DbC
# ---------------------------------------------------------------------------


def compute_tempo(events: SwingEvents) -> TempoMetrics:
    """Compute swing tempo metrics from detected events.

    Preconditions:
        - events.top_time >= events.address_time
        - events.impact_time > events.top_time

    Postconditions:
        - backswing_time >= 0
        - downswing_time > 0
        - tempo_ratio > 0

    Args:
        events: SwingEvents instance.

    Returns:
        TempoMetrics containing backswing time, downswing time, and tempo ratio.
    """
    require(events.top_time >= events.address_time, "top_time must be >= address_time")
    require(
        events.impact_time > events.top_time, "impact_time must be strictly > top_time"
    )

    bs_time = float(events.top_time - events.address_time)
    ds_time = float(events.impact_time - events.top_time)
    ratio = float(bs_time / ds_time)

    ensure(bs_time >= 0.0, "backswing_time must be non-negative")
    ensure(ds_time > 0.0, "downswing_time must be positive")
    ensure(ratio >= 0.0, "tempo_ratio must be non-negative")

    return TempoMetrics(
        backswing_time=bs_time,
        downswing_time=ds_time,
        tempo_ratio=ratio,
    )


def _resolve_vector(
    markers: dict[str, np.ndarray],
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
    fallback_dim: int,
) -> np.ndarray:
    """Resolve a bilateral segment vector from left and right marker sets."""
    left_pt: np.ndarray | None = None
    right_pt: np.ndarray | None = None

    for lbl in left_labels:
        if lbl in markers:
            left_pt = _fill_nans_3d(markers[lbl])
            break
    for lbl in right_labels:
        if lbl in markers:
            right_pt = _fill_nans_3d(markers[lbl])
            break

    if left_pt is not None and right_pt is not None:
        return left_pt - right_pt

    # Fallback to zero vector
    return np.zeros((fallback_dim, 3), dtype=np.float64)


def compute_segment_rotations(
    motion: SwingMotion,
    events: SwingEvents,
) -> SegmentRotationMetrics:
    """Compute pelvis, upper-trunk and shoulder-girdle turn and X-factor.

    Definitions (see ``swing_comparison.turn``; Z up, golfer faces -X, target -Y):
    - Pelvis: WaistLeft/WaistRight line turn relative to address.
    - Upper trunk: BackLeft/BackRight line turn relative to address.
    - Shoulder girdle: ShoulderBack line turn (rides the scapula) relative to address.
    - X-factor: upper-trunk turn minus pelvis turn; the shoulder-girdle variant is
      ``x_factor_shoulder_girdle``.
    - X-factor stretch: maximum absolute X-factor during the swing.
    A pair with a missing marker or a long gap yields NaN with a reason in
    ``result.turn``, never zero.

    Preconditions:
        - len(motion.t) >= 4
        - 0 <= events.address_idx <= events.top_idx <= events.impact_idx < len(motion.t)

    Returns:
        SegmentRotationMetrics with angle time series and event values.
    """
    n = len(motion.t)
    require(
        0 <= events.address_idx <= events.top_idx <= events.impact_idx < n,
        "Events must satisfy 0 <= address <= top <= impact < n",
    )
    turn = marker_turn_lines(motion.markers, motion.t, events)

    def at(line: LineTurn, idx: int) -> float:
        return float(line.turn_deg[idx])

    xf = turn.x_factor.turn_deg
    stretch = float(np.nanmax(np.abs(xf))) if np.isfinite(xf).any() else float("nan")
    a, tp, im = events.address_idx, events.top_idx, events.impact_idx
    return SegmentRotationMetrics(
        pelvis_yaw=turn.pelvis.turn_deg,
        shoulder_girdle_yaw=turn.shoulder_girdle.turn_deg,
        upper_trunk_yaw=turn.upper_trunk.turn_deg,
        x_factor=xf,
        x_factor_shoulder_girdle=turn.x_factor_shoulder_girdle.turn_deg,
        pelvis_yaw_address=at(turn.pelvis, a),
        pelvis_yaw_top=at(turn.pelvis, tp),
        pelvis_yaw_impact=at(turn.pelvis, im),
        shoulder_girdle_yaw_address=at(turn.shoulder_girdle, a),
        shoulder_girdle_yaw_top=at(turn.shoulder_girdle, tp),
        shoulder_girdle_yaw_impact=at(turn.shoulder_girdle, im),
        upper_trunk_yaw_address=at(turn.upper_trunk, a),
        upper_trunk_yaw_top=at(turn.upper_trunk, tp),
        upper_trunk_yaw_impact=at(turn.upper_trunk, im),
        x_factor_address=at(turn.x_factor, a),
        x_factor_top=at(turn.x_factor, tp),
        x_factor_impact=at(turn.x_factor, im),
        x_factor_stretch=stretch,
        turn=turn,
    )


def _segment_yaw_angular_speed(
    motion: SwingMotion,
    t: np.ndarray,
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
) -> np.ndarray:
    """Return the yaw angular speed (deg/s) of a left/right marker-pair segment."""
    vec = _resolve_vector(
        motion.markers, left_labels, right_labels, fallback_dim=len(t)
    )
    yaw = np.unwrap(np.arctan2(vec[:, 1], vec[:, 0]))
    return np.degrees(np.abs(np.gradient(yaw, t)))


def _has_marker_pair(
    motion: SwingMotion,
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
) -> bool:
    """Return True if a left and a right marker with finite data both exist."""

    def _usable(labels: tuple[str, ...]) -> bool:
        return any(
            lbl in motion.markers and not np.all(np.isnan(motion.markers[lbl]))
            for lbl in labels
        )

    return _usable(left_labels) and _usable(right_labels)


def _thorax_yaw_angular_speed(
    motion: SwingMotion, t: np.ndarray
) -> tuple[np.ndarray, str]:
    """Return (thorax yaw speed deg/s, proxy name) for the kinematic sequence.

    Prefers the BackLeft/BackRight trunk markers; the acromion shoulder line is
    only a fallback because it keeps rotating with the arms after impact.
    """
    if _has_marker_pair(
        motion, CAPTURE_A_TRUNK_LEFT_LABELS, CAPTURE_A_TRUNK_RIGHT_LABELS
    ):
        labels, proxy = (
            (CAPTURE_A_TRUNK_LEFT_LABELS, CAPTURE_A_TRUNK_RIGHT_LABELS),
            "trunk_back_markers",
        )
    elif _has_marker_pair(
        motion, CAPTURE_A_SHOULDER_LEFT_LABELS, CAPTURE_A_SHOULDER_RIGHT_LABELS
    ):
        labels, proxy = (
            (CAPTURE_A_SHOULDER_LEFT_LABELS, CAPTURE_A_SHOULDER_RIGHT_LABELS),
            "shoulder_line",
        )
    else:
        return np.zeros(len(t), dtype=np.float64), "unavailable"
    return _segment_yaw_angular_speed(motion, t, *labels), proxy


def _lead_arm_angular_speed(motion: SwingMotion, t: np.ndarray, n: int) -> np.ndarray:
    """Return the 3D angular speed (deg/s) of the shoulder-to-wrist vector."""
    sh_pt: np.ndarray | None = None
    for lbl in ("LShoulderTop", "LShoulderBack"):
        if lbl in motion.markers:
            sh_pt = _fill_nans_3d(motion.markers[lbl])
            break
    wrist_pt: np.ndarray | None = None
    for lbl in ("LWristTop", "LElbowOut"):
        if lbl in motion.markers:
            wrist_pt = _fill_nans_3d(motion.markers[lbl])
            break
    if sh_pt is None or wrist_pt is None:
        return np.zeros(n, dtype=np.float64)

    arm_v = wrist_pt - sh_pt
    norms = np.linalg.norm(arm_v, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    u_arm = arm_v / norms
    du_arm = np.gradient(u_arm, t, axis=0)
    return np.degrees(np.linalg.norm(du_arm, axis=-1))


def _club_endpoints(
    motion: SwingMotion,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (club_head, grip) point arrays, falling back to Capture-A marker names."""
    head_pt = motion.club_head
    if head_pt is None and "Marker_2:2:1" in motion.markers:
        head_pt = motion.markers["Marker_2:2:1"]
    grip_pt = motion.grip
    if grip_pt is None and "Marker_3:3:1" in motion.markers:
        grip_pt = motion.markers["Marker_3:3:1"]
    return head_pt, grip_pt


def _club_angular_speed(motion: SwingMotion, t: np.ndarray, n: int) -> np.ndarray:
    """Return the 3D angular speed (deg/s) of the grip-to-clubhead shaft vector."""
    head_pt, grip_pt = _club_endpoints(motion)

    if head_pt is None or grip_pt is None:
        return np.zeros(n, dtype=np.float64)

    shaft_v = _fill_nans_3d(head_pt) - _fill_nans_3d(grip_pt)
    norms = np.linalg.norm(shaft_v, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    u_shaft = shaft_v / norms
    du_shaft = np.gradient(u_shaft, t, axis=0)
    return np.degrees(np.linalg.norm(du_shaft, axis=-1))


def _extract_kinematic_peak(
    name: str,
    omega: np.ndarray,
    t: np.ndarray,
    window: slice,
    w_start: int,
    fallback_idx: int,
) -> SegmentKinematicPeak:
    """Return the peak angular speed for one segment within ``window``."""
    sub = omega[window]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return SegmentKinematicPeak(name, 0.0, float(t[fallback_idx]), fallback_idx)
    peak_val = float(np.nanmax(sub))
    local_idx = int(np.nanargmax(sub))
    global_idx = w_start + local_idx
    return SegmentKinematicPeak(name, peak_val, float(t[global_idx]), global_idx)


def compute_kinematic_sequence(
    motion: SwingMotion,
    events: SwingEvents,
) -> KinematicSequenceMetrics:
    """Compute peak angular speeds and temporal sequence for Pelvis, Thorax, Arm, and Club.

    Definitions:
    - Pelvis: Angular speed of pelvis yaw rotation (deg/s).
    - Thorax: Angular speed of thorax yaw rotation (deg/s).
    - Lead arm: 3D angular speed of the shoulder-to-wrist vector (deg/s).
    - Club: 3D angular speed of the grip-to-clubhead shaft vector (deg/s).
    - Order: Chronological sequence of peak speeds. Proximal-to-distal ordering is:
      ("pelvis", "thorax", "lead_arm", "club").

    Preconditions:
        - len(motion.t) >= 4
        - events.top_idx < events.impact_idx

    Returns:
        KinematicSequenceMetrics with peaks, times, and ordering.
    """
    t = motion.t
    n = len(t)
    require(
        0 <= events.top_idx < events.impact_idx <= events.finish_idx < n,
        "Events must satisfy 0 <= top < impact <= finish < n",
    )

    omega_pelvis = _segment_yaw_angular_speed(
        motion, t, CAPTURE_A_PELVIS_LEFT_LABELS, CAPTURE_A_PELVIS_RIGHT_LABELS
    )
    omega_thorax, thorax_proxy = _thorax_yaw_angular_speed(motion, t)
    omega_arm = _lead_arm_angular_speed(motion, t, n)
    omega_club = _club_angular_speed(motion, t, n)

    # Search window: downswing plus a follow-through margin defined in seconds
    # so the window does not depend on the capture rate.
    downswing_s = float(t[events.impact_idx] - t[events.top_idx])
    margin_s = POST_IMPACT_MARGIN_FRACTION * downswing_s + POST_IMPACT_MARGIN_S
    w_start = events.top_idx
    w_end = int(np.searchsorted(t, t[events.impact_idx] + margin_s, side="right"))
    w_end = min(n, max(w_end, events.impact_idx + 1))
    window = slice(w_start, w_end)

    peak_p = _extract_kinematic_peak(
        "pelvis", omega_pelvis, t, window, w_start, events.top_idx
    )
    peak_t = _extract_kinematic_peak(
        "thorax", omega_thorax, t, window, w_start, events.top_idx
    )
    peak_a = _extract_kinematic_peak(
        "lead_arm", omega_arm, t, window, w_start, events.top_idx
    )
    peak_c = _extract_kinematic_peak(
        "club", omega_club, t, window, w_start, events.top_idx
    )

    # Sort segments by peak time
    peaks = [peak_p, peak_t, peak_a, peak_c]
    peaks_sorted = sorted(peaks, key=lambda p: p.peak_time)
    order = tuple(p.name for p in peaks_sorted)
    is_p2d = order == ("pelvis", "thorax", "lead_arm", "club")

    return KinematicSequenceMetrics(
        pelvis=peak_p,
        thorax=peak_t,
        lead_arm=peak_a,
        club=peak_c,
        order=order,
        is_proximal_to_distal=is_p2d,
        thorax_proxy=thorax_proxy,
    )


def compute_lead_arm_metrics(
    motion: SwingMotion,
    events: SwingEvents,
) -> LeadArmMetrics:
    """Compute lead elbow included angle and maximum flexion from address to impact.

    Definitions:
    - Elbow included angle: Angle between upper arm (elbow -> shoulder) and
      forearm (elbow -> wrist) in degrees. Fully extended = 180 degrees.
    - Maximum flexion: 180 - min_included_angle in degrees.

    Preconditions:
        - 0 <= events.address_idx <= events.impact_idx < len(motion.t)

    Args:
        motion: SwingMotion instance.
        events: SwingEvents instance.

    Returns:
        LeadArmMetrics containing included angle time series and key event values.
    """
    n = len(motion.t)
    require(
        0 <= events.address_idx <= events.impact_idx < n,
        "Events must satisfy 0 <= address <= impact < n",
    )

    # Resolve shoulder S, elbow E, wrist W
    s_pt: np.ndarray | None = None
    for lbl in ("LShoulderTop", "LShoulderBack"):
        if lbl in motion.markers:
            s_pt = _fill_nans_3d(motion.markers[lbl])
            break
    e_pt: np.ndarray | None = None
    if "LElbowOut" in motion.markers:
        e_pt = _fill_nans_3d(motion.markers["LElbowOut"])
    w_pt: np.ndarray | None = None
    if "LWristTop" in motion.markers:
        w_pt = _fill_nans_3d(motion.markers["LWristTop"])

    if s_pt is not None and e_pt is not None and w_pt is not None:
        v1 = s_pt - e_pt
        v2 = w_pt - e_pt
        dot = np.sum(v1 * v2, axis=-1)
        norm1 = np.linalg.norm(v1, axis=-1)
        norm2 = np.linalg.norm(v2, axis=-1)
        denom = np.where(norm1 * norm2 == 0, 1.0, norm1 * norm2)
        cos_ang = np.clip(dot / denom, -1.0, 1.0)
        elbow_angle = np.degrees(np.arccos(cos_ang))
    else:
        elbow_angle = np.full(n, 180.0, dtype=np.float64)

    # Address to impact slice
    ai_slice = elbow_angle[events.address_idx : events.impact_idx + 1]
    min_inc = float(np.nanmin(ai_slice)) if ai_slice.size > 0 else 180.0
    max_flex = float(180.0 - min_inc)

    return LeadArmMetrics(
        elbow_angle_deg=elbow_angle,
        elbow_angle_address=float(elbow_angle[events.address_idx]),
        elbow_angle_top=float(elbow_angle[events.top_idx]),
        elbow_angle_impact=float(elbow_angle[events.impact_idx]),
        min_included_angle=min_inc,
        max_flexion_deg=max_flex,
    )


def compute_wrist_metrics(
    motion: SwingMotion,
    events: SwingEvents,
) -> WristMetrics:
    """Compute lead wrist hinge angle at top of backswing and impact.

    Definitions:
    - Hinge angle: Angle between forearm vector (elbow -> wrist) and club shaft
      vector (grip -> clubhead) in degrees.
      ~90 deg at top of backswing indicates a full 90-degree wrist set/lag.
      ~0-15 deg at impact indicates shaft alignment with lead forearm.

    Preconditions:
        - 0 <= events.top_idx <= events.impact_idx < len(motion.t)

    Args:
        motion: SwingMotion instance.
        events: SwingEvents instance.

    Returns:
        WristMetrics with hinge angle time series and event values.
    """
    n = len(motion.t)
    require(
        0 <= events.top_idx <= events.impact_idx < n,
        "Events must satisfy 0 <= top <= impact < n",
    )

    # Forearm: elbow -> wrist
    e_pt = motion.markers.get("LElbowOut")
    w_pt = motion.markers.get("LWristTop")
    if e_pt is not None and w_pt is not None:
        v_forearm = _fill_nans_3d(w_pt) - _fill_nans_3d(e_pt)
    else:
        v_forearm = np.tile([0.0, 1.0, 0.0], (n, 1))

    # Shaft: grip -> head
    head_pt, grip_pt = _club_endpoints(motion)

    if head_pt is not None and grip_pt is not None:
        v_shaft = _fill_nans_3d(head_pt) - _fill_nans_3d(grip_pt)
    elif motion.shaft_axis is not None:
        v_shaft = motion.shaft_axis
    else:
        v_shaft = np.tile([0.0, 1.0, 0.0], (n, 1))

    dot = np.sum(v_forearm * v_shaft, axis=-1)
    n1 = np.linalg.norm(v_forearm, axis=-1)
    n2 = np.linalg.norm(v_shaft, axis=-1)
    denom = np.where(n1 * n2 == 0, 1.0, n1 * n2)
    cos_ang = np.clip(dot / denom, -1.0, 1.0)
    hinge_angle = np.degrees(np.arccos(cos_ang))

    return WristMetrics(
        hinge_angle_deg=hinge_angle,
        hinge_angle_top=float(hinge_angle[events.top_idx]),
        hinge_angle_impact=float(hinge_angle[events.impact_idx]),
    )


def compute_hand_path_metrics(
    motion: SwingMotion,
    events: SwingEvents,
) -> HandPathMetrics:
    """Compute grip path length and elevation metrics.

    Definitions:
    - path_length_total_m: Integrated 3D arc length of grip trajectory over full capture.
    - path_length_swing_m: Integrated 3D arc length of grip from address to finish.
    - max_height_top_m: Maximum Z height of grip at / near top of backswing.

    Preconditions:
        - 0 <= events.address_idx <= events.top_idx <= events.finish_idx < len(motion.t)

    Args:
        motion: SwingMotion instance.
        events: SwingEvents instance.

    Returns:
        HandPathMetrics instance.
    """
    n = len(motion.t)
    require(
        0 <= events.address_idx <= events.top_idx <= events.finish_idx < n,
        "Events must satisfy 0 <= address <= top <= finish < n",
    )

    grip = motion.grip
    if grip is None:
        for candidate in ("Marker_3:3:1", "LWristTop", "grip"):
            if candidate in motion.markers:
                grip = motion.markers[candidate]
                break
    if grip is None:
        grip = np.zeros((n, 3), dtype=np.float64)

    grip_clean = _fill_nans_3d(grip)

    # Arc lengths
    diffs = np.diff(grip_clean, axis=0)
    step_lens = np.linalg.norm(diffs, axis=-1)
    total_len = float(np.sum(step_lens))

    swing_step_lens = step_lens[events.address_idx : events.finish_idx]
    swing_len = float(np.sum(swing_step_lens)) if swing_step_lens.size > 0 else 0.0

    # Max height in backswing / top
    top_window = grip_clean[events.address_idx : events.top_idx + 1, 2]
    max_h_top = (
        float(np.nanmax(top_window))
        if top_window.size > 0
        else float(grip_clean[events.top_idx, 2])
    )

    return HandPathMetrics(
        path_length_total_m=total_len,
        path_length_swing_m=swing_len,
        max_height_top_m=max_h_top,
        height_address_m=float(grip_clean[events.address_idx, 2]),
        height_impact_m=float(grip_clean[events.impact_idx, 2]),
    )


def compute_club_metrics(
    motion: SwingMotion,
    events: SwingEvents,
) -> ClubMetrics:
    """Compute clubhead speed, shaft lean, and face angle at impact.

    Definitions:
    - Clubhead speed: ||d(clubhead)/dt|| at impact (m/s).
    - Shaft lean: Lean angle in the X-Z plane at impact in degrees. Forward lean
      (grip ahead of head along target line +X) is positive (delofting).
    - Face angle: Angle in degrees between clubface horizontal normal and target line (+X).
      Positive = open face (pointing to right of target line).

    Preconditions:
        - 0 <= events.impact_idx < len(motion.t)

    Args:
        motion: SwingMotion instance.
        events: SwingEvents instance.

    Returns:
        ClubMetrics instance.
    """
    t = motion.t
    n = len(t)
    require(0 <= events.impact_idx < n, "Events must satisfy 0 <= impact < n")

    # 1. Clubhead speed
    head_pt = motion.club_head
    if head_pt is None:
        for candidate in ("Marker_2:2:1", "club_head", "CH"):
            if candidate in motion.markers:
                head_pt = motion.markers[candidate]
                break
    if head_pt is None:
        head_pt = np.zeros((n, 3), dtype=np.float64)

    head_clean = _fill_nans_3d(head_pt)
    v_head = np.gradient(head_clean, t, axis=0)
    speed = np.linalg.norm(v_head, axis=-1)
    impact_speed = float(speed[events.impact_idx])
    peak_speed = float(np.nanmax(speed))

    # 2. Shaft lean at impact: arctan2(dx, dz) where dx = grip_x - head_x, dz = grip_z - head_z
    grip_pt = motion.grip
    if grip_pt is None:
        for candidate in ("Marker_3:3:1", "grip"):
            if candidate in motion.markers:
                grip_pt = motion.markers[candidate]
                break
    if grip_pt is None:
        grip_pt = head_clean + np.array([0.0, 0.0, 1.0])

    grip_clean = _fill_nans_3d(grip_pt)
    dx = float(grip_clean[events.impact_idx, 0] - head_clean[events.impact_idx, 0])
    dz = float(grip_clean[events.impact_idx, 2] - head_clean[events.impact_idx, 2])
    shaft_lean_deg = float(np.degrees(np.arctan2(dx, dz)))

    # 3. Face angle at impact relative to target line (+X)
    face_angle_deg: float | None = None
    if motion.face_normal is not None:
        fn = motion.face_normal[events.impact_idx]
        if np.all(np.isfinite(fn)):
            # Horizontal face angle relative to +X target line
            face_angle_deg = float(np.degrees(np.arctan2(fn[1], fn[0])))

    return ClubMetrics(
        impact_club_head_speed_m_s=impact_speed,
        peak_club_head_speed_m_s=peak_speed,
        impact_shaft_lean_deg=shaft_lean_deg,
        impact_face_angle_deg=face_angle_deg,
    )


def compute_all_metrics(
    motion: SwingMotion,
    events: SwingEvents | None = None,
) -> SwingMetrics:
    """Compute all swing comparison metrics for a SwingMotion instance.

    Args:
        motion: SwingMotion instance.
        events: Optional pre-computed SwingEvents; if None, detected automatically.

    Returns:
        SwingMetrics composite container.
    """
    ev = events if events is not None else detect_events(motion)
    return SwingMetrics(
        tempo=compute_tempo(ev),
        segment_rotation=compute_segment_rotations(motion, ev),
        kinematic_sequence=compute_kinematic_sequence(motion, ev),
        lead_arm=compute_lead_arm_metrics(motion, ev),
        wrist=compute_wrist_metrics(motion, ev),
        hand_path=compute_hand_path_metrics(motion, ev),
        club=compute_club_metrics(motion, ev),
    )


def _resample_phase(
    t: np.ndarray,
    data: np.ndarray,
    start_time: float,
    end_time: float,
    n_points: int,
) -> np.ndarray:
    """Resample 3D trajectory data over [start_time, end_time] onto n_points."""
    if end_time <= start_time:
        return np.repeat(data[0:1, :], n_points, axis=0)

    target_t = np.linspace(start_time, end_time, n_points)
    out = np.empty((n_points, data.shape[1]), dtype=np.float64)
    for col in range(data.shape[1]):
        out[:, col] = np.interp(target_t, t, data[:, col])
    return out


def _compute_scalar_metric_differences(
    metrics_a: SwingMetrics, metrics_b: SwingMetrics
) -> dict[str, float | None]:
    """Return the b-minus-a scalar differences for every compared metric."""

    def _seg_diff(attr: str) -> float:
        return float(
            getattr(metrics_b.segment_rotation, attr)
            - getattr(metrics_a.segment_rotation, attr)
        )

    diffs: dict[str, float | None] = {
        "backswing_time_s": metrics_b.tempo.backswing_time
        - metrics_a.tempo.backswing_time,
        "downswing_time_s": metrics_b.tempo.downswing_time
        - metrics_a.tempo.downswing_time,
        "tempo_ratio": metrics_b.tempo.tempo_ratio - metrics_a.tempo.tempo_ratio,
        "pelvis_yaw_top_deg": metrics_b.segment_rotation.pelvis_yaw_top
        - metrics_a.segment_rotation.pelvis_yaw_top,
        "pelvis_yaw_impact_deg": (
            metrics_b.segment_rotation.pelvis_yaw_impact
            - metrics_a.segment_rotation.pelvis_yaw_impact
        ),
        "shoulder_girdle_yaw_top_deg": _seg_diff("shoulder_girdle_yaw_top"),
        "shoulder_girdle_yaw_impact_deg": _seg_diff("shoulder_girdle_yaw_impact"),
        "upper_trunk_yaw_top_deg": _seg_diff("upper_trunk_yaw_top"),
        "upper_trunk_yaw_impact_deg": _seg_diff("upper_trunk_yaw_impact"),
        # Deprecated keys: shoulder-girdle values, kept for existing consumers.
        "thorax_yaw_top_deg": _seg_diff("shoulder_girdle_yaw_top"),
        "thorax_yaw_impact_deg": _seg_diff("shoulder_girdle_yaw_impact"),
        "x_factor_address_deg": (
            metrics_b.segment_rotation.x_factor_address
            - metrics_a.segment_rotation.x_factor_address
        ),
        "x_factor_top_deg": metrics_b.segment_rotation.x_factor_top
        - metrics_a.segment_rotation.x_factor_top,
        "x_factor_impact_deg": (
            metrics_b.segment_rotation.x_factor_impact
            - metrics_a.segment_rotation.x_factor_impact
        ),
        "x_factor_stretch_deg": (
            metrics_b.segment_rotation.x_factor_stretch
            - metrics_a.segment_rotation.x_factor_stretch
        ),
        "pelvis_peak_speed_deg_s": (
            metrics_b.peak_speed("pelvis") - metrics_a.peak_speed("pelvis")
        ),
        "thorax_peak_speed_deg_s": (
            metrics_b.peak_speed("thorax") - metrics_a.peak_speed("thorax")
        ),
        "lead_arm_peak_speed_deg_s": (
            metrics_b.peak_speed("lead_arm") - metrics_a.peak_speed("lead_arm")
        ),
        "club_peak_speed_deg_s": (
            metrics_b.peak_speed("club") - metrics_a.peak_speed("club")
        ),
        "elbow_min_included_angle_deg": (
            metrics_b.lead_arm.min_included_angle
            - metrics_a.lead_arm.min_included_angle
        ),
        "elbow_max_flexion_deg": metrics_b.lead_arm.max_flexion_deg
        - metrics_a.lead_arm.max_flexion_deg,
        "wrist_hinge_top_deg": metrics_b.wrist.hinge_angle_top
        - metrics_a.wrist.hinge_angle_top,
        "wrist_hinge_impact_deg": metrics_b.wrist.hinge_angle_impact
        - metrics_a.wrist.hinge_angle_impact,
        "hand_path_length_swing_m": (
            metrics_b.hand_path.path_length_swing_m
            - metrics_a.hand_path.path_length_swing_m
        ),
        "hand_max_height_top_m": metrics_b.hand_path.max_height_top_m
        - metrics_a.hand_path.max_height_top_m,
        "impact_club_head_speed_m_s": (
            metrics_b.club.impact_club_head_speed_m_s
            - metrics_a.club.impact_club_head_speed_m_s
        ),
        "peak_club_head_speed_m_s": (
            metrics_b.club.peak_club_head_speed_m_s
            - metrics_a.club.peak_club_head_speed_m_s
        ),
        "impact_shaft_lean_deg": metrics_b.club.impact_shaft_lean_deg
        - metrics_a.club.impact_shaft_lean_deg,
        "impact_face_angle_deg": (
            metrics_b.club.impact_face_angle_deg - metrics_a.club.impact_face_angle_deg
            if (
                metrics_b.club.impact_face_angle_deg is not None
                and metrics_a.club.impact_face_angle_deg is not None
            )
            else None
        ),
    }
    # Unavailable turn lines are NaN; report "no comparison" instead of a NaN.
    turn_prefixes = ("shoulder_girdle", "upper_trunk", "thorax", "x_factor")
    for key, value in diffs.items():
        if key.startswith(turn_prefixes) and value is not None and value != value:
            diffs[key] = None
    return diffs


def _compute_shared_marker_rms(
    a: SwingMotion,
    b: SwingMotion,
    events_a: SwingEvents,
    events_b: SwingEvents,
) -> dict[str, float]:
    """Return per-marker RMSE between time-normalized Address->Impact->Finish phases."""
    shared_labels = sorted(set(a.markers.keys()) & set(b.markers.keys()))
    shared_rms: dict[str, float] = {}

    n_phase1 = 100
    n_phase2 = 50

    for lbl in shared_labels:
        pts_a = _fill_nans_3d(a.markers[lbl])
        pts_b = _fill_nans_3d(b.markers[lbl])

        p1_a = _resample_phase(
            a.t, pts_a, events_a.address_time, events_a.impact_time, n_phase1
        )
        p2_a = _resample_phase(
            a.t, pts_a, events_a.impact_time, events_a.finish_time, n_phase2
        )
        norm_a = np.concatenate([p1_a, p2_a], axis=0)

        p1_b = _resample_phase(
            b.t, pts_b, events_b.address_time, events_b.impact_time, n_phase1
        )
        p2_b = _resample_phase(
            b.t, pts_b, events_b.impact_time, events_b.finish_time, n_phase2
        )
        norm_b = np.concatenate([p1_b, p2_b], axis=0)

        err_sq = np.sum((norm_b - norm_a) ** 2, axis=-1)
        shared_rms[lbl] = float(np.sqrt(np.mean(err_sq)))

    return shared_rms


def compare(a: SwingMotion, b: SwingMotion) -> ComparisonReport:
    """Compare two swings, computing all metrics, differences, and time-normalized RMS.

    Time-normalization aligns each swing across Address -> Impact -> Finish:
    - Phase 1 (Address to Impact): 100 normalized points
    - Phase 2 (Impact to Finish): 50 normalized points
    For any marker present in both motions, the Root Mean Square Error (RMSE)
    across the aligned phases is computed.

    Args:
        a: Reference swing motion.
        b: Candidate swing motion.

    Returns:
        ComparisonReport containing metrics for both, differences, and trajectory RMS.
    """
    events_a = detect_events(a)
    events_b = detect_events(b)

    metrics_a = compute_all_metrics(a, events_a)
    metrics_b = compute_all_metrics(b, events_b)

    differences = _compute_scalar_metric_differences(metrics_a, metrics_b)
    shared_rms = _compute_shared_marker_rms(a, b, events_a, events_b)
    mean_rms = float(np.mean(list(shared_rms.values()))) if shared_rms else 0.0

    return ComparisonReport(
        motion_a_events=events_a,
        motion_b_events=events_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        differences=differences,
        shared_marker_rms=shared_rms,
        mean_marker_rms=mean_rms,
    )
