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

from src.shared.python.swing_comparison.motion import (
    CAPTURE_A_PELVIS_LEFT_LABELS,
    CAPTURE_A_PELVIS_RIGHT_LABELS,
    CAPTURE_A_SHOULDER_LEFT_LABELS,
    CAPTURE_A_SHOULDER_RIGHT_LABELS,
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
    """Pelvis and thorax rotation, and X-factor separation.

    All angles in degrees. Yaw is rotation about vertical +Z in the XY ground plane.

    Attributes:
        pelvis_yaw: Pelvis yaw angle time series (N,) in degrees.
        thorax_yaw: Thorax yaw angle time series (N,) in degrees.
        x_factor: X-factor (thorax yaw minus pelvis yaw) time series (N,) in degrees.
        pelvis_yaw_address: Pelvis yaw at address in degrees.
        pelvis_yaw_top: Pelvis yaw at top of backswing in degrees.
        pelvis_yaw_impact: Pelvis yaw at impact in degrees.
        thorax_yaw_address: Thorax yaw at address in degrees.
        thorax_yaw_top: Thorax yaw at top of backswing in degrees.
        thorax_yaw_impact: Thorax yaw at impact in degrees.
        x_factor_address: X-factor at address in degrees.
        x_factor_top: X-factor at top of backswing in degrees.
        x_factor_impact: X-factor at impact in degrees.
        x_factor_stretch: Maximum absolute X-factor magnitude during the swing in degrees.
    """

    pelvis_yaw: np.ndarray
    thorax_yaw: np.ndarray
    x_factor: np.ndarray
    pelvis_yaw_address: float
    pelvis_yaw_top: float
    pelvis_yaw_impact: float
    thorax_yaw_address: float
    thorax_yaw_top: float
    thorax_yaw_impact: float
    x_factor_address: float
    x_factor_top: float
    x_factor_impact: float
    x_factor_stretch: float

    def to_dict(self) -> dict[str, float]:
        return {
            "pelvis_yaw_address_deg": float(self.pelvis_yaw_address),
            "pelvis_yaw_top_deg": float(self.pelvis_yaw_top),
            "pelvis_yaw_impact_deg": float(self.pelvis_yaw_impact),
            "thorax_yaw_address_deg": float(self.thorax_yaw_address),
            "thorax_yaw_top_deg": float(self.thorax_yaw_top),
            "thorax_yaw_impact_deg": float(self.thorax_yaw_impact),
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
    """

    pelvis: SegmentKinematicPeak
    thorax: SegmentKinematicPeak
    lead_arm: SegmentKinematicPeak
    club: SegmentKinematicPeak
    order: tuple[str, ...]
    is_proximal_to_distal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pelvis": self.pelvis.to_dict(),
            "thorax": self.thorax.to_dict(),
            "lead_arm": self.lead_arm.to_dict(),
            "club": self.club.to_dict(),
            "order": list(self.order),
            "is_proximal_to_distal": bool(self.is_proximal_to_distal),
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
    """Compute pelvis yaw, thorax yaw, and X-factor metrics.

    Definitions:
    - Pelvis yaw: Rotation angle in the horizontal XY plane from hip markers (WaistLeft -> WaistRight),
      measured relative to the address orientation in degrees.
    - Thorax yaw: Rotation angle in the horizontal XY plane from shoulder markers (LShoulder -> RShoulder),
      measured relative to the address orientation in degrees.
    - X-factor: Separation angle (thorax yaw minus pelvis yaw) in degrees.
    - X-factor stretch: Maximum absolute X-factor separation during the swing in degrees.

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

    # 1. Pelvis vector (right hip to left hip)
    pelvis_vec = _resolve_vector(
        motion.markers,
        CAPTURE_A_PELVIS_LEFT_LABELS,
        CAPTURE_A_PELVIS_RIGHT_LABELS,
        fallback_dim=n,
    )
    raw_p_yaw = np.unwrap(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))
    pelvis_yaw = np.degrees(raw_p_yaw - raw_p_yaw[events.address_idx])

    # 2. Thorax vector (right shoulder to left shoulder)
    thorax_vec = _resolve_vector(
        motion.markers,
        CAPTURE_A_SHOULDER_LEFT_LABELS,
        CAPTURE_A_SHOULDER_RIGHT_LABELS,
        fallback_dim=n,
    )
    raw_t_yaw = np.unwrap(np.arctan2(thorax_vec[:, 1], thorax_vec[:, 0]))
    thorax_yaw = np.degrees(raw_t_yaw - raw_t_yaw[events.address_idx])

    # 3. X-Factor (Thorax - Pelvis)
    x_factor = thorax_yaw - pelvis_yaw
    stretch = float(np.nanmax(np.abs(x_factor)))

    return SegmentRotationMetrics(
        pelvis_yaw=pelvis_yaw,
        thorax_yaw=thorax_yaw,
        x_factor=x_factor,
        pelvis_yaw_address=float(pelvis_yaw[events.address_idx]),
        pelvis_yaw_top=float(pelvis_yaw[events.top_idx]),
        pelvis_yaw_impact=float(pelvis_yaw[events.impact_idx]),
        thorax_yaw_address=float(thorax_yaw[events.address_idx]),
        thorax_yaw_top=float(thorax_yaw[events.top_idx]),
        thorax_yaw_impact=float(thorax_yaw[events.impact_idx]),
        x_factor_address=float(x_factor[events.address_idx]),
        x_factor_top=float(x_factor[events.top_idx]),
        x_factor_impact=float(x_factor[events.impact_idx]),
        x_factor_stretch=stretch,
    )


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

    # 1. Pelvis angular speed
    pelvis_vec = _resolve_vector(
        motion.markers,
        CAPTURE_A_PELVIS_LEFT_LABELS,
        CAPTURE_A_PELVIS_RIGHT_LABELS,
        fallback_dim=n,
    )
    p_yaw = np.unwrap(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))
    omega_pelvis = np.degrees(np.abs(np.gradient(p_yaw, t)))

    # 2. Thorax angular speed
    thorax_vec = _resolve_vector(
        motion.markers,
        CAPTURE_A_SHOULDER_LEFT_LABELS,
        CAPTURE_A_SHOULDER_RIGHT_LABELS,
        fallback_dim=n,
    )
    t_yaw = np.unwrap(np.arctan2(thorax_vec[:, 1], thorax_vec[:, 0]))
    omega_thorax = np.degrees(np.abs(np.gradient(t_yaw, t)))

    # 3. Lead arm angular speed (shoulder to wrist)
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
    if sh_pt is not None and wrist_pt is not None:
        arm_v = wrist_pt - sh_pt
        norms = np.linalg.norm(arm_v, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        u_arm = arm_v / norms
        du_arm = np.gradient(u_arm, t, axis=0)
        omega_arm = np.degrees(np.linalg.norm(du_arm, axis=-1))
    else:
        omega_arm = np.zeros(n, dtype=np.float64)

    # 4. Club angular speed
    head_pt = motion.club_head
    if head_pt is None and "Marker_2:2:1" in motion.markers:
        head_pt = motion.markers["Marker_2:2:1"]
    grip_pt = motion.grip
    if grip_pt is None and "Marker_3:3:1" in motion.markers:
        grip_pt = motion.markers["Marker_3:3:1"]

    if head_pt is not None and grip_pt is not None:
        shaft_v = _fill_nans_3d(head_pt) - _fill_nans_3d(grip_pt)
        norms = np.linalg.norm(shaft_v, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        u_shaft = shaft_v / norms
        du_shaft = np.gradient(u_shaft, t, axis=0)
        omega_club = np.degrees(np.linalg.norm(du_shaft, axis=-1))
    else:
        omega_club = np.zeros(n, dtype=np.float64)

    # Search window: downswing plus small follow-through margin
    margin = max(3, int(0.15 * (events.impact_idx - events.top_idx) + 5))
    w_start = events.top_idx
    w_end = min(n, events.impact_idx + margin)
    window = slice(w_start, w_end)

    def extract_peak(name: str, omega: np.ndarray) -> SegmentKinematicPeak:
        sub = omega[window]
        if sub.size == 0 or np.all(np.isnan(sub)):
            return SegmentKinematicPeak(
                name, 0.0, float(t[events.top_idx]), events.top_idx
            )
        peak_val = float(np.nanmax(sub))
        local_idx = int(np.nanargmax(sub))
        global_idx = w_start + local_idx
        return SegmentKinematicPeak(name, peak_val, float(t[global_idx]), global_idx)

    peak_p = extract_peak("pelvis", omega_pelvis)
    peak_t = extract_peak("thorax", omega_thorax)
    peak_a = extract_peak("lead_arm", omega_arm)
    peak_c = extract_peak("club", omega_club)

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
    head_pt = motion.club_head
    if head_pt is None and "Marker_2:2:1" in motion.markers:
        head_pt = motion.markers["Marker_2:2:1"]
    grip_pt = motion.grip
    if grip_pt is None and "Marker_3:3:1" in motion.markers:
        grip_pt = motion.markers["Marker_3:3:1"]

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

    # 1. Scalar differences (b - a)
    differences: dict[str, float | None] = {
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
        "thorax_yaw_top_deg": metrics_b.segment_rotation.thorax_yaw_top
        - metrics_a.segment_rotation.thorax_yaw_top,
        "thorax_yaw_impact_deg": (
            metrics_b.segment_rotation.thorax_yaw_impact
            - metrics_a.segment_rotation.thorax_yaw_impact
        ),
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
            metrics_b.kinematic_sequence.pelvis.peak_speed
            - metrics_a.kinematic_sequence.pelvis.peak_speed
        ),
        "thorax_peak_speed_deg_s": (
            metrics_b.kinematic_sequence.thorax.peak_speed
            - metrics_a.kinematic_sequence.thorax.peak_speed
        ),
        "lead_arm_peak_speed_deg_s": (
            metrics_b.kinematic_sequence.lead_arm.peak_speed
            - metrics_a.kinematic_sequence.lead_arm.peak_speed
        ),
        "club_peak_speed_deg_s": (
            metrics_b.kinematic_sequence.club.peak_speed
            - metrics_a.kinematic_sequence.club.peak_speed
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

    # 2. Time-normalized trajectory RMS of shared markers
    shared_labels = sorted(set(a.markers.keys()) & set(b.markers.keys()))
    shared_rms: dict[str, float] = {}

    n_phase1 = 100
    n_phase2 = 50

    for lbl in shared_labels:
        pts_a = _fill_nans_3d(a.markers[lbl])
        pts_b = _fill_nans_3d(b.markers[lbl])

        # Normalize A
        p1_a = _resample_phase(
            a.t, pts_a, events_a.address_time, events_a.impact_time, n_phase1
        )
        p2_a = _resample_phase(
            a.t, pts_a, events_a.impact_time, events_a.finish_time, n_phase2
        )
        norm_a = np.concatenate([p1_a, p2_a], axis=0)

        # Normalize B
        p1_b = _resample_phase(
            b.t, pts_b, events_b.address_time, events_b.impact_time, n_phase1
        )
        p2_b = _resample_phase(
            b.t, pts_b, events_b.impact_time, events_b.finish_time, n_phase2
        )
        norm_b = np.concatenate([p1_b, p2_b], axis=0)

        err_sq = np.sum((norm_b - norm_a) ** 2, axis=-1)
        rms = float(np.sqrt(np.mean(err_sq)))
        shared_rms[lbl] = rms

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
