"""Posture, translation, impact, and GRF moment metrics (GM-5 #14428).

Covers:
1. Trunk forward tilt and lateral bend across Chu et al. (2010) standard events.
2. Pelvis and thorax sway, thrust, and lift translations and linear velocities.
3. Clubhead speed at impact joined to body kinematics in a unified summary.
4. GRF pivoting moment (Han et al. 2019) with strict provenance enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.contracts import ensure, require

DEFAULT_LAB_FRAME = "X_target_Y_ball_Z_up"


@dataclass(frozen=True)
class ChuEventFrames:
    """Frame indices of Chu et al. (2010) standard golf swing events."""

    address: int
    top: int
    mid_downswing: int
    impact: int
    follow_through: int

    def __post_init__(self) -> None:
        valid = (
            0
            <= self.address
            <= self.top
            <= self.mid_downswing
            <= self.impact
            <= self.follow_through
        )
        require(
            valid,
            f"Frames must satisfy 0 <= addr <= top <= mid <= imp <= ft; got {self}",
        )


@dataclass(frozen=True)
class ChuEventPosture:
    """Trunk posture angles at a single swing event."""

    event_name: str
    frame_index: int
    forward_tilt_deg: float
    lateral_bend_deg: float
    axial_rotation_deg: float = 0.0


@dataclass(frozen=True)
class TrunkPostureEventReport:
    """Trunk posture angles across Chu et al. (2010) events."""

    address: ChuEventPosture
    top_of_backswing: ChuEventPosture
    mid_downswing: ChuEventPosture
    impact: ChuEventPosture
    follow_through: ChuEventPosture
    lab_frame: str = DEFAULT_LAB_FRAME


@dataclass(frozen=True)
class SegmentTranslationKinematics:
    """Address-relative translation (sway, thrust, lift) and linear velocities."""

    segment: str
    sway_m: tuple[float, ...]
    thrust_m: tuple[float, ...]
    lift_m: tuple[float, ...]
    sway_velocity_mps: tuple[float, ...]
    thrust_velocity_mps: tuple[float, ...]
    lift_velocity_mps: tuple[float, ...]
    reference_frame: str = DEFAULT_LAB_FRAME


@dataclass(frozen=True)
class ImpactBodyClubSummary:
    """Joined record of clubhead speed and body kinematics at impact."""

    club_speed_mps: float
    impact_frame: int
    impact_time_s: float | None
    pelvis_sway_m: float
    pelvis_thrust_m: float
    pelvis_lift_m: float
    pelvis_sway_velocity_mps: float
    pelvis_thrust_velocity_mps: float
    pelvis_lift_velocity_mps: float
    thorax_sway_m: float
    thorax_thrust_m: float
    thorax_lift_m: float
    thorax_sway_velocity_mps: float
    thorax_thrust_velocity_mps: float
    thorax_lift_velocity_mps: float
    trunk_forward_tilt_deg: float
    trunk_lateral_bend_deg: float
    lab_frame: str = DEFAULT_LAB_FRAME


@dataclass(frozen=True)
class HanPivotingMoment:
    """Ground reaction moment / pivoting moment per Han et al. (2019)."""

    pivoting_moment_nm: float
    source: str
    foot: str = "trail"
    axis: str = "vertical"
    plate_data_path: str | None = None
    reference: str = "Han et al. (2019)"


def _as_vector3(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    require(arr.shape == (3,), f"{name} must have shape (3,), got {arr.shape}", arr)
    require(bool(np.all(np.isfinite(arr))), f"{name} must be finite", arr)
    return arr


def _validated_frame(
    target_line: ArrayLike, toward_ball: ArrayLike, tolerance: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal triad (target, toward_ball, vertical)."""
    target = _as_vector3(target_line, "target_line")
    ball = _as_vector3(toward_ball, "toward_ball")
    t_norm = float(np.linalg.norm(target))
    b_norm = float(np.linalg.norm(ball))
    require(abs(t_norm - 1.0) <= tolerance, "target_line must be unit vector", t_norm)
    require(abs(b_norm - 1.0) <= tolerance, "toward_ball must be unit vector", b_norm)
    dot = float(target @ ball)
    require(
        abs(dot) <= tolerance, "target_line and toward_ball must be orthogonal", dot
    )
    return target, ball, np.cross(target, ball)


def _calc_posture(
    v: np.ndarray,
    ball: np.ndarray,
    target: np.ndarray,
    vert: np.ndarray,
    name: str,
    f: int,
) -> ChuEventPosture:
    f_tilt = math.degrees(math.atan2(float(v @ ball), abs(float(v @ vert))))
    l_bend = math.degrees(math.atan2(float(v @ target), abs(float(v @ vert))))
    return ChuEventPosture(name, f, f_tilt, l_bend)


def compute_trunk_posture_events(
    trunk_axes: ArrayLike,
    events: ChuEventFrames,
    target_line: ArrayLike,
    toward_ball: ArrayLike,
    *,
    lab_frame: str = DEFAULT_LAB_FRAME,
    tolerance: float = 1e-6,
) -> TrunkPostureEventReport:
    """Compute trunk forward tilt and lateral bend across Chu et al. (2010) events."""
    axes = np.asarray(trunk_axes, dtype=float)
    require(axes.ndim == 2 and axes.shape[1] == 3, "trunk_axes must have shape (N, 3)")
    require(
        len(axes) > events.follow_through, "trunk_axes length insufficient for events"
    )
    target, ball, vert = _validated_frame(target_line, toward_ball, tolerance)

    ev_pairs = (
        ("address", events.address),
        ("top_of_backswing", events.top),
        ("mid_downswing", events.mid_downswing),
        ("impact", events.impact),
        ("follow_through", events.follow_through),
    )
    p = [_calc_posture(axes[f], ball, target, vert, name, f) for name, f in ev_pairs]
    return TrunkPostureEventReport(p[0], p[1], p[2], p[3], p[4], lab_frame=lab_frame)


def compute_segment_translations(
    positions: ArrayLike,
    address_frame: int,
    dt: float,
    segment_name: str,
    target_line: ArrayLike,
    toward_ball: ArrayLike,
    *,
    lab_frame: str = DEFAULT_LAB_FRAME,
) -> SegmentTranslationKinematics:
    """Compute address-relative sway, thrust, lift translations and linear velocities."""
    pos = np.asarray(positions, dtype=float)
    require(pos.ndim == 2 and pos.shape[1] == 3, "positions must have shape (N, 3)")
    require(dt > 0.0, "dt must be positive", dt)
    require(0 <= address_frame < len(pos), "address_frame out of bounds", address_frame)
    target, ball, vert = _validated_frame(target_line, toward_ball)

    delta = pos - pos[address_frame]
    sway = delta @ target
    thrust = delta @ ball
    lift = delta @ vert

    return SegmentTranslationKinematics(
        segment=segment_name,
        sway_m=tuple(float(x) for x in sway),
        thrust_m=tuple(float(x) for x in thrust),
        lift_m=tuple(float(x) for x in lift),
        sway_velocity_mps=tuple(float(x) for x in np.gradient(sway, dt)),
        thrust_velocity_mps=tuple(float(x) for x in np.gradient(thrust, dt)),
        lift_velocity_mps=tuple(float(x) for x in np.gradient(lift, dt)),
        reference_frame=lab_frame,
    )


def join_impact_metrics(
    club_speed_mps: float,
    impact_frame: int,
    pelvis: SegmentTranslationKinematics,
    thorax: SegmentTranslationKinematics,
    trunk: TrunkPostureEventReport,
    *,
    impact_time_s: float | None = None,
    lab_frame: str = DEFAULT_LAB_FRAME,
) -> ImpactBodyClubSummary:
    """Join clubhead speed at impact to body kinematic translations and postures."""
    require(
        club_speed_mps >= 0.0, "club_speed_mps must be non-negative", club_speed_mps
    )
    require(
        0 <= impact_frame < len(pelvis.sway_m), "impact_frame out of range for pelvis"
    )
    require(
        0 <= impact_frame < len(thorax.sway_m), "impact_frame out of range for thorax"
    )

    idx = impact_frame
    t_imp = trunk.impact
    return ImpactBodyClubSummary(
        float(club_speed_mps),
        impact_frame,
        impact_time_s,
        pelvis.sway_m[idx],
        pelvis.thrust_m[idx],
        pelvis.lift_m[idx],
        pelvis.sway_velocity_mps[idx],
        pelvis.thrust_velocity_mps[idx],
        pelvis.lift_velocity_mps[idx],
        thorax.sway_m[idx],
        thorax.thrust_m[idx],
        thorax.lift_m[idx],
        thorax.sway_velocity_mps[idx],
        thorax.thrust_velocity_mps[idx],
        thorax.lift_velocity_mps[idx],
        t_imp.forward_tilt_deg,
        t_imp.lateral_bend_deg,
        lab_frame,
    )


def evaluate_pivoting_moment(
    pivoting_moment_nm: float,
    source: str,
    *,
    foot: str = "trail",
    axis: str = "vertical",
    plate_data_path: str | Path | None = None,
) -> HanPivotingMoment:
    """Evaluate Han et al. (2019) GRF pivoting moment with strict provenance contract."""
    require(math.isfinite(pivoting_moment_nm), "pivoting_moment_nm must be finite")
    valid_sources = ("measured_force_plate", "modelled")
    require(source in valid_sources, f"source must be in {valid_sources}, got {source}")

    path_str: str | None = None
    if source == "measured_force_plate":
        require(
            plate_data_path is not None, "measured_force_plate requires plate_data_path"
        )
        path_str = str(plate_data_path).strip()
        require(
            len(path_str) > 0, "measured_force_plate requires non-empty plate_data_path"
        )
    elif plate_data_path is not None:
        path_str = str(plate_data_path).strip()

    record = HanPivotingMoment(float(pivoting_moment_nm), source, foot, axis, path_str)
    ensure(record.source == source, "source must match declared parameter")
    return record
