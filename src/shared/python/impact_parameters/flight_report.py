"""Ball-flight report of one impact state for the showpiece render (SHOW-C7, #14563).

A reporting layer over existing models, with no new physics: the clubhead
state at impact goes through :class:`SwingBallFlightPipeline` (club-ball impact
solver, then aerodynamic flight).  The pipeline frame is x down-range, y left,
z up, which is :meth:`TargetFrame.components` of the UD world frame.

Only carry is computed: the repository has no bounce/roll model, so ``total``
is reported unavailable with the reason, never invented.  The ``lateral_m``
sign is + = golfer's right of the target line (an open face goes right).
The clubhead angular velocity is passed as zero (the series carries none), so
gear-effect spin is not modelled.  A flight outside the simulator's validity
(no spin decay: extreme sidespin turns the ball past 90 degrees) is flagged
``valid=False`` and never presented as a result.  The 45 degree landing-azimuth
limit is an assumption (a real driver lands well inside it), not a measured bound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.physics.ball_flight_validity import (
    assess_trajectory_validity,
    launch_spin_envelope,
    report_margin_rpm,
)
from src.shared.python.physics.swing_ball_flight_pipeline import (
    SwingBallFlightPipeline,
    SwingState,
)

from src.shared.python.physics.impact_model.types import ImpactParameters

from .club_impact import ClubImpact, rigid_gear_head_for
from .clubhead_series import ClubheadSeries
from .conditions import ImpactConditions
from .target_frame import TargetFrame

_MPS_TO_MPH = 2.2369362920544
TOTAL_UNAVAILABLE = "no bounce/roll model in the repository (carry only)"


@dataclass(frozen=True)
class FlightSummary:
    """Reported ball flight; ``total_m`` is ``None`` with ``total_unavailable``."""

    carry_m: float
    lateral_m: float
    max_height_m: float
    flight_time_s: float
    ball_speed_mph: float
    launch_deg: float
    spin_rpm: float
    down_range_m: float
    trajectory_m: np.ndarray  # (n, 3) down-range, left, up
    total_m: float | None = None
    total_unavailable: str = TOTAL_UNAVAILABLE
    valid: bool = True
    invalid_reason: str = ""
    #: seconds after launch of each ``trajectory_m`` point (``None``: not recorded)
    times_s: np.ndarray | None = None
    #: launch spin-axis tilt, degrees, + = right (``None``: not reported)
    spin_axis_deg: float | None = None


def spin_axis_tilt_deg(axis: np.ndarray) -> float:
    """Tilt of a launch spin axis from pure backspin, degrees, + = right.

    ``axis`` is in the pipeline frame (x down-range, y left, z up), where pure
    backspin is about -y; an axis dipped below horizontal on the right
    (negative z) curves the ball right, the launch-monitor sign. Raises
    ``ValueError`` for an axis that is zero or not finite.
    """
    vec = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vec))
    if vec.shape != (3,) or not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"spin axis must be a finite non-zero 3-vector, got {axis}")
    return math.degrees(math.asin(max(-1.0, min(1.0, -float(vec[2]) / norm))))


def _invalid_reason(path: np.ndarray) -> str:
    """Delegate to :func:`~physics.ball_flight_validity.assess_trajectory_validity`."""
    return assess_trajectory_validity(path).invalid_reason


def swing_state_from_series(
    series: ClubheadSeries,
    index: int,
    frame: TargetFrame,
    club: ClubImpact | None = None,
    impact_offset_m: tuple[float, float] | None = None,
) -> SwingState:
    """Clubhead state at sample ``index`` in the pipeline frame.

    Raises:
        ValueError: ``index`` out of range or the face normal is unobservable.

    ``club`` sets head mass and inertia (``None``: pipeline defaults);
    ``impact_offset_m`` is the strike (toe, high) from the face centre.
    """
    require(0 <= index < len(series), "index out of range", index)
    face_normals = series.face_normal
    if face_normals is None:
        raise ValueError(f"face normal unavailable: {series.face_unobservable_reason}")
    velocity = np.asarray(frame.components(series.velocity_mps[index]), dtype=float)
    normal = np.asarray(frame.components(face_normals[index]), dtype=float)
    normal /= np.linalg.norm(normal)
    extra: dict[str, object] = {}
    if club is not None:
        extra = {
            "clubhead_mass": club.head_mass_kg,
            "clubhead_moi": club.head_moi_kg_m2,
        }
    if impact_offset_m is not None:
        extra["impact_offset"] = np.asarray(impact_offset_m, dtype=float)
    return SwingState(
        clubhead_velocity=velocity,
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal,
        clubhead_loft_deg=math.degrees(math.asin(float(normal[2]))),
        engine_name="mujoco",
        **extra,  # type: ignore[arg-type]
    )


def spin_envelope_reason(
    meta: Mapping[str, Any], launch_spin_rpm: float, club: str | None
) -> str:
    """Invalid reason when the impact spin beat the club class envelope, else "".

    Preconditions: ``meta`` is the pipeline result metadata.
    Postcondition: the reason names the class of ``club``, not always "driver".
    """
    if not meta.get("launch_spin_clamped"):
        return ""
    envelope = launch_spin_envelope(club)
    pre_clamp_rpm = float(meta.get("pre_clamp_launch_spin_rpm", launch_spin_rpm))
    if pre_clamp_rpm <= envelope.max_rpm + report_margin_rpm():
        return ""
    return (
        f"impact spin exceeds the {envelope.label} validity envelope "
        f"({pre_clamp_rpm:.0f} rpm); trajectory after capping is not presented "
        "(SHOW-C7 / #14621)"
    )


def flight_for_impact(
    series: ClubheadSeries,
    index: int,
    frame: TargetFrame,
    club: str | None = None,
    club_impact: ClubImpact | None = None,
    impact_offset_m: tuple[float, float] | None = None,
) -> FlightSummary:
    """Carry flight of the impact state at ``index`` (impact solver + aerodynamics).

    ``club`` names the club; its class sets the launch-spin validity envelope
    (``None``: the driver envelope). ``club_impact`` and ``impact_offset_m`` as
    in :func:`swing_state_from_series`; the club's COR, evaluated at the
    face's normal approach speed (ball at rest), replaces the pipeline
    default. An off-centre strike's gear spin uses the rigid-contact law with
    the club's head from :func:`club_impact.rigid_gear_head_for` (#15443).
    """
    state = swing_state_from_series(series, index, frame, club_impact, impact_offset_m)
    cor: dict[str, float] = {}
    if club_impact is not None:
        normal_speed = float(
            np.dot(state.clubhead_velocity, state.clubhead_orientation)
        )
        cor = {"cor": club_impact.cor_at(max(normal_speed, 0.0))}
    params = ImpactParameters(**cor, rigid_gear_head=rigid_gear_head_for(club))
    result = SwingBallFlightPipeline(impact_params=params).run(state, club=club)
    path = np.array([p.position for p in result.trajectory], dtype=float)
    launch = result.launch_conditions
    meta = result.metadata
    if not bool(meta.get("flight_valid", True)):
        reason = str(meta.get("flight_invalid_reason", "") or "")
    else:
        reason = _invalid_reason(path)
        if not reason:
            reason = spin_envelope_reason(meta, float(launch.spin_rate), club)
    summary = FlightSummary(
        carry_m=float(result.carry_m),
        lateral_m=-float(path[-1, 1]),
        max_height_m=float(result.max_height_m),
        flight_time_s=float(result.flight_time_s),
        ball_speed_mph=float(launch.velocity) * _MPS_TO_MPH,
        launch_deg=math.degrees(launch.launch_angle),
        spin_rpm=float(launch.spin_rate),
        down_range_m=float(path[-1, 0]),
        trajectory_m=path,
        valid=not reason,
        invalid_reason=reason,
        spin_axis_deg=spin_axis_tilt_deg(launch.spin_axis),
        times_s=np.array([p.time for p in result.trajectory], dtype=float),
    )
    ensure(math.isfinite(summary.carry_m), "carry must be finite", summary.carry_m)
    return summary


def _fmt(value: float | None, unit: str = "deg") -> str:
    return "n/a" if value is None else f"{value:+.1f} {unit}"


def overlay_lines(
    title: str, conditions: ImpactConditions, flight: FlightSummary
) -> list[str]:
    """Text overlay: what the render is, the impact conditions, carry/total."""
    total = (
        f"{flight.total_m:.0f} m"
        if flight.total_m is not None
        else f"n/a ({flight.total_unavailable})"
    )
    if not flight.valid:
        carry = f"FLIGHT INVALID: {flight.invalid_reason}; carry n/a, total {total}"
    else:
        carry = (
            f"carry {flight.carry_m:.0f} m (down-range {flight.down_range_m:.0f} m)"
            f"   lateral {flight.lateral_m:+.0f} m   apex {flight.max_height_m:.0f} m"
            f"   total {total}"
        )
    return [
        title,
        f"clubhead {conditions.speed_mph:.1f} mph   AoA {conditions.aoa_deg:+.1f} deg"
        f"   face {_fmt(conditions.face_deg)}   path {_fmt(conditions.path_deg)}",
        f"ball {flight.ball_speed_mph:.0f} mph   launch {flight.launch_deg:.1f} deg"
        f"   spin {flight.spin_rpm:.0f} rpm",
        carry,
    ]
