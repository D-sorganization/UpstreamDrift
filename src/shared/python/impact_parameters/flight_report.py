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
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.physics.ball_flight_validity import (
    MAX_DRIVER_LAUNCH_SPIN_RPM,
    assess_trajectory_validity,
)
from src.shared.python.physics.swing_ball_flight_pipeline import (
    SwingBallFlightPipeline,
    SwingState,
)

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


def _invalid_reason(path: np.ndarray) -> str:
    """Delegate to :func:`~physics.ball_flight_validity.assess_trajectory_validity`."""
    return assess_trajectory_validity(path).invalid_reason


def swing_state_from_series(
    series: ClubheadSeries, index: int, frame: TargetFrame
) -> SwingState:
    """Clubhead state at sample ``index`` in the pipeline frame.

    Raises:
        ValueError: ``index`` out of range or the face normal is unobservable.
    """
    require(0 <= index < len(series), "index out of range", index)
    face_normals = series.face_normal
    if face_normals is None:
        raise ValueError(f"face normal unavailable: {series.face_unobservable_reason}")
    velocity = np.asarray(frame.components(series.velocity_mps[index]), dtype=float)
    normal = np.asarray(frame.components(face_normals[index]), dtype=float)
    normal /= np.linalg.norm(normal)
    return SwingState(
        clubhead_velocity=velocity,
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal,
        clubhead_loft_deg=math.degrees(math.asin(float(normal[2]))),
        engine_name="mujoco",
    )


def flight_for_impact(
    series: ClubheadSeries, index: int, frame: TargetFrame
) -> FlightSummary:
    """Carry flight of the impact state at ``index`` (impact solver + aerodynamics)."""
    result = SwingBallFlightPipeline().run(
        swing_state_from_series(series, index, frame)
    )
    path = np.array([p.position for p in result.trajectory], dtype=float)
    launch = result.launch_conditions
    meta = result.metadata
    if not bool(meta.get("flight_valid", True)):
        reason = str(meta.get("flight_invalid_reason", "") or "")
    else:
        reason = _invalid_reason(path)
        if not reason and meta.get("launch_spin_clamped"):
            pre_clamp_rpm = float(
                meta.get("pre_clamp_launch_spin_rpm", launch.spin_rate)
            )
            if pre_clamp_rpm > MAX_DRIVER_LAUNCH_SPIN_RPM + 1.0:
                reason = (
                    "impact spin exceeds the driver validity envelope "
                    f"({pre_clamp_rpm:.0f} rpm); trajectory after capping is not presented (SHOW-C7 / #14621)"
                )
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
