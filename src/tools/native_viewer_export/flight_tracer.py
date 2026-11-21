"""Ball flight and shot tracer of native exports (#15150).

From the impact frame on the ball leaves the tee along the carry trajectory of
the repository impact and ball-flight model
(``impact_parameters.flight_report``), launched from the clubhead state the
impact card reads. The flight is a model prediction from kinematic impact
conditions, not a measurement. This module holds the flight in swing time and
world coordinates and the validated overlay options, read from
``flight_tracer_settings.json`` (see its ``sources``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.impact_parameters.flight_report import (
    FlightSummary,
    flight_for_impact,
)
from src.tools.native_viewer_export import impact_card

if TYPE_CHECKING:
    from src.shared.python.impact_parameters.target_frame import TargetFrame
    from src.tools.native_viewer_export.ball import BallPlacement

Array = NDArray[np.float64]
SETTINGS_PATH = Path(__file__).with_name("flight_tracer_settings.json")
ON_TEE, IN_FLIGHT, LANDED = "on tee", "in flight", "landed"
NO_IMPACT, MODEL = "no impact detected", "model prediction"
OUTSIDE = "outside the model validity envelope"


def _channels(value: Any, count: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise TypeError(f"{name} must be {count} integers")
    if not all(isinstance(c, int) and 0 <= c <= 255 for c in value):
        raise ValueError(f"{name} channels must be integers in 0..255")
    return tuple(value)


@dataclass(frozen=True)
class TracerSettings:
    """Validated options of the ball flight overlay (meanings: the JSON file)."""

    tracer_color_rgba: tuple[int, ...]
    tracer_width_px: int
    fade_after_s: float | None
    ball_color_rgb: tuple[int, ...]
    min_ball_radius_px: int
    fov_y_deg: float
    near_clip_m: float
    landing_hold_s: float
    max_hold_s: float
    draw_invalid_flight: bool

    def __post_init__(self) -> None:
        rgba = _channels(self.tracer_color_rgba, 4, "tracer_color_rgba")
        object.__setattr__(self, "tracer_color_rgba", rgba)
        rgb = _channels(self.ball_color_rgb, 3, "ball_color_rgb")
        object.__setattr__(self, "ball_color_rgb", rgb)
        for name in ("tracer_width_px", "min_ball_radius_px"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.fov_y_deg < 180.0:
            raise ValueError("fov_y_deg must lie inside (0, 180)")
        if not (math.isfinite(self.near_clip_m) and self.near_clip_m > 0.0):
            raise ValueError("near_clip_m must be positive")
        for name in ("landing_hold_s", "max_hold_s", "fade_after_s"):
            value = getattr(self, name)
            if name == "fade_after_s" and value is None:
                continue
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and not negative")
        if not isinstance(self.draw_invalid_flight, bool):
            raise TypeError("draw_invalid_flight must be true or false")


def load_tracer_settings(path: Path | None = None) -> TracerSettings:
    """Settings from ``path`` (default: the file beside this module).

    Raises ``ValueError``/``TypeError`` for a missing, unknown or bad entry.
    """
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    try:
        return TracerSettings(**document)
    except TypeError as exc:
        if "argument" not in str(exc):
            raise
        raise ValueError(f"flight tracer settings: {exc}") from exc


@dataclass(frozen=True)
class FlightPlayback:
    """The ball flight of one swing in swing time and world coordinates.

    Without a drawable flight ``path_m`` is ``None`` and ``status`` says why.
    ``path_m[0]`` is the ball on the tee at ``impact_time_s``; ``times_s`` are
    seconds after impact. Postcondition: the ball is on the path at every time
    from impact on and rests on its last point after landing.
    """

    settings: TracerSettings
    status: str
    impact_time_s: float = 0.0
    times_s: Array | None = None
    path_m: Array | None = None
    radius_m: float = 0.0
    lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.settings, TracerSettings):
            raise TypeError("settings must be a TracerSettings")
        if self.path_m is None:
            return
        times, path = np.asarray(self.times_s), np.asarray(self.path_m)
        if path.ndim != 2 or path.shape[1] != 3 or times.shape != path.shape[:1]:
            raise ValueError("path_m must be (n, 3) with one time per point")
        if len(times) < 2 or not (np.isfinite(path).all() and np.isfinite(times).all()):
            raise ValueError("a flight needs two or more finite points")
        if times[0] != 0.0 or (np.diff(times) <= 0.0).any():
            raise ValueError("times_s must start at 0 and increase")
        if not (math.isfinite(self.impact_time_s) and self.radius_m > 0.0):
            raise ValueError("impact_time_s must be finite and radius_m positive")

    @property
    def duration_s(self) -> float:
        """Flight time from impact to landing (0 without a flight)."""
        return 0.0 if self.times_s is None else float(self.times_s[-1])

    def flown(self, time_s: float) -> Array | None:
        """World points of the tracer at swing time ``time_s`` (last = the ball).

        ``None`` before impact or without a flight. Raises ``ValueError`` for a
        time that is not finite.
        """
        if not math.isfinite(time_s):
            raise ValueError(f"time_s must be finite, got {time_s}")
        if self.path_m is None or self.times_s is None:
            return None
        since = time_s - self.impact_time_s
        if since < 0.0:
            return None
        tau = min(since, self.duration_s)
        count = int(np.searchsorted(self.times_s, tau, side="left"))
        ball = [np.interp(tau, self.times_s, self.path_m[:, i]) for i in range(3)]
        points = np.vstack([self.path_m[:count], ball])
        fade = self.settings.fade_after_s
        if fade is not None:
            keep = np.append(self.times_s[:count], tau) >= since - fade
            keep[-1] = True
            points = points[keep]
        return points

    def hud_lines(self, time_s: float) -> tuple[str, ...]:
        """HUD text at ``time_s``: the state of the ball, then the flight numbers."""
        if self.path_m is None:
            return (f"Ball flight: {self.status}",)
        since = time_s - self.impact_time_s
        if since < 0.0:
            return (f"Ball flight: {ON_TEE}",)
        phase = LANDED if since >= self.duration_s else IN_FLIGHT
        return (f"Ball flight ({self.status}): {phase}", *self.lines)


def playback_from_flight(
    flight: FlightSummary,
    frame: TargetFrame,
    impact_time_s: float,
    ball: BallPlacement,
    settings: TracerSettings,
) -> FlightPlayback:
    """Playback of ``flight`` launched from ``ball`` along the target ``frame``.

    A flight the model marks invalid is returned without a path (its reason as
    the status) unless ``settings.draw_invalid_flight``.
    """
    if not flight.valid and not settings.draw_invalid_flight:
        reason = flight.invalid_reason.split(";")[0]
        return FlightPlayback(settings, f"not shown, {reason}")
    local = np.asarray(flight.trajectory_m, dtype=float)
    axes = np.vstack([frame.x_t, frame.y_t, frame.z_t])
    path = np.asarray(ball.position_m, dtype=float) + (local - local[0]) @ axes
    times = flight.times_s
    if times is None:
        times = np.linspace(0.0, flight.flight_time_s, len(path))
    side = "left" if flight.lateral_m < 0.0 else "right"
    lines = (
        f"Ball {flight.ball_speed_mph:.0f} mph   Launch {flight.launch_deg:.1f} deg"
        f"   Spin {flight.spin_rpm:.0f} rpm",
        f"Carry {flight.carry_m:.0f} m   Apex {flight.max_height_m:.0f} m   Side "
        f"{abs(flight.lateral_m):.0f} m {side}   Time {flight.flight_time_s:.1f} s",
    )
    status = MODEL if flight.valid else f"{MODEL}, {OUTSIDE}"
    start = np.asarray(times, dtype=float) - float(times[0])
    return FlightPlayback(
        settings, status, impact_time_s, start, path, ball.radius_m, lines
    )


def playback_from_series(
    series: ClubheadSeries,
    impact_time_s: float,
    ball: BallPlacement,
    settings: TracerSettings,
) -> FlightPlayback:
    """Flight launched from the clubhead state the impact card reads.

    That is the last sample before ``impact_time_s``. Raises what the impact
    and flight models raise for a series they cannot use.
    """
    frame = impact_card.address_target_frame(series)
    state = impact_card.contact_index(series, impact_time_s) - 1
    flight = flight_for_impact(series, state, frame)
    return playback_from_flight(flight, frame, impact_time_s, ball, settings)
