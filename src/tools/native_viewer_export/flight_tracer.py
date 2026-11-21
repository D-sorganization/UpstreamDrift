"""Ball flight and shot tracer of native exports (#15150).

From the impact frame on the ball leaves the tee along the carry trajectory of
the repository impact and ball-flight model
(``impact_parameters.flight_report``), launched from the clubhead state the
impact card reads. The flight is a model prediction from kinematic impact
conditions, not a measurement. Ball and tracer are drawn on every fixed view
tile through a pinhole camera matching the view preset: a 2D overlay that the
scene depth (``flight_depth``) lets the golfer hide, with a ``landing_marker`` on
the ground. The clip is held on its last pose until the ball has landed. Options are read from ``flight_tracer_settings.json`` (see ``sources``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from src.shared.python.golf_view_presets import get_view_preset
from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.impact_parameters.flight_report import (
    FlightSummary,
    flight_for_impact,
)
from src.tools.native_viewer_export import flight_depth, impact_card, landing_marker
from src.tools.native_viewer_export.landing_marker import LandingMarkerSettings
from src.tools.native_viewer_export.impact_state import series_at_contact
from src.tools.native_viewer_export.overlay2d import pinhole_for_view

if TYPE_CHECKING:
    from src.shared.python.impact_parameters.target_frame import TargetFrame
    from src.tools.native_viewer_export.ball import BallPlacement
    from src.tools.native_viewer_export.core import ExportSettings, SwingInput

Array = NDArray[np.float64]
Image8 = NDArray[np.uint8]
logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).with_name("flight_tracer_settings.json")
ON_TEE, IN_FLIGHT, LANDED = "on tee", "in flight", "landed"
NO_IMPACT, MODEL = "no impact detected", "model prediction"
OUTSIDE = "outside the model validity envelope"
club_series = impact_card._club_series


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
    occlude_by_scene: bool = True
    occlusion_tolerance_m: float = 0.02
    occlusion_tolerance_fraction: float = 0.005
    occlusion_step_px: float = 3.0
    landing_marker: LandingMarkerSettings | None = None

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
        for name in ("draw_invalid_flight", "occlude_by_scene"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be true or false")
        for name in ("occlusion_tolerance_m", "occlusion_tolerance_fraction"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0.0):
                raise ValueError(f"{name} must be finite and not negative")
        if not self.occlusion_step_px > 0.0:
            raise ValueError("occlusion_step_px must be positive")


def load_tracer_settings(path: Path | None = None) -> TracerSettings:
    """Settings from ``path`` (default: the file beside this module).

    Raises ``ValueError``/``TypeError`` for a missing, unknown or bad entry.
    """
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    marker = document.pop("landing_marker", None)
    if marker is not None:
        marker = {k: tuple(v) if isinstance(v, list) else v for k, v in marker.items()}
        document["landing_marker"] = LandingMarkerSettings(**marker)
    try:
        return TracerSettings(**document)
    except TypeError as exc:
        if "argument" not in str(exc):
            raise
        raise ValueError(f"flight tracer settings: {exc}") from exc


project = flight_depth.project


def occlusion_for(style: TracerSettings, depth_m: Array | None) -> Any:
    """The depth test for one tile, or ``None`` without scene depth or when off."""
    if depth_m is None or not style.occlude_by_scene:
        return None
    tol, frac = style.occlusion_tolerance_m, style.occlusion_tolerance_fraction
    step = style.occlusion_step_px
    return flight_depth.Occlusion(
        np.asarray(depth_m), tol, frac, step, style.near_clip_m
    )


def draw_trace(
    tile: Image8,
    camera: Any,
    points_m: Array,
    radius_m: float,
    style: TracerSettings,
    depth_m: Array | None = None,
) -> Image8:
    """``tile`` with the tracer through ``points_m`` and the ball at the last one.

    Points behind the camera, and (with ``depth_m``, metres per pixel, and
    ``occlude_by_scene``) points behind the scene surface, are left out.
    """
    occlusion = occlusion_for(style, depth_m)
    image = Image.fromarray(tile)
    pen = ImageDraw.Draw(image, "RGBA")
    colour, near = style.tracer_color_rgba, style.near_clip_m
    width = style.tracer_width_px
    flight_depth.draw_line(pen, camera, points_m, colour, width, near, occlusion)
    pixels, depth = project(camera, points_m[-1:], near)
    shown = bool(np.isfinite(pixels).all())
    if shown and occlusion is not None:
        shown = bool(occlusion.visible(camera, points_m[-1:])[1][0])
    if shown:
        projected = float(camera.matrix[1, 1]) * radius_m / float(depth[-1])
        r = max(float(style.min_ball_radius_px), projected)
        u, v = (float(c) for c in pixels[-1])
        pen.ellipse([u - r, v - r, u + r, v + r], fill=(*style.ball_color_rgb, 255))
    return np.asarray(image, dtype=np.uint8)


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

    def held(self, shown: SwingInput, feed: Any, step_s: float) -> tuple[Any, Any]:
        """``shown`` with its last pose repeated until the ball has landed.

        Frames are added every ``step_s`` of swing time up to landing plus
        ``landing_hold_s``, at most ``max_hold_s``; ``feed`` then repeats its
        last frame. Unchanged without a flight. Raises ``ValueError`` for a
        step that is not positive.
        """
        if not (math.isfinite(step_s) and step_s > 0.0):
            raise ValueError(f"step_s must be positive, got {step_s}")
        times = shown.sample_times_s
        if self.path_m is None or times is None:
            return shown, feed
        style = self.settings
        landed = self.impact_time_s + self.duration_s + style.landing_hold_s
        end = min(landed, times[-1] + style.max_hold_s)
        extra = math.ceil((end - times[-1]) / step_s - 1e-9)
        if extra <= 0:
            return shown, feed
        last = shown.n_states - 1
        more = tuple(times[-1] + step_s * (i + 1) for i in range(extra))
        q = np.concatenate([shown.q, np.repeat(shown.q[-1:], extra, axis=0)])
        v = shown.v
        if v is not None:  # the held pose is at rest
            v = np.concatenate([v, np.zeros((extra, v.shape[1]))])
        held = replace(shown, q=q, v=v, sample_times_s=(*times, *more))
        if feed is not None:
            source = feed.frame_at
            feed = replace(feed, frame_at=lambda i: source(min(int(i), last)))
        return held, feed

    def draw(
        self,
        tiles: Mapping[str, Image8],
        time_s: float,
        lookat_m: Sequence[float],
        distance_m: float | None,
        depth_m: Mapping[str, Array] | None = None,
    ) -> dict[str, Image8]:
        """``tiles`` (per view) with the ball and tracer at ``time_s``.

        A view that follows a moving target is left as it is. ``depth_m`` is the
        scene depth per view; a view without it is drawn over the scene.
        """
        points = self.flown(time_s)
        out = dict(tiles)
        if points is None:
            return out
        landed = time_s - self.impact_time_s >= self.duration_s
        fov = math.radians(self.settings.fov_y_deg)
        for view, tile in tiles.items():
            preset = get_view_preset(view)
            if preset.tracks is not None:
                continue
            size = (int(tile.shape[1]), int(tile.shape[0]))
            camera = pinhole_for_view(preset, lookat_m, distance_m, fov, size)
            depth = (depth_m or {}).get(view)
            if depth is None and self.settings.occlude_by_scene:
                self._warn_once(view)
            out[view] = self._paint(tile, camera, points, depth, landed)
        return out

    def _warn_once(self, view: str) -> None:
        if not self.__dict__.setdefault("_warned", False):
            self.__dict__["_warned"] = True
            logger.warning("scene depth unavailable (%s): tracer over the scene", view)

    def _paint(
        self, tile: Image8, camera: Any, points: Array, depth: Any, landed: bool
    ) -> Image8:
        """One tile: landing marker beneath, then the tracer and ball."""
        style, marker = self.settings, self.settings.landing_marker
        if marker is not None and self.path_m is not None:
            ring = landed or marker.when == "from_start"
            tile = landing_marker.paint(tile, camera, self.path_m[-1], style, ring)
        return draw_trace(tile, camera, points, self.radius_m, style, depth)


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

    That is the state where the face reaches the ball
    (:func:`impact_state.series_at_contact`) and the flight starts at that
    instant, so the tracer leaves the ball as the face meets it. Without a
    contact state (the face does not reach the ball) it is the last sample
    before ``impact_time_s``, started at ``impact_time_s``. Raises what the
    impact and flight models raise for a series they cannot use.
    """
    frame = impact_card.address_target_frame(series)
    contact = series_at_contact(series, ball.position_m, ball.radius_m, impact_time_s)
    if contact is None:
        state = impact_card.contact_index(series, impact_time_s) - 1
        flight = flight_for_impact(series, state, frame)
        return playback_from_flight(flight, frame, impact_time_s, ball, settings)
    moved, state = contact
    flight = flight_for_impact(moved, state, frame)
    start_s = float(moved.times_s[state])
    return playback_from_flight(flight, frame, start_s, ball, settings)


def flight_for_export(
    swing: SwingInput, settings: ExportSettings
) -> FlightPlayback | None:
    """The export's ball flight, or ``None`` when the option is off.

    Without a known impact time or a placed ball, or when the models fail, the
    playback has no path and its status states the reason.
    """
    config: Any = settings.ball_flight
    if config is None:
        return None
    impact = settings.impact_time_s
    if impact is None:
        if not swing.has_impact_provenance:
            return FlightPlayback(config, NO_IMPACT)
        impact = swing.impact_time_s
    if swing.ball is None:
        return FlightPlayback(config, "no ball placed")
    try:
        return playback_from_series(club_series(swing), impact, swing.ball, config)
    except impact_card._EXPECTED_ERRORS as exc:
        return FlightPlayback(config, f"unavailable: {exc}")
