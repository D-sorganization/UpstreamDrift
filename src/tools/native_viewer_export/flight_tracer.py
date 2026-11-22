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
from src.shared.python.impact_parameters.club_impact import (
    ClubImpact,
    club_impact_from_spec,
)
from src.shared.python.impact_parameters.extract import MPS_TO_MPH, face_location_mm
from src.shared.python.impact_parameters.flight_report import (
    FlightSummary,
    flight_for_impact,
)
from src.tools.native_viewer_export import (
    aim_line,
    contact_verdict,
    flight_depth,
    impact_card,
    landing_marker,
)
from src.tools.native_viewer_export.aim_line import AimLineSettings, load_aim_settings
from src.tools.native_viewer_export.display_units import (
    METRIC,
    DisplayUnits,
    distance_text,
    load_display_units,
    speed_text,
)
from src.tools.native_viewer_export.flight_inset import (
    FlightInsetSettings,
    load_inset_settings,
    shows_inset,
)
from src.tools.native_viewer_export.landing_marker import LandingMarkerSettings
from src.tools.native_viewer_export.impact_state import series_at_contact
from src.tools.native_viewer_export.behind_camera import look_map, lookat_of
from src.tools.native_viewer_export.overlay2d import pinhole_for_view
from src.tools.native_viewer_export.rollout import (
    Rollout,
    RolloutSettings,
    compute_rollout,
    load_rollout_settings,
)

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
#: status (HUD line and manifest) of a face that missed the ball
NO_CONTACT_STATUS = "no contact: no flight"
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
    inset: FlightInsetSettings | None = None  # flight_inset_settings.json
    ball_shading: flight_depth.BallShading | None = None
    display_units: DisplayUnits = METRIC  # load_tracer_settings: display_units.py
    rollout: RolloutSettings | None = None  # rollout_settings.json
    hole_view: Any = None  # hole_view.HoleViewSettings (--hole-view); None = off
    aim: AimLineSettings | None = None  # aim_line_settings.json (#15443)

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


def load_tracer_settings(
    path: Path | None = None, units: DisplayUnits | None = None
) -> TracerSettings:
    """Settings from ``path`` (default: the file beside this module).

    Distance labels (HUD, inset, carry label) use ``units`` (default: the
    ``display_units_settings.json`` beside this module): its unit replaces the
    ``unit`` entries of the inset and the carry label.

    Raises ``ValueError``/``TypeError`` for a missing, unknown or bad entry.
    """
    units = units or load_display_units()
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    document["display_units"] = units
    marker = document.pop("landing_marker", None)
    if marker is not None:
        marker = {k: tuple(v) if isinstance(v, list) else v for k, v in marker.items()}
        marker["carry_label_unit"] = units.distance_unit
        marker["carry_label_scale"] = units.distance_scale
        document["landing_marker"] = LandingMarkerSettings(**marker)
    shading = document.pop("ball_shading", None)
    if shading is not None:
        document["ball_shading"] = flight_depth.BallShading(**shading)
    beside = (path or SETTINGS_PATH).with_name("flight_inset_settings.json")
    inset = load_inset_settings(beside if beside.is_file() else None)
    document["inset"] = replace(
        inset, unit=units.distance_unit, unit_scale=units.distance_scale
    )
    document["rollout"] = load_rollout_settings()
    document["aim"] = load_aim_settings()
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
    if np.isfinite(pixels).all():
        projected = float(camera.matrix[1, 1]) * radius_m / float(depth[-1])
        r = max(float(style.min_ball_radius_px), projected)
        centre = (float(pixels[-1, 0]), float(pixels[-1, 1]))
        sphere = (float(depth[-1]), radius_m)
        flight_depth.draw_ball(
            image,
            centre,
            r,
            style.ball_color_rgb,
            sphere,
            occlusion,
            style.ball_shading,
        )
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
    carry_m: float | None = None
    #: target-frame axes (target, left, up) rows, the model's apex height and
    #: right-positive side offset: the numbers the flight inset shows
    axes_m: Array | None = None
    apex_m: float | None = None
    lateral_m: float | None = None
    #: launch angle (degrees) and spin (rpm) of the model flight, for summaries
    launch_deg: float | None = None
    spin_rpm: float | None = None
    ball_speed_mph: float | None = None  # of the impact solve; the smash factor's
    rollout: Rollout | None = None  # roll after the carry (a model assumption)
    #: aim line of the plan view and wide view (aim_line.py): its turn from the
    #: address line, label and the landing's side against it; ``None`` = address
    aim_yaw_rad: float | None = None
    aim_text: str | None = None
    aim_lateral_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.settings, TracerSettings):
            raise TypeError("settings must be a TracerSettings")
        if self.carry_m is not None and not (
            math.isfinite(self.carry_m) and self.carry_m >= 0.0
        ):
            raise ValueError("carry_m must be finite and not negative")
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

    def display_unit_names(self) -> dict[str, str]:
        """Unit names of the labels (for the manifest, whose values stay SI)."""
        units = self.settings.display_units
        return {"distance": units.distance_unit, "speed": units.speed_unit}

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
        last frame, ellipsoids and HUD lines included. Unchanged without a flight. Raises ``ValueError`` for a
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
        return held, None if feed is None else feed.held_after(last)

    def draw(
        self,
        tiles: Mapping[str, Image8],
        time_s: float,
        lookat_m: Sequence[float] | Mapping[str, Sequence[float]],
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
            look = lookat_of(lookat_m, view)
            camera = pinhole_for_view(preset, look, distance_m, fov, size)
            depth = (depth_m or {}).get(view)
            if depth is None and self.settings.occlude_by_scene:
                self._warn_once(view)
            label = not shows_inset(self.settings.inset, view)
            out[view] = self._paint(tile, camera, points, depth, landed, label)
        return out

    def grid_tiles(
        self,
        tiles: Mapping[str, Image8],
        time_s: float,
        lookat_m: Sequence[float] | Mapping[str, Sequence[float]],
        distance_m: float | None,
    ) -> dict[str, Image8]:
        """``tiles`` for the composed grid: the carry label an inset replaces.

        A single view with the inset states the carry there; the grid draws no
        inset, so its tile gets the ring label back (once landed).
        """
        out = dict(tiles)
        landed = time_s - self.impact_time_s >= self.duration_s
        if self.path_m is None or not landed:
            return out
        fov = math.radians(self.settings.fov_y_deg)
        for view, tile in tiles.items():
            if not shows_inset(self.settings.inset, view):
                continue
            size = (int(tile.shape[1]), int(tile.shape[0]))
            camera = pinhole_for_view(
                get_view_preset(view), lookat_of(lookat_m, view), distance_m, fov, size
            )
            out[view] = self._label(tile, camera, grid=True)
        return out

    def _label(self, tile: Image8, camera: Any, grid: bool = False) -> Image8:
        """``tile`` with the carry label beside the ring (where configured)."""
        style = self.settings
        if style.landing_marker is None or self.path_m is None:
            return tile
        if self.carry_m is None:
            return tile
        end = self.path_m[-1]
        return landing_marker.paint_label(
            tile, camera, end, style, self.carry_m, grid=grid
        )

    def _paint_aim(self, tile: Image8, camera: Any, depth: Any) -> Image8:
        """``tile`` with the aim line on the ground (nothing without one)."""
        points, style = _aim_line_points(self), self.settings
        aim = style.aim
        if points is None or aim is None:
            return tile
        image = Image.fromarray(tile)
        pen, hide = ImageDraw.Draw(image, "RGBA"), occlusion_for(style, depth)
        width = aim.width_for(tile.shape[0])
        for colour, size in (
            (aim.outline_color_rgba, width + aim.outline_extra_px),
            (aim.line_color_rgba, width),
        ):
            flight_depth.draw_line(
                pen, camera, points, colour, size, style.near_clip_m, hide
            )
        return np.asarray(image, dtype=np.uint8)

    def _warn_once(self, view: str) -> None:
        if not self.__dict__.setdefault("_warned", False):
            self.__dict__["_warned"] = True
            logger.warning("scene depth unavailable (%s): tracer over the scene", view)

    def _paint(
        self,
        tile: Image8,
        camera: Any,
        points: Array,
        depth: Any,
        landed: bool,
        label: bool = True,
    ) -> Image8:
        """One tile: landing marker beneath, then the tracer and ball.

        ``label`` False leaves the carry label out (an inset states it).
        """
        style, marker = self.settings, self.settings.landing_marker
        tile = self._paint_aim(tile, camera, depth)
        if marker is not None and self.path_m is not None:
            ring = landed or marker.when == "from_start"
            hide = occlusion_for(style, depth)
            tile = landing_marker.paint_track(tile, camera, points, style, hide, landed)
            tile = landing_marker.paint(tile, camera, self.path_m[-1], style, ring)
            if landed and label:
                tile = self._label(tile, camera)
        return draw_trace(tile, camera, points, self.radius_m, style, depth)


def _aim_line_points(play: FlightPlayback) -> Array | None:
    """Ground end points of the drawn aim line, ``None`` when none is drawn."""
    style = play.settings
    if play.aim_yaw_rad is None or style.aim is None or play.path_m is None:
        return None
    axes = np.asarray(play.axes_m)
    full = (play.path_m - play.path_m[0]) @ axes.T
    ends = aim_line.rotate_plan(full[-1], play.aim_yaw_rad)
    reach = max(float(ends[0]), 0.0)
    marker = style.landing_marker
    ground_z = marker.ground_z_m if marker is not None else 0.0
    length = reach * (1.0 + style.aim.overshoot_fraction)
    return aim_line.ground_line(
        play.path_m[0], axes, play.aim_yaw_rad, length, ground_z
    )


def _flight_lines(
    flight: FlightSummary, side: str, units: DisplayUnits
) -> tuple[str, str]:
    """The two HUD lines of ``flight``; distances and speed in ``units``."""
    speed = speed_text(flight.ball_speed_mph, units)
    carry = distance_text(flight.carry_m, units)
    apex = distance_text(flight.max_height_m, units)
    offset = distance_text(abs(flight.lateral_m), units)
    offset = f"{offset} {side}" if side else offset
    return (
        f"Ball {speed}   Launch {flight.launch_deg:.1f} deg"
        f"   Spin {flight.spin_rpm:.0f} rpm",
        f"Carry {carry}   Apex {apex}   Side {offset}"
        f"   Time {flight.flight_time_s:.1f} s",
    )


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
    lines = _flight_lines(flight, side, settings.display_units)
    status = MODEL if flight.valid else f"{MODEL}, {OUTSIDE}"
    start = np.asarray(times, dtype=float) - float(times[0])
    return FlightPlayback(
        settings,
        status,
        impact_time_s,
        start,
        path,
        ball.radius_m,
        lines,
        flight.carry_m,
        axes,
        flight.max_height_m,
        flight.lateral_m,
        flight.launch_deg,
        flight.spin_rpm,
        flight.ball_speed_mph,
    )


def with_rollout(
    playback: FlightPlayback, flight: FlightSummary, club: str | None
) -> FlightPlayback:
    """``playback`` with the roll and total of ``flight`` struck with ``club``.

    Adds a HUD line stating them and the model assumption; unchanged when the
    roll is off, the flight is invalid or ``club`` has no runout class.
    """
    settings = playback.settings
    roll = compute_rollout(flight, club, settings.rollout)
    if roll is None or playback.path_m is None or settings.rollout is None:
        return playback
    line = roll.hud_text(settings.display_units, settings.rollout)
    return replace(playback, rollout=roll, lines=(*playback.lines, line))


def strike_offset_m(
    series: ClubheadSeries, index: int, ball: BallPlacement
) -> tuple[float, float] | None:
    """Strike (toe, high) from the face centre [m], ``None`` if unobservable/off-face."""
    try:
        point = impact_card.ball_contact_point(series, index, ball)
        loc = face_location_mm(series, index, point)
    except ValueError:
        return None
    if isinstance(loc, str):
        return None
    if math.hypot(*loc) > impact_card.MAX_FACE_OFFSET_M * 1000.0:
        return None
    return loc[0] * 1e-3, loc[1] * 1e-3


def with_model_smash(overlays: Any) -> Any:
    """``overlays`` whose impact card and factors panel show the model smash factor.

    The ball speed is that of the flight (the impact solve that drives the
    tracer); without a flight path or a ball speed ``overlays`` is returned
    unchanged and the rows keep their "unavailable" reason.
    """
    flight = None if overlays is None else overlays.flight
    speed = None if flight is None else flight.ball_speed_mph
    if speed is None:
        return overlays
    card, factors = overlays.card, overlays.factors
    if card is not None:
        card = impact_card.with_model_smash(card, speed)
    if factors is not None:
        factors = factors.with_model_smash(speed)
    return replace(overlays, card=card, factors=factors)


def with_aim(playback: FlightPlayback, flight: FlightSummary) -> FlightPlayback:
    """``playback`` read against the aim line its settings ask for (#15443).

    Unchanged for ``auto`` (no shot shape asked for) and for a flight without a
    path. The HUD gets the side against the aim line and a line naming it.
    """
    aim = playback.settings.aim
    if aim is None or aim.aim_line == "auto" or playback.path_m is None:
        return playback
    fields = aim_line.aimed_fields(playback, aim.aim_line, aim)
    side_m, side = aim_line.side_text_args(fields["aim_lateral_m"])
    units = playback.settings.display_units
    own = _flight_lines(replace(flight, lateral_m=side_m), side, units)
    lines = (*own, *playback.lines[2:], fields["aim_text"])
    return replace(playback, lines=lines, **fields)


def attach_flight(overlays: Any, swing: Any, settings: Any) -> Any:
    """``overlays`` with the export's flight and the model smash factor.

    One flight evaluation (:func:`flight_for_export`) feeds the flight HUD, the
    impact card and the factors panel. Without ``settings.ball_flight``
    ``overlays`` is returned unchanged. ``overlays`` may be ``None``.
    """
    if settings.ball_flight is None:
        return overlays
    from src.tools.native_viewer_export.core import ClipOverlays

    panel = None if overlays is None else overlays.factors
    verdict = None if panel is None else panel.verdict
    flight = flight_for_export(swing, settings, verdict)
    overlays = replace(overlays or ClipOverlays(), flight=flight)
    return with_model_smash(overlays)


def manifest_record(overlays: Any) -> dict[str, Any] | None:
    """Manifest entry of the flight: SI values and the display unit names.

    ``None`` without a flight; a flight without a path gives its status under
    ``"unavailable"``. Speeds are m/s, distances m, whatever the labels show.
    """
    flight = None if overlays is None else overlays.flight
    if flight is None:
        return None
    record: dict[str, Any] = {"units": flight.display_unit_names()}
    if flight.path_m is None:
        return {**record, "unavailable": flight.status}
    speed = flight.ball_speed_mph
    record.update(
        carry_m=flight.carry_m,
        apex_m=flight.apex_m,
        lateral_m=flight.lateral_m,
        ball_speed_mps=None if speed is None else speed / MPS_TO_MPH,
    )
    if flight.rollout is not None:
        record["rollout"] = flight.rollout.record()
    return record


def playback_from_series(
    series: ClubheadSeries,
    impact_time_s: float,
    ball: BallPlacement,
    settings: TracerSettings,
    club: str | None = None,
    club_impact: ClubImpact | None = None,
    verdict: contact_verdict.ContactVerdict | None = None,
) -> FlightPlayback:
    """Flight launched from the clubhead state the impact card reads.

    A ``verdict`` of no contact (given, e.g. the factors panel's, else decided
    here by the same rule) gives a playback without a path whose status is
    :data:`NO_CONTACT_STATUS`: a face that missed the ball launches nothing.

    That is the state where the face reaches the ball
    (:func:`impact_state.series_at_contact`) and the flight starts at that
    instant, so the tracer leaves the ball as the face meets it. Without a
    contact state (the face does not reach the ball) it is the last sample
    before ``impact_time_s``, started at ``impact_time_s``. ``club`` gives the
    head mass, inertia and COR of the club struck with (``None``: the pipeline
    defaults) and the strike offset on the face goes to the impact model. Raises what the
    impact and flight models raise for a series they cannot use.
    """
    if verdict is None:
        verdict = contact_verdict.verdict_for(series, impact_time_s, ball)
    if not verdict.contact:
        return FlightPlayback(settings, NO_CONTACT_STATUS)
    frame = impact_card.address_target_frame(series)
    contact = series_at_contact(series, ball.position_m, ball.radius_m, impact_time_s)
    if contact is None:
        state = impact_card.contact_index(series, impact_time_s) - 1
        flight = flight_for_impact(series, state, frame, club, club_impact)
        play = playback_from_flight(flight, frame, impact_time_s, ball, settings)
        return with_aim(with_rollout(play, flight, club), flight)
    moved, state = contact
    offset = strike_offset_m(moved, state, ball)
    flight = flight_for_impact(moved, state, frame, club, club_impact, offset)
    start_s = float(moved.times_s[state])
    play = playback_from_flight(flight, frame, start_s, ball, settings)
    return with_aim(with_rollout(play, flight, club), flight)


def _spec_club(swing: SwingInput) -> Mapping[str, Any] | None:
    """The ``club`` entry of the bundle specification, ``None`` if unreadable."""
    try:
        entry = json.loads(swing.bundle.spec_bytes).get("club")
    except (ValueError, AttributeError):
        return None
    return entry if isinstance(entry, Mapping) else None


def flight_for_export(
    swing: SwingInput,
    settings: ExportSettings,
    verdict: contact_verdict.ContactVerdict | None = None,
) -> FlightPlayback | None:
    """The export's ball flight, or ``None`` when the option is off.

    ``verdict`` is the contact decision already made (the factors panel's);
    without it the decision is made here by the same rule.

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
    if verdict is not None and not verdict.contact:
        return FlightPlayback(config, NO_CONTACT_STATUS)
    try:
        club_impact = club_impact_from_spec(_spec_club(swing))
        return playback_from_series(
            club_series(swing),
            impact,
            swing.ball,
            config,
            swing.club,
            club_impact,
            verdict,
        )
    except impact_card._EXPECTED_ERRORS as exc:
        return FlightPlayback(config, f"unavailable: {exc}")


def grid_cells(
    flight: FlightPlayback | None,
    tiles: Mapping[str, Image8],
    time_s: float,
    settings: Any,
) -> Mapping[str, Image8]:
    """Tiles for the composed grid: ``tiles`` itself when there is no flight.

    ``settings`` supplies the export's ``lookat_m`` and ``distance_m``.
    """
    if flight is None:
        return tiles
    looks = look_map(settings)
    return flight.grid_tiles(tiles, time_s, looks, settings.distance_m)
