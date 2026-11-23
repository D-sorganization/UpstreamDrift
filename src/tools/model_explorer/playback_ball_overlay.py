"""Ball, ball flight and aim line of the live Results Playback panel (#15150).

Qt-free. The ball, the shot shape and the flight are the native-viewer
export's own (``native_viewer_export``): ``core.prepare_shot`` places the
ball and rolls the club for the shape, ``flight_tracer.flight_for_export``
flies it and ``FlightPlayback`` paints the tracer, landing marker and finish
aim line through a pinhole camera matching the panel's MuJoCo camera. The
ball at address is a render-only sphere (``backends._ball.draw_job_ball``).
The flight is computed only when asked, on one background thread;
settings are in ``playback_ball_overlay_settings.json``.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input.bundle import InputBundle
from src.tools.model_explorer.golfer_playback import DEFAULT_LOOKAT_M
from src.tools.model_explorer.pose_results import PoseFrameSource
from src.tools.native_viewer_export import shot_setup
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.core import BackendUnavailable, SwingInput

SETTINGS_PATH = Path(__file__).with_name("playback_ball_overlay_settings.json")
SHAPES = shot_setup.SHAPES


@dataclass(frozen=True)
class BallOverlaySettings:
    """Validated panel defaults (meanings: the settings JSON ``sources``)."""

    show_ball: bool = True
    show_flight: bool = False
    show_aim_line: bool = False
    shot_shape: str = "straight"
    default_club: str = "Driver"
    poll_interval_ms: int = 100
    flight_unavailable_tooltip: str = "Ball flight is unavailable: {reason}"

    def __post_init__(self) -> None:
        for name in ("show_ball", "show_flight", "show_aim_line"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.shot_shape not in SHAPES:
            raise ValueError(f"shot_shape must be one of {SHAPES}")
        if not isinstance(self.poll_interval_ms, int) or self.poll_interval_ms < 1:
            raise ValueError("poll_interval_ms must be a positive integer")
        if "{reason}" not in self.flight_unavailable_tooltip:
            raise ValueError("flight_unavailable_tooltip must contain {reason}")


def load_overlay_settings(path: Path | None = None) -> BallOverlaySettings:
    """Settings from ``path`` (default: the JSON beside this module)."""
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    return BallOverlaySettings(**document)


def flight_unavailable_reason() -> str | None:
    """Why the flight model cannot run here, or ``None`` when it can."""
    from src.shared.python.physics.rust_kernel import is_rust_available

    if is_rust_available():
        return None
    return "the upstream_physics Rust flight wheel is missing or out of date"


def _club_name(spec_bytes: bytes, default: str) -> str:
    club = json.loads(spec_bytes).get("club")
    name = club.get("name") if isinstance(club, dict) else None
    return str(name) if name else default


def swing_for_source(pose: PoseFrameSource, club: str = "Driver") -> SwingInput:
    """The export's ``SwingInput`` for a played run (time from its first sample).

    A run that is itself a same-input bundle keeps the bundle (and its
    recorded impact); any other run gets a bundle wrapping its ``q`` at its
    mean sample interval. Raises ``ValueError`` for a run of zero duration.
    """
    if not isinstance(pose, PoseFrameSource):
        raise TypeError(f"pose must be a PoseFrameSource, got {type(pose).__name__}")
    q = np.asarray(pose.q, dtype=float)
    try:
        bundle = InputBundle.load(pose.path)
    except (OSError, KeyError, ValueError):
        bundle = None
    if bundle is None or bundle.reference_q.shape != q.shape:
        dt_s = float(pose.time_s[-1] - pose.time_s[0]) / (len(q) - 1)
        if not dt_s > 0.0:
            raise ValueError("the run has zero duration; no ball flight time base")
        rest = np.zeros_like(q)
        order = pose.coordinate_order
        bundle = InputBundle(
            pose.spec_bytes, order, dt_s, q[0], rest[0], rest[1:], q, rest, pose.engine
        )
    return SwingInput(bundle, q, pose.run_name, club, pose.engine)


@dataclass(frozen=True)
class ShotLayout:
    """Ball and flight of one shot shape, in swing time (``note``: the HUD text)."""

    shape: str
    ball: BallPlacement | None
    flight: Any = None  # flight_tracer.FlightPlayback
    roll_deg: float = 0.0
    note: str = ""


class BallOverlay:
    """Computes (in the background) and draws the ball and flight of a run."""

    def __init__(
        self,
        pose: PoseFrameSource,
        settings: BallOverlaySettings,
        flight_reason: str | None = None,
    ) -> None:
        if not isinstance(settings, BallOverlaySettings):
            raise TypeError("settings must be a BallOverlaySettings")
        self._pose = pose
        self._settings = settings
        self.flight_reason = flight_reason
        self._t0 = float(pose.time_s[0])
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="ball-overlay")
        self._futures: dict[str | None, Future[ShotLayout]] = {}
        self._base: tuple[SwingInput, float | None, str] | None = None
        self._straight: float | str | None = None

    # ---- computation (worker thread) --------------------------------------

    def request(self, shape: str, flight: bool = False) -> Future[ShotLayout]:
        """The (cached) background computation of ``shape``.

        Without ``flight`` only the ball at address is placed (cheap, no
        shot-shape solve, the same for every shape); with it the club is
        rolled for the shape and the ball is flown.
        """
        if shape not in SHAPES:
            raise ValueError(f"shot shape must be one of {SHAPES}, got {shape!r}")
        key = shape if flight else None
        if key not in self._futures:
            self._futures[key] = self._executor.submit(self._layout, shape, flight)
        return self._futures[key]

    def layout(self, shape: str, flight: bool = False) -> ShotLayout | None:
        """The finished layout of ``shape``, ``None`` while it is computed."""
        future = self.request(shape, flight)
        if not future.done():
            return None
        error = future.exception()
        if error is not None:
            return ShotLayout(shape, None, note=f"Ball unavailable: {error}")
        return future.result()

    def _swing(self) -> tuple[SwingInput, float | None, str]:
        if self._base is None:
            from src.tools.native_viewer_export.overlay import detect_impact_time_s

            club = _club_name(self._pose.spec_bytes, self._settings.default_club)
            swing = swing_for_source(self._pose, club)
            impact, why = None, ""
            if not swing.has_impact_provenance:
                try:
                    impact = detect_impact_time_s(swing)
                except (ValueError, KeyError, BackendUnavailable) as exc:
                    why = f"no impact detected: {exc}"
            self._base = (swing, impact, why)
        return self._base

    def _tracer(self) -> Any:
        from src.tools.native_viewer_export import flight_tracer

        tracer = flight_tracer.load_tracer_settings()
        aim = tracer.aim or flight_tracer.load_aim_settings()
        return replace(tracer, aim=replace(aim, aim_line="finish"))

    def _straight_roll(self, swing: SwingInput, settings: Any) -> float | str:
        """The solved straight club roll (degrees), or why it has none."""
        if self._straight is None:
            try:
                _, record = shot_setup.solve_shot_shape(
                    swing, replace(settings, shot_shape="straight")
                )
                self._straight = float((record or {})["straight_roll_deg"])
            except (ValueError, KeyError) as exc:
                self._straight = f"no shot shape: {exc}"
        return self._straight

    def _layout(self, shape: str, flown: bool) -> ShotLayout:
        from src.tools.native_viewer_export import flight_tracer
        from src.tools.native_viewer_export.core import ExportSettings, prepare_shot

        swing, impact, why = self._swing()
        tracer = None if not flown or self.flight_reason or why else self._tracer()
        settings = ExportSettings(impact_time_s=impact, ball_flight=tracer)
        straight = 0.0 if tracer is None else self._straight_roll(swing, settings)
        roll = 0.0
        if isinstance(straight, str):
            why, settings = straight, replace(settings, ball_flight=None)
        elif tracer is not None:
            limits = shot_setup.load_shot_setup_settings()
            roll = shot_setup.shape_roll(shape, straight, limits)
        settings = replace(settings, club_roll_in_hands_deg=roll)
        shot, settings, _ = prepare_shot(swing, settings)
        flight = flight_tracer.flight_for_export(shot, settings)
        note = why or (self.flight_reason if flown else None) or "Ball at address"
        if flight is not None:
            lines = flight.lines if flight.path_m is not None else ()
            note = " | ".join((f"Ball flight: {flight.status}", *lines))
            note = f"{shape.title()}: club roll {roll:+.1f} deg in hands | {note}"
        return ShotLayout(shape, shot.ball, flight, float(roll), note)

    # ---- drawing (caller's thread) ----------------------------------------

    def decorate_scene(
        self, scene: Any, time_s: float, layout: ShotLayout | None
    ) -> None:
        """Add the ball (and its tee) at run time ``time_s`` to a render scene."""
        ball = None if layout is None else layout.ball
        if ball is None:
            return
        from src.tools.native_viewer_export.backends._ball import draw_job_ball

        visible = (ball.visible_at(time_s - self._t0),)  # the job fields it reads
        draw_job_ball(
            scene, SimpleNamespace(ball=ball.to_dict(), ball_visible=visible), 0
        )

    def paint(
        self,
        rgb: NDArray[np.uint8],
        view: str,
        time_s: float,
        layout: ShotLayout | None,
        *,
        flight: bool,
        aim: bool,
        depth_m: NDArray[np.float64] | None = None,
    ) -> NDArray[np.uint8]:
        """``rgb`` with the aim line (whole run) and the flight (after impact)."""
        play = None if layout is None else layout.flight
        if play is None or play.path_m is None:
            return rgb
        if aim:
            from src.tools.native_viewer_export.overlay2d import pinhole_for_view

            fov = math.radians(play.settings.fov_y_deg)
            size = (int(rgb.shape[1]), int(rgb.shape[0]))
            camera = pinhole_for_view(view, DEFAULT_LOOKAT_M, None, fov, size)
            rgb = np.asarray(play.paint_aim(rgb, camera, depth_m))
        if flight:
            depth = None if depth_m is None else {view: depth_m}
            bare = replace(play, aim_yaw_rad=None)  # the aim line is drawn above
            tiles = bare.draw(
                {view: rgb}, time_s - self._t0, DEFAULT_LOOKAT_M, None, depth
            )
            rgb = np.asarray(tiles[view])
        return rgb

    def close(self) -> None:
        """Stop the background thread (pending shapes are dropped); idempotent."""
        self._executor.shutdown(wait=False, cancel_futures=True)
