"""Dispersion sample traces on the wide view of export clips (#15150 B1).

With ``--dispersion-traces`` the export replays the swing ``n_shots`` times
with the random club roll and ball position of :mod:`shot_dispersion` (around
the club roll and ball shift the clip itself plays) and draws every sample's
flight, on the nominal clock, as a thin translucent line on the configured
views. The single-shot tracer is then painted over them, so the nominal shot
stays emphasised; the HUD states the mean and spread of carry and side. The
lines use the single-shot tracer's projection, scene occlusion and line
drawing (``flight_tracer`` / ``flight_depth``). Options and every value:
``dispersion_traces_settings.json`` (see ``sources``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.shared.python.golf_view_presets import get_view_preset
from src.tools.native_viewer_export import flight_depth
from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.compositor import KeyEntry
from src.tools.native_viewer_export.display_units import DisplayUnits
from src.tools.native_viewer_export.flight_tracer import (
    FlightPlayback,
    _channels,
    occlusion_for,
)

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).with_name("dispersion_traces_settings.json")
_STATS = ("carry_mean", "carry_sd", "side_mean", "side_sd")
_PROBE = {**dict.fromkeys(_STATS, 0.0), "n": 1, "unit": "m"}  # label_format check
_CHANNEL_MAX = 255.0  # sample_color_rgba channels are 0..255, matplotlib's 0..1


@dataclass(frozen=True)
class HoleLineStyle:
    """Colour (0..1 RGBA), width in points and drawing order of hole-still lines."""

    colour: tuple[float, float, float, float]
    width_pt: float
    zorder: float


@dataclass(frozen=True)
class DispersionTraceSettings:
    """Validated options of the sample traces (meanings: the JSON file)."""

    views: tuple[str, ...]
    sample_color_rgba: tuple[int, ...]
    sample_width_px: int
    label_format: str
    hole_sample_width_pt: float
    hole_sample_zorder: float
    hole_nominal_zorder: float
    hole_clip_margin: float
    key_samples: str
    key_nominal: str
    hole_key_samples: str

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("views must name at least one view preset")
        for view in self.views:
            get_view_preset(view)  # ValueError: unknown view preset
        rgba = _channels(self.sample_color_rgba, 4, "sample_color_rgba")
        object.__setattr__(self, "sample_color_rgba", rgba)
        object.__setattr__(self, "views", tuple(self.views))
        width = self.sample_width_px
        if not isinstance(width, int) or width < 1:
            raise ValueError("sample_width_px must be a positive integer")
        try:
            self.label_format.format(**_PROBE)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"label_format cannot be filled: {exc}") from exc
        hole_width = self.hole_sample_width_pt
        if not (math.isfinite(hole_width) and hole_width > 0.0):
            raise ValueError("hole_sample_width_pt must be positive and finite")
        orders = (self.hole_sample_zorder, self.hole_nominal_zorder)
        if not (all(map(math.isfinite, orders)) and orders[0] < orders[1]):
            raise ValueError("hole_sample_zorder must be finite and below the nominal")
        if not (math.isfinite(self.hole_clip_margin) and self.hole_clip_margin >= 0.0):
            raise ValueError("hole_clip_margin must be non-negative and finite")
        for name in ("key_samples", "key_nominal"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        try:
            self.hole_key_samples.format(n=1)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"hole_key_samples cannot be filled: {exc}") from exc

    def hole_key_text(self, n: int) -> str:
        """The hole still's legend line of ``n`` samples."""
        return self.hole_key_samples.format(n=n)

    def hole_sample_style(self) -> HoleLineStyle:
        """The sample traces' style on a hole still (``sample_color_rgba`` in 0..1)."""
        r, g, b, a = (c / _CHANNEL_MAX for c in self.sample_color_rgba)
        width, order = self.hole_sample_width_pt, self.hole_sample_zorder
        return HoleLineStyle((r, g, b, a), width, order)


def load_trace_settings(path: Path | None = None) -> DispersionTraceSettings:
    """Settings from ``path`` (default: beside this module); ``ValueError`` if bad."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    try:
        return DispersionTraceSettings(**raw)
    except TypeError as exc:
        raise ValueError(f"dispersion trace settings: {exc}") from exc


def spread_label(
    result: sd.DispersionResult, cfg: DispersionTraceSettings, units: DisplayUnits
) -> str:
    """``label_format`` filled with mean +/- sd of carry and side in ``units``.

    Raises ``ValueError`` when no available shot has a carry and a side.
    """
    from src.tools.native_viewer_export.dispersion_hole import spread_text

    return spread_text(result, cfg.label_format, units)


@dataclass(frozen=True)
class DispersionOverlay:
    """Sample flights on the nominal clock, their style and the HUD label."""

    settings: DispersionTraceSettings
    samples: tuple[FlightPlayback, ...]
    label: str

    def flown(self, time_s: float) -> list[np.ndarray]:
        """Each sample's flown points at ``time_s`` (empty before impact)."""
        found = (s.flown(time_s) for s in self.samples)
        return [p for p in found if p is not None and len(p) > 1]

    def paint(
        self, view: str, tile: Any, camera: Any, time_s: float, depth: Any
    ) -> Any:
        """``tile`` with the sample lines (unchanged for a view not configured)."""
        paths = self.flown(time_s)
        if view not in self.settings.views or not paths:
            return tile
        style = self.samples[0].settings
        image = Image.fromarray(tile)
        pen, hide = ImageDraw.Draw(image, "RGBA"), occlusion_for(style, depth)
        colour, width = self.settings.sample_color_rgba, self.settings.sample_width_px
        for points in paths:
            flight_depth.draw_line(
                pen, camera, points, colour, width, style.near_clip_m, hide
            )
        return np.asarray(image, dtype=np.uint8)

    def key_entries(self, nominal_rgba: Sequence[int]) -> tuple[KeyEntry, ...]:
        """Colour key rows: the samples, then the nominal shot (``nominal_rgba``)."""
        cfg = self.settings
        sr, sg, sb = (int(c) for c in cfg.sample_color_rgba[:3])
        nr, ng, nb = (int(c) for c in nominal_rgba[:3])
        return ((cfg.key_samples, (sr, sg, sb)), (cfg.key_nominal, (nr, ng, nb)))

    def paint_hole(
        self, axes: tuple[Any, Any], paths_m: list[np.ndarray], hole: Any, scale: float
    ) -> None:
        """The sample ``paths_m`` (hole frame) thin beneath the hole still's shot.

        Each path is cut at the axis limits (``hole_clip_margin``) and the
        plan's legend gains one line for the samples.
        """
        from matplotlib.lines import Line2D

        style = self.settings.hole_sample_style()
        draw_on_hole(
            axes,
            paths_m,
            hole,
            scale,
            style,
            "dispersion_sample",
            self.settings.hole_clip_margin,
        )
        plan = axes[0]
        old = plan.get_legend()
        if old is None:
            return
        handles = [
            *old.legend_handles,
            Line2D([], [], color=style.colour, lw=style.width_pt),
        ]
        texts = [t.get_text() for t in old.get_texts()]
        label = self.settings.hole_key_text(len(paths_m))
        size = old.get_texts()[0].get_fontsize()
        plan.legend(
            handles,
            [*texts, label],
            loc="upper center",
            fontsize=size,
            ncol=1,
            framealpha=0.9,
        ).set_zorder(old.get_zorder())


def draw_on_hole(
    axes: tuple[Any, Any],
    paths_m: list[np.ndarray],
    hole: Any,
    scale: float,
    style: HoleLineStyle,
    gid: str,
    margin: float = 0.0,
) -> None:
    """Draw each hole-frame path on the (plan, profile) axes of a hole figure.

    ``paths_m`` are (n, 3) world points of ``hole`` (as ``HoleShot.path_m``);
    the plan is (down-range, left), the profile (down-range, height), both in
    display units (``scale`` per metre). Every line gets ``style`` and ``gid``.
    The parts of a path outside the axis limits (widened by ``margin`` of each
    span, ``ValueError`` if negative) are not drawn; the crossing is cut exactly.
    """
    from src.tools.golf_environment.shots import hole_axes

    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError("scale must be positive and finite")
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError("margin must be non-negative and finite")
    tee, forward, left = hole_axes(hole)
    plan, side = axes
    kw = {"color": style.colour, "lw": style.width_pt, "zorder": style.zorder}
    for path in paths_m:
        rel = np.asarray(path, dtype=float) - tee
        x = scale * (rel @ forward)
        for ax, y in ((plan, scale * (rel @ left)), (side, scale * rel[:, 2])):
            xb, yb = _widened(ax.get_xlim(), margin), _widened(ax.get_ylim(), margin)
            for run in clip_to_bounds(np.column_stack([x, y]), xb, yb):
                ax.plot(run[:, 0], run[:, 1], gid=gid, **kw)


def _widened(limits: tuple[float, float], margin: float) -> tuple[float, float]:
    """``limits`` (low, high) widened by ``margin`` of their span each side."""
    lo, hi = sorted(limits)
    pad = margin * (hi - lo)
    return lo - pad, hi + pad


def clip_to_bounds(
    xy: np.ndarray, x_bounds: tuple[float, float], y_bounds: tuple[float, float]
) -> list[np.ndarray]:
    """The runs of the polyline ``xy`` ((n, 2)) inside the box, cut at its edges.

    Each run is (m, 2) with m >= 2; a path wholly outside gives ``[]``, one wholly
    inside gives ``[xy]``. Raises ``ValueError`` for a bad shape or inverted bounds.
    """
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 1:
        raise ValueError("xy must be an (n, 2) array of points")
    if not (x_bounds[0] <= x_bounds[1] and y_bounds[0] <= y_bounds[1]):
        raise ValueError("bounds must be (low, high) with low <= high")
    lo = np.array([x_bounds[0], y_bounds[0]])
    hi = np.array([x_bounds[1], y_bounds[1]])
    runs: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        d = b - a
        t0, t1 = 0.0, 1.0
        for k in range(2):  # Liang-Barsky on the segment a + t d
            if d[k] == 0.0:
                if not lo[k] <= a[k] <= hi[k]:
                    t0, t1 = 1.0, 0.0
                continue
            ta, tb = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:  # wholly outside: close the open run
            if current:
                runs.append(current)
            current = []
            continue
        start, end = a + t0 * d, a + t1 * d
        if not current:
            current = [start]
        current.append(end)
        if t1 < 1.0:  # left the box: close the run
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [np.array(r) for r in runs if len(r) >= 2]


def overlay_for(
    result: sd.DispersionResult,
    nominal: FlightPlayback,
    cfg: DispersionTraceSettings | None = None,
) -> DispersionOverlay:
    """The overlay of ``result``'s available shots, timed from ``nominal``'s impact.

    Preconditions: ``nominal`` has a flight path and at least one shot has a
    world path (``ValueError``). Postcondition: one sample per such shot, each
    starting at the nominal impact time with the nominal ball radius.
    """
    if nominal.path_m is None:
        raise ValueError("the nominal shot has no flight path to time samples by")
    cfg = cfg or load_trace_settings()
    shots = [o for o in result.available if o.path_m is not None]
    if not shots:
        raise ValueError("no shot has a world flight path to draw")
    samples = tuple(
        FlightPlayback(
            nominal.settings,
            nominal.status,
            nominal.impact_time_s,
            np.asarray(o.times_s, dtype=float),
            np.asarray(o.path_m, dtype=float),
            nominal.radius_m,
        )
        for o in shots
    )
    label = spread_label(result, cfg, nominal.settings.display_units)
    return DispersionOverlay(cfg, samples, label)


def with_dispersion(
    overlays: Any,
    swing: Any,
    raw_settings: Any,
    played_settings: Any,
    evaluate: sd.Evaluator | None = None,
) -> Any:
    """``overlays`` whose flight carries the sample traces, when asked for.

    ``swing`` and ``raw_settings`` are the export's input before
    ``prepare_shot`` (each sample applies its own roll and ball shift);
    ``played_settings`` gives the club roll and ball shift the clip plays,
    the centre of the samples. Unchanged when the option is off or the nominal
    shot has no flight. ``evaluate`` replaces the replay (tests).
    """
    tracer = getattr(raw_settings, "ball_flight", None)
    cfg = getattr(tracer, "dispersion_traces", None)
    flight = None if overlays is None else overlays.flight
    if cfg is None or flight is None:
        return overlays
    requested = tuple(getattr(raw_settings, "views", ()))
    if not set(cfg.views) & set(requested):
        logger.warning(
            "dispersion traces are drawn on %s, none of the requested views %s "
            "(views of dispersion_traces_settings.json)",
            ", ".join(cfg.views),
            ", ".join(requested),
        )
    if flight.path_m is None:
        logger.warning("no dispersion traces: nominal flight %s", flight.status)
        return overlays
    roll = played_settings.club_roll_in_hands_deg
    shift = played_settings.ball_position_shift_m
    if not (math.isfinite(roll) and math.isfinite(shift)):
        raise ValueError("the played club roll and ball shift must be finite")
    study = replace(
        sd.load_dispersion_settings(), base_club_roll_deg=roll, base_ball_shift_m=shift
    )
    result = sd.run_dispersion(swing, raw_settings, sd.sample_setups(study), evaluate)
    try:
        overlay = overlay_for(result, flight, cfg)
    except ValueError as exc:
        logger.warning("no dispersion traces: %s", exc)
        return overlays
    return replace(overlays, flight=replace(flight, dispersion=overlay))
