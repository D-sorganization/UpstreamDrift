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
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.shared.python.golf_view_presets import get_view_preset
from src.tools.native_viewer_export import flight_depth
from src.tools.native_viewer_export import shot_dispersion as sd
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


@dataclass(frozen=True)
class DispersionTraceSettings:
    """Validated options of the sample traces (meanings: the JSON file)."""

    views: tuple[str, ...]
    sample_color_rgba: tuple[int, ...]
    sample_width_px: int
    label_format: str

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
