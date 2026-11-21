"""Settings, layout and geometry of the flight inset (#15150).

From behind the golfer a long carry is foreshortened onto the horizon, so the
3D tracer cannot be read. The opt-in 2D inset shows the same flight as a side
profile (height against down-range distance, vertical scale exaggerated by a
setting) and a plan view (lateral offset against down-range distance). This
module holds the validated options, the placement of the inset clear of the
impact panels and the HUD rows, and the flight in target-frame coordinates.
Nothing is drawn yet. Options: flight_inset_settings.json (see sources).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tools.native_viewer_export.landing_marker import rgba_channels

Array = NDArray[np.float64]
Rect = tuple[int, int, int, int]
Box = tuple[float, float, float, float]
SETTINGS_PATH = Path(__file__).with_name("flight_inset_settings.json")
POSITIONS = ("top_right", "top_left", "bottom_right", "bottom_left")
_RGBA = (
    "curve_color_rgba",
    "ground_color_rgba",
    "target_line_color_rgba",
    "dot_color_rgba",
    "marker_color_rgba",
)
_RGB = ("text_color_rgb", "title_color_rgb")
_FORMATS = {
    "apex_label_format": {"apex": 1.0, "unit": "m"},
    "carry_label_format": {"carry": 1.0, "unit": "m"},
    "side_label_format": {"side": 1.0, "unit": "m", "direction": "R"},
}


def _positive(
    settings: object, names: Sequence[str], kind: type[int] | type[float]
) -> None:
    for name in names:
        value: Any = getattr(settings, name)
        numeric = (int, float) if kind is float else int
        if isinstance(value, bool) or not isinstance(value, numeric) or value <= 0:
            raise ValueError(f"{name} must be a positive {kind.__name__}")


def _fraction(settings: object, names: Sequence[str]) -> None:
    for name in names:
        value = getattr(settings, name)
        if not (math.isfinite(value) and 0.0 < value <= 1.0):
            raise ValueError(f"{name} must lie in (0, 1]")


@dataclass(frozen=True)
class FlightInsetSettings:
    """Validated options of the flight inset (meanings: the settings JSON)."""

    enabled: bool = False
    position: str = "top_right"
    width_fraction: float = 0.3
    height_fraction: float = 0.4
    min_height_fraction: float = 0.2
    edge_margin_px: int = 8
    panel_gap_px: int = 8
    padding_px: int = 10
    profile_share: float = 0.58
    opacity: float = 0.62
    vertical_exaggeration: float | None = 6.0
    min_lateral_half_span_m: float = 15.0
    hide_before_launch: bool = False
    curve_color_rgba: tuple[int, ...] = (255, 70, 40, 255)
    curve_width_px: int = 3
    ground_color_rgba: tuple[int, ...] = (200, 200, 200, 200)
    ground_width_px: int = 1
    target_line_color_rgba: tuple[int, ...] = (255, 255, 255, 200)
    target_line_width_px: int = 1
    dot_color_rgba: tuple[int, ...] = (255, 255, 255, 255)
    dot_radius_px: int = 4
    marker_color_rgba: tuple[int, ...] = (255, 220, 40, 255)
    marker_radius_px: int = 4
    text_color_rgb: tuple[int, ...] = (255, 255, 255)
    title_color_rgb: tuple[int, ...] = (255, 235, 90)
    font_height_fraction: float = 0.022
    font_scale: float = 1.0
    profile_title: str = "Side Profile"
    plan_title: str = "Plan View"
    unit: str = "m"
    unit_scale: float = 1.0
    apex_label_format: str = "Apex {apex:.0f} {unit}"
    carry_label_format: str = "Carry {carry:.0f} {unit}"
    side_label_format: str = "{side:.0f} {unit} {direction}"
    side_left_text: str = "L"
    side_right_text: str = "R"

    def __post_init__(self) -> None:
        for name in ("enabled", "hide_before_launch"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be true or false")
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}")
        for name in _RGBA:
            object.__setattr__(self, name, rgba_channels(getattr(self, name), name))
        for name in _RGB:
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise TypeError(f"{name} must be 3 integers")
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in value):
                raise ValueError(f"{name} channels must be integers in 0..255")
            object.__setattr__(self, name, tuple(value))
        self._validate_numbers()
        self._validate_texts()

    def _validate_numbers(self) -> None:
        for name in ("edge_margin_px", "panel_gap_px", "padding_px"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _positive(
            self,
            ("curve_width_px", "ground_width_px", "target_line_width_px"),
            int,
        )
        _positive(self, ("dot_radius_px", "marker_radius_px"), int)
        _positive(
            self,
            ("min_lateral_half_span_m", "font_height_fraction", "font_scale"),
            float,
        )
        _positive(self, ("unit_scale",), float)
        _fraction(self, ("width_fraction", "height_fraction", "min_height_fraction"))
        _fraction(self, ("profile_share",))
        if self.min_height_fraction > self.height_fraction:
            raise ValueError("min_height_fraction must not exceed height_fraction")
        if not (math.isfinite(self.opacity) and 0.0 <= self.opacity <= 1.0):
            raise ValueError("opacity must lie in [0, 1]")
        exag = self.vertical_exaggeration
        if exag is not None and not (math.isfinite(exag) and exag >= 1.0):
            raise ValueError("vertical_exaggeration must be null or at least 1")

    def _validate_texts(self) -> None:
        for name in ("profile_title", "plan_title", "unit"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        for name in ("side_left_text", "side_right_text"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name, probe in _FORMATS.items():
            try:
                getattr(self, name).format(**probe)
            except (KeyError, IndexError, ValueError, AttributeError) as exc:
                raise ValueError(f"{name} is not usable: {exc!r}") from exc


def load_inset_settings(path: Path | None = None) -> FlightInsetSettings:
    """Settings from ``path`` (default: the file beside this module).

    Raises ``ValueError``/``TypeError`` for a missing, unknown or bad entry.
    """
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    document = {k: tuple(v) if isinstance(v, list) else v for k, v in document.items()}
    try:
        return FlightInsetSettings(**document)
    except TypeError as exc:
        if "argument" not in str(exc):
            raise
        raise ValueError(f"flight inset settings: {exc}") from exc


def _overlap_x(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2]


def layout_rect(
    shape: tuple[int, ...],
    settings: FlightInsetSettings,
    obstacles: Sequence[Rect],
    top_px: int,
    limit_px: int,
) -> Rect | None:
    """Pixel box ``(x0, y0, x1, y1)`` of the inset in a frame of ``shape``.

    ``top_px`` is the first free row (below the view label), ``limit_px`` the
    first row of the HUD rows. A top inset stacks below, a bottom inset above,
    every obstacle sharing its columns; the height shrinks to fit and is
    ``None`` (not drawn) below ``min_height_fraction``. Postcondition: the box
    lies inside the frame, above ``limit_px`` and clear of every obstacle.
    """
    height, width = int(shape[0]), int(shape[1])
    margin, gap = settings.edge_margin_px, settings.panel_gap_px
    box_w = min(int(round(width * settings.width_fraction)), width - 2 * margin)
    x0 = margin if settings.position.endswith("left") else width - margin - box_w
    column: Rect = (x0, 0, x0 + box_w, height)
    shared = [o for o in obstacles if _overlap_x(column, o)]
    low, high = max(top_px, margin), min(limit_px, height - margin)
    if settings.position.startswith("top"):
        low = max([low, *(o[3] + gap for o in shared)])
    else:
        high = min([high, *(o[1] - gap for o in shared)])
    want = int(round(height * settings.height_fraction))
    least = int(round(height * settings.min_height_fraction))
    box_h = min(want, high - low)
    if box_w < 1 or box_h < max(least, 1):
        return None
    if settings.position.startswith("top"):
        return (x0, low, x0 + box_w, low + box_h)
    return (x0, high - box_h, x0 + box_w, high)


def flight_coordinates(
    path_m: Array, axes_m: Array, origin_m: Array | None = None
) -> Array:
    """``path_m`` as (down-range, left, up) metres from ``origin_m``.

    ``axes_m`` rows are the target-frame axes (target, left, up) that launched
    the flight; ``origin_m`` defaults to the first point. Raises ``ValueError``
    for a path or axes of the wrong shape.
    """
    path, axes = np.asarray(path_m, dtype=float), np.asarray(axes_m, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or axes.shape != (3, 3):
        raise ValueError("path_m must be (n, 3) and axes_m (3, 3)")
    origin = path[0] if origin_m is None else np.asarray(origin_m, dtype=float)
    return (path - origin) @ axes.T  # type: ignore[no-any-return]


@dataclass(frozen=True)
class Plot:
    """One plot area and its scales: pixels = origin + metres * scale."""

    box: Box
    origin_x: float
    origin_y: float
    scale_x: float
    scale_y: float

    def pixels(self, down_m: Array, up_m: Array) -> Array:
        """Pixel ``(u, v)`` rows of down-range and vertical metres (up is -v)."""
        return np.column_stack(
            [
                self.origin_x + np.asarray(down_m) * self.scale_x,
                self.origin_y - np.asarray(up_m) * self.scale_y,
            ]
        )


def _gain(exaggeration: float | None, scale_x: float, fit: float) -> float:
    """Vertical scale: ``exaggeration`` times ``scale_x`` but never beyond ``fit``."""
    return fit if exaggeration is None else min(exaggeration * scale_x, fit)


def plots_for(
    rect: Rect,
    settings: FlightInsetSettings,
    font_px: int,
    extent_m: tuple[float, float, float],
) -> tuple[Plot, Plot]:
    """Profile and plan plots of the inset in ``rect`` for a flight.

    ``extent_m`` is (down-range end, apex height, largest lateral offset). Both
    plots share the down-range scale. The curve fits inside its plot by
    construction (postcondition), whatever the exaggeration.
    """
    down, apex, lateral = (max(float(v), 1e-6) for v in extent_m)
    x0, y0, x1, y1 = (float(v) for v in rect)
    pad, title = float(settings.padding_px), font_px * 1.4
    r = float(max(settings.marker_radius_px, settings.dot_radius_px)) + 1.0
    inner = y1 - y0 - 3 * pad
    split = y0 + pad + inner * settings.profile_share
    left, right = x0 + pad + r, x1 - pad - r
    scale_x = (right - left) / down
    exag = settings.vertical_exaggeration
    top = y0 + pad + title
    base = split - r - font_px * 1.4  # room beneath the ground line for the carry
    fit = (base - top - font_px * 1.5) / apex
    profile = Plot(
        (x0 + pad, top, x1 - pad, split),
        left,
        base,
        scale_x,
        _gain(exag, scale_x, max(fit, 1e-6)),
    )
    low = split + pad + title
    bottom = y1 - pad
    half = max(lateral, settings.min_lateral_half_span_m)
    mid = (low + bottom) / 2.0
    fit_plan = ((bottom - low) / 2.0 - r) / half
    plan = Plot(
        (x0 + pad, low, x1 - pad, bottom),
        left,
        mid,
        scale_x,
        _gain(exag, scale_x, max(fit_plan, 1e-6)),
    )
    return profile, plan
