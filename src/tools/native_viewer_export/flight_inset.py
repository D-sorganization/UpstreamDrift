"""Flight inset of native export flight clips (#15150).

From behind the golfer a long carry is foreshortened onto the horizon, so the
3D tracer cannot be read. This opt-in 2D panel, like a broadcast shot tracer
graphic, shows the same flight twice: a side profile (height against down-range
distance, vertical scale exaggerated by a setting) and a plan view (lateral
offset against down-range distance with the target line). Both are revealed in
step with the tracer and keep the full curve and labels once landed. The
numbers (carry, apex, side offset) are the flight model's, never recomputed.
The panel is placed clear of the view label, the impact panels and the HUD rows.
Options: ``flight_inset_settings.json`` (see ``sources``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from src.shared.python.golf_view_presets import get_view_preset
from src.tools.native_viewer_export import compositor, ellipsoid_panel
from src.tools.native_viewer_export.landing_marker import rgba_channels

if TYPE_CHECKING:
    from src.tools.native_viewer_export.compositor import HudInfo
    from src.tools.native_viewer_export.core import ClipOverlays
    from src.tools.native_viewer_export.flight_tracer import FlightPlayback
    from src.tools.native_viewer_export.rollout import Rollout

Array = NDArray[np.float64]
Image8 = NDArray[np.uint8]
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
    "roll_color_rgba",
)
_RGB = ("text_color_rgb", "title_color_rgb")
_FORMATS = {
    "apex_label_format": {"apex": 1.0, "unit": "m"},
    "carry_label_format": {"carry": 1.0, "unit": "m"},
    "roll_label_format": {"roll": 1.0, "total": 1.0, "unit": "m"},
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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
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
    min_flight_fraction_shown: float = 0.5
    curve_color_rgba: tuple[int, ...] = (255, 70, 40, 255)
    curve_width_px: int = 3
    ground_color_rgba: tuple[int, ...] = (200, 200, 200, 200)
    ground_width_px: int = 1
    target_line_color_rgba: tuple[int, ...] = (255, 255, 255, 200)
    target_line_width_px: int = 1
    dot_color_rgba: tuple[int, ...] = (255, 255, 255, 255)
    dot_radius_px: int = 4
    marker_color_rgba: tuple[int, ...] = (255, 220, 40, 255)
    roll_color_rgba: tuple[int, ...] = (120, 220, 255, 255)
    roll_width_px: int = 5
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
    roll_label_format: str = "Roll {roll:.0f} {unit}  Total {total:.0f} {unit} (model)"
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
            (
                "curve_width_px",
                "ground_width_px",
                "target_line_width_px",
                "roll_width_px",
            ),
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
        _fraction(self, ("profile_share", "min_flight_fraction_shown"))
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
    label_rows: int = 1,
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
    base = split - r - font_px * 1.4 * label_rows  # carry (+ roll) rows
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


@dataclass(frozen=True)
class LabelSpec:
    """One inset label: text, anchor point and the box it must stay inside."""

    text: str
    xy: tuple[float, float]
    size: int
    colour: tuple[int, ...]
    anchor: str
    bounds: tuple[float, float, float, float]


def _placed(spec: LabelSpec) -> tuple[int, int, int, tuple[int, int, int, int]]:
    """Top-left ``(x, y)``, ``size`` and the box ``(x0, y0, x1, y1)`` of ``spec``.

    The size shrinks (to at least 9 px) until the text is as narrow as its
    bounds, and the anchor is clamped so the text lies inside them.
    """
    x0, y0, x1, y1 = spec.bounds
    size = spec.size
    box = compositor.text_box(spec.text, (0, 0), size)
    while size > 9 and box[2] - box[0] > x1 - x0:
        size -= 1
        box = compositor.text_box(spec.text, (0, 0), size)
    width, height = box[2] - box[0], box[3] - box[1]
    shift = {"left": 0.0, "centre": width / 2.0, "right": float(width)}[spec.anchor]
    x = max(x0, min(spec.xy[0] - shift, x1 - width))
    y = max(y0, min(spec.xy[1], y1 - height))
    return int(x), int(y), size, compositor.text_box(spec.text, (int(x), int(y)), size)


def label_box(spec: LabelSpec) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` the label of ``spec`` is drawn in."""
    return tuple(int(v) for v in _placed(spec)[3])  # type: ignore[return-value]


def _text(image: Image8, spec: LabelSpec) -> Image8:
    """Outlined label of ``spec`` by the HUD renderer, inside its bounds."""
    x, y, size, _ = _placed(spec)
    rgb = (int(spec.colour[0]), int(spec.colour[1]), int(spec.colour[2]))
    return compositor.draw_label(image, spec.text, (x, y), size=size, fill=rgb)


def _line(
    pen: ImageDraw.ImageDraw, pts: Array, colour: tuple[int, ...], w: int
) -> None:
    if len(pts) > 1:
        pen.line([(float(u), float(v)) for u, v in pts], fill=colour, width=w)
        for u, v in pts[[0, -1]]:  # round caps hide the joint gaps
            pen.ellipse([u - w / 2, v - w / 2, u + w / 2, v + w / 2], fill=colour)


def _dot(pen: ImageDraw.ImageDraw, xy: Array, r: int, colour: tuple[int, ...]) -> None:
    u, v = float(xy[0]), float(xy[1])
    pen.ellipse([u - r, v - r, u + r, v + r], fill=colour)


@dataclass(frozen=True)
class InsetData:
    """What the inset shows at one time, in metres from the launch point."""

    revealed: Array
    full: Array
    apex_m: float
    carry_m: float
    lateral_m: float
    landed: bool
    started: bool = True
    roll: Rollout | None = None

    @property
    def roll_end(self) -> Array:
        """Rest point (down, left, up) of the roll, the landing point without one."""
        end = self.full[-1].copy()
        if self.roll is not None:
            end[:2] += self.roll.end_offset_m
        return end

    @property
    def passed_apex(self) -> bool:
        """The revealed curve has reached the highest point of the flight."""
        return self.started and bool(
            self.revealed[:, 2].max() >= self.full[:, 2].max() - 1e-9
        )


def inset_data(flight: FlightPlayback, time_s: float) -> InsetData | None:
    """The inset data of ``flight`` at ``time_s`` (nothing revealed before launch).

    ``None`` when the flight has no drawable path or no model numbers.
    """
    path, axes = flight.path_m, flight.axes_m
    numbers = (flight.carry_m, flight.apex_m, flight.lateral_m)
    if path is None or axes is None or any(n is None for n in numbers):
        return None
    full = flight_coordinates(path, axes)
    points = flight.flown(time_s)
    landed = time_s - flight.impact_time_s >= flight.duration_s
    shown = full[:1] if points is None else flight_coordinates(points, axes, path[0])
    carry, apex, lateral = (float(n or 0.0) for n in numbers)
    started = points is not None
    return InsetData(shown, full, apex, carry, lateral, landed, started, flight.rollout)


def _apex_pixel(profile: Plot, data: InsetData) -> Array:
    """Pixel ``(u, v)`` of the flight's highest point in the profile plot."""
    top = int(np.argmax(data.full[:, 2]))
    return profile.pixels(data.full[top : top + 1, 0], data.full[top : top + 1, 2])[0]


def _fmt(settings: FlightInsetSettings, name: str, **values: Any) -> str:
    return str(getattr(settings, name).format(unit=settings.unit, **values))


def _draw_plots(
    pen: ImageDraw.ImageDraw,
    plots: tuple[Plot, Plot],
    data: InsetData,
    cfg: FlightInsetSettings,
) -> None:
    """Axes, curves, markers and the ball dot of both plots onto ``pen``."""
    profile, plan = plots
    left, right = profile.box[0], profile.box[2]
    base = (profile.origin_y, plan.origin_y)
    pen.line(
        [(left, base[0]), (right, base[0])], cfg.ground_color_rgba, cfg.ground_width_px
    )
    pen.line(
        [(left, base[1]), (right, base[1])],
        cfg.target_line_color_rgba,
        cfg.target_line_width_px,
    )
    down, left_m, up = (data.revealed[:, i] for i in range(3))
    a_pix = profile.pixels(down, up)
    p_pix = plan.pixels(down, left_m)
    for pix in (a_pix, p_pix) if data.started else ():
        _line(pen, pix, cfg.curve_color_rgba, cfg.curve_width_px)
        _dot(pen, pix[-1], cfg.dot_radius_px, cfg.dot_color_rgba)
    marker = cfg.marker_radius_px
    if data.passed_apex:
        _dot(pen, _apex_pixel(profile, data), marker, cfg.marker_color_rgba)
    if data.landed:
        for pix in (a_pix, p_pix):
            _dot(pen, pix[-1], marker, cfg.marker_color_rgba)
        _draw_roll(pen, plots, data, cfg)


def _draw_roll(
    pen: ImageDraw.ImageDraw,
    plots: tuple[Plot, Plot],
    data: InsetData,
    cfg: FlightInsetSettings,
) -> None:
    """The roll as a ground track from the landing point to the rest point."""
    if data.roll is None:
        return
    pair = np.vstack([data.full[-1], data.roll_end])
    for plot, axis in zip(plots, (2, 1), strict=True):
        pix = plot.pixels(pair[:, 0], pair[:, axis])
        _line(pen, pix, cfg.roll_color_rgba, cfg.roll_width_px)
        _dot(pen, pix[-1], cfg.marker_radius_px, cfg.roll_color_rgba)


def _uncrowded(titles: list[LabelSpec], labels: list[LabelSpec]) -> list[LabelSpec]:
    """``labels`` in priority order without any that would cover another text.

    The apex label is the least important (carry, roll and side come first), so
    a short inset loses it rather than printing two labels over each other.
    """
    ranked = sorted(labels, key=lambda spec: spec.text.startswith("Apex"))
    kept: list[LabelSpec] = []
    boxes = [label_box(t) for t in titles]
    for spec in ranked:
        box = label_box(spec)
        clear = all(
            box[2] <= o[0] or o[2] <= box[0] or box[3] <= o[1] or o[3] <= box[1]
            for o in boxes
        )
        if clear:
            kept.append(spec)
            boxes.append(box)
    return [spec for spec in labels if spec in kept]


def label_specs(
    plots: tuple[Plot, Plot], data: InsetData, cfg: FlightInsetSettings, px: int
) -> tuple[list[LabelSpec], list[LabelSpec]]:
    """``(titles, labels)``: every inset text with the box it must stay inside.

    Titles sit in the row above their plot; every other label is clamped inside
    its plot's axes, so none leaves the inset or covers a title.
    """
    profile, plan = plots
    scale, gap, row = cfg.unit_scale, float(cfg.marker_radius_px + 3), px * 1.4
    titles = []
    for plot, title in ((profile, cfg.profile_title), (plan, cfg.plan_title)):
        x0, y0, x1, _ = plot.box
        bounds = (x0, y0 - row, x1, y0)
        titles.append(
            LabelSpec(title, (x0, y0 - row), px, cfg.title_color_rgb, "left", bounds)
        )
    labels: list[LabelSpec] = []
    text_rgb = cfg.text_color_rgb
    if data.passed_apex:
        u, v = _apex_pixel(profile, data)
        text = _fmt(cfg, "apex_label_format", apex=data.apex_m * scale)
        labels.append(
            LabelSpec(text, (u, v - gap - row), px, text_rgb, "centre", profile.box)
        )
    if data.landed:
        end = data.full[-1]
        u, v = profile.pixels(end[:1], end[2:])[0]
        text = _fmt(cfg, "carry_label_format", carry=data.carry_m * scale)
        spec = LabelSpec(text, (u + gap, v + gap), px, text_rgb, "right", profile.box)
        labels.append(spec)
        if data.roll is not None:
            ru, rv = profile.pixels(data.roll_end[:1], data.roll_end[2:])[0]
            roll, total = data.roll.roll_m * scale, data.roll.total_m * scale
            text = _fmt(cfg, "roll_label_format", roll=roll, total=total)
            at = (ru + gap, label_box(spec)[3])  # directly beneath the carry
            colour = cfg.roll_color_rgba
            labels.append(LabelSpec(text, at, px, colour, "right", profile.box))
        u, v = plan.pixels(end[:1], end[1:2])[0]
        side = cfg.side_left_text if data.lateral_m < 0.0 else cfg.side_right_text
        text = _fmt(
            cfg, "side_label_format", side=abs(data.lateral_m) * scale, direction=side
        )
        # on the empty side of the target line, away from the curve
        y = plan.origin_y - row - gap if data.lateral_m >= 0.0 else plan.origin_y + gap
        labels.append(LabelSpec(text, (u + gap, y), px, text_rgb, "right", plan.box))
    return titles, _uncrowded(titles, labels)


def _draw_labels(
    image: Image8,
    plots: tuple[Plot, Plot],
    data: InsetData,
    cfg: FlightInsetSettings,
    font_px: int,
) -> Image8:
    """Titles and, as the flight earns them, the apex, carry, roll and side labels."""
    titles, labels = label_specs(plots, data, cfg, font_px)
    for spec in (*titles, *labels):
        image = _text(image, spec)
    return image


def font_px_for(shape: tuple[int, ...], settings: FlightInsetSettings) -> int:
    """Label size in pixels for a frame of ``shape`` (at least 9)."""
    scaled = int(shape[0]) * settings.font_height_fraction * settings.font_scale
    return max(9, round(scaled))


def paint(
    img: Image8,
    flight: FlightPlayback,
    time_s: float,
    rect: Rect,
    settings: FlightInsetSettings,
) -> Image8:
    """Copy of ``img`` with the inset of ``flight`` at ``time_s`` in ``rect``.

    Only pixels inside ``rect`` change. Before launch the axes are empty (or
    nothing is drawn with ``hide_before_launch``). Raises ``ValueError`` for a
    ``rect`` outside the frame.
    """
    x0, y0, x1, y1 = rect
    if not (0 <= x0 < x1 <= img.shape[1] and 0 <= y0 < y1 <= img.shape[0]):
        raise ValueError(f"rect {rect} lies outside the frame {img.shape[:2]}")
    data = inset_data(flight, time_s)
    if data is None or (not data.started and settings.hide_before_launch):
        return img
    reach = np.vstack([data.full, data.roll_end])
    extent = (reach[:, 0].max(), data.apex_m, np.abs(reach[:, 1]).max())
    size = font_px_for(img.shape, settings)
    rows = 1 if data.roll is None else 2
    plots = plots_for((0, 0, x1 - x0, y1 - y0), settings, size, extent, rows)
    region = img[y0:y1, x0:x1].astype(np.float32) * (1.0 - settings.opacity)
    pil = Image.fromarray(region.astype(np.uint8))
    _draw_plots(ImageDraw.Draw(pil, "RGBA"), plots, data, settings)
    crop = _draw_labels(np.asarray(pil), plots, data, settings, size)
    out = img.copy()
    out[y0:y1, x0:x1] = crop
    return out


def obstacles_for(
    overlays: ClipOverlays,
    feed: Any,
    index: int,
    shape: tuple[int, ...],
    limit_px: int,
) -> list[Rect]:
    """Boxes of the panels drawn in a frame of ``shape``: factors or card, ellipsoids."""
    boxes: list[Rect] = []
    if overlays.factors is not None:
        boxes.append(overlays.factors.bbox(shape))
    elif overlays.card is not None:
        boxes.append(compositor.impact_card_bbox(shape, overlays.card))
    panel = overlays.ellipsoids
    if panel is not None and feed is not None:
        boxes.append(panel.fit(shape, feed.hand_panel_rows(index), limit_px)[0])
    return boxes


def draw_all(
    overlays: ClipOverlays | None,
    feed: Any,
    index: int,
    hud: HudInfo,
    outputs: dict[str, Image8],
) -> dict[str, Image8]:
    """``outputs`` with the flight inset of frame ``index`` where it is enabled.

    Fixed views only (a view that follows a target has no flight). The inset
    sits below the view label, clear of the impact panels and above the HUD rows.
    """
    flight = None if overlays is None else overlays.flight
    cfg = None if flight is None else flight.settings.inset
    if overlays is None or flight is None or cfg is None or not cfg.enabled:
        return outputs
    out = dict(outputs)
    for name, frame in outputs.items():
        if not shows_inset(cfg, name):
            continue
        limit = ellipsoid_panel.hud_top_px(frame.shape, hud)
        obstacles = obstacles_for(overlays, feed, index, frame.shape, limit)
        top = compositor.view_label_box(frame.shape, "")[3]
        top += compositor.HUD_LAYOUT.panel_gap_px
        rect = layout_rect(frame.shape, cfg, obstacles, top, limit)
        if rect is not None:
            out[name] = paint(frame, flight, hud.time_s, rect, cfg)
    return out


def for_clip(
    overlays: ClipOverlays | None, window: tuple[float, float] | None
) -> ClipOverlays | None:
    """``overlays`` with the inset switched off for a clip too short for it.

    Decided once per clip, not per frame. ``window`` is the clip's swing-time
    window (an impact-window clip) or ``None`` for a clip held until the ball
    lands, which always keeps the inset. A windowed clip keeps it only when its
    last frame reaches ``min_flight_fraction_shown`` of the flight time after
    impact; otherwise the plots would be empty lines hiding the club. Switching
    it off also gives the ring carry label back, as before the inset existed.
    """
    flight = None if overlays is None else overlays.flight
    cfg = None if flight is None else flight.settings.inset
    if window is None or flight is None or cfg is None or not cfg.enabled:
        return overlays
    if flight.duration_s <= 0.0:
        return overlays
    shown = (window[1] - flight.impact_time_s) / flight.duration_s
    if shown >= cfg.min_flight_fraction_shown:
        return overlays
    quiet = replace(flight.settings, inset=replace(cfg, enabled=False))
    return replace(overlays, flight=replace(flight, settings=quiet))  # type: ignore[type-var]


def shows_inset(cfg: FlightInsetSettings | None, view: str) -> bool:
    """Whether ``view`` gets the inset: enabled, on a fixed-camera preset view."""
    return cfg is not None and cfg.enabled and _fixed(view)


def _fixed(view: str) -> bool:
    """A preset view with a fixed camera; the composed 2x2 grid is not one."""
    try:
        return get_view_preset(view).tracks is None
    except (KeyError, ValueError):
        return False
