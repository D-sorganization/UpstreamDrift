"""Landing marker, ground track and carry label of the shot tracer (#15150).

A ground ring at the landing point puts a far landing on the turf instead of
the horizon. The ring has a minimum screen size: a marker size, not the landing
area. Optional aids make the landing readable in a wide clip: the ground track
(the flight path projected onto the ground, revealed with the tracer), a drop
line from the ball to its ground point while in flight, and a carry label once
landed. The label shows the carry the flight model reports, never a recomputed
one. Options: ``flight_tracer_settings.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from src.tools.native_viewer_export import flight_depth
from src.tools.native_viewer_export.flight_depth import project

if TYPE_CHECKING:
    from src.tools.native_viewer_export.flight_tracer import TracerSettings

Array = NDArray[np.float64]
Image8 = NDArray[np.uint8]
WHEN = ("at_landing", "from_start")


@dataclass(frozen=True)
class LandingMarkerSettings:
    """Validated options of the landing marker (meanings: the settings JSON)."""

    when: str = "at_landing"
    ground_z_m: float = 0.0
    radius_m: float = 3.0
    min_radius_px: int = 10
    min_aspect: float = 0.35
    color_rgba: tuple[int, ...] = (255, 220, 40, 255)
    line_width_px: int = 3
    ground_track: bool = False
    ground_track_color_rgba: tuple[int, ...] = (255, 220, 40, 120)
    ground_track_width_px: int = 2
    drop_line: bool = False
    drop_line_color_rgba: tuple[int, ...] = (255, 255, 255, 110)
    drop_line_width_px: int = 1
    occlude_track: bool = True
    carry_label: bool = False
    carry_label_format: str = "Carry {carry:.0f} {unit}"
    carry_label_unit: str = "m"
    carry_label_scale: float = 1.0
    carry_label_size_px: int = 22
    carry_label_color_rgba: tuple[int, ...] = (255, 255, 255, 255)
    carry_label_outline_rgba: tuple[int, ...] = (0, 0, 0, 255)
    carry_label_outline_px: int = 2
    carry_label_gap_px: int = 8
    grid_carry_label_margin_px: int = 12
    grid_carry_label_keep_out: tuple[float, ...] | None = (0.25, 0.1, 0.75, 0.9)

    def __post_init__(self) -> None:
        if self.when not in WHEN:
            raise ValueError(f"when must be one of {WHEN}, got {self.when!r}")
        for name in ("radius_m", "min_aspect"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("min_radius_px", "line_width_px"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._validate_aids()

    def _validate_aids(self) -> None:
        for name in ("ground_track", "drop_line", "occlude_track", "carry_label"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be true or false")
        for name in (
            "ground_track_color_rgba",
            "drop_line_color_rgba",
            "carry_label_color_rgba",
            "carry_label_outline_rgba",
        ):
            object.__setattr__(self, name, rgba_channels(getattr(self, name), name))
        for name in ("ground_track_width_px", "drop_line_width_px"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("carry_label_size_px",):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("carry_label_outline_px", "carry_label_gap_px"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._validate_grid_label()
        if not (math.isfinite(self.carry_label_scale) and self.carry_label_scale > 0):
            raise ValueError("carry_label_scale must be positive and finite")
        if not isinstance(self.carry_label_unit, str):
            raise TypeError("carry_label_unit must be a string")
        try:
            self.carry_label_format.format(carry=1.0, unit=self.carry_label_unit)
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ValueError(f"carry_label_format is not usable: {exc!r}") from exc

    def _validate_grid_label(self) -> None:
        margin = self.grid_carry_label_margin_px
        if isinstance(margin, bool) or not isinstance(margin, int):
            raise TypeError("grid_carry_label_margin_px must be an integer")
        if margin < 0:
            raise ValueError("grid_carry_label_margin_px must not be negative")
        keep = self.grid_carry_label_keep_out
        if keep is None:
            return
        ok = isinstance(keep, (list, tuple)) and len(keep) == 4
        ok = ok and all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in keep)
        if not ok or not (keep[0] < keep[2] and keep[1] < keep[3]):
            raise ValueError(
                "grid_carry_label_keep_out must be null or (x0, y0, x1, y1) "
                "fractions of the tile in 0..1 with x0 < x1 and y0 < y1"
            )
        object.__setattr__(self, "grid_carry_label_keep_out", tuple(map(float, keep)))


def rgba_channels(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{name} must be 4 integers")
    if not all(isinstance(c, int) and 0 <= c <= 255 for c in value):
        raise ValueError(f"{name} channels must be integers in 0..255")
    return tuple(value)


def _ground(points_m: Array, z_m: float) -> Array:
    ground = np.array(points_m, dtype=float).reshape(-1, 3)
    ground[:, 2] = z_m
    return ground


def paint(
    tile: Image8,
    camera: Any,
    landing_m: Array,
    style: TracerSettings,
    show_ring: bool,
) -> Image8:
    """``tile`` with the ring at ``landing_m`` when ``show_ring``.

    The ring lies on the ground, where a far depth buffer is too coarse to
    test it, so the scene never hides it.
    """
    m, near = style.landing_marker, style.near_clip_m
    if m is None:
        return tile
    image = Image.fromarray(tile)
    pen = ImageDraw.Draw(image, "RGBA")
    centre = _ground(landing_m, m.ground_z_m)
    (u, v), depth = (a[0] for a in project(camera, centre, near))
    if show_ring and np.isfinite(u):
        r = max(float(m.min_radius_px), camera.matrix[1, 1] * m.radius_m / depth)
        box = [u - r, v - r * m.min_aspect, u + r, v + r * m.min_aspect]
        pen.ellipse(box, outline=m.color_rgba, width=m.line_width_px)
    return np.asarray(image, dtype=np.uint8)


def ground_track_points(path_m: Array, z_m: float) -> Array:
    """``path_m`` projected straight down onto the ground height ``z_m``.

    Raises ``ValueError`` for a path that is not finite.
    """
    path = np.asarray(path_m, dtype=float)
    if not np.isfinite(path).all():
        raise ValueError("path_m must be finite")
    return _ground(path, z_m)


def carry_text(marker: LandingMarkerSettings, carry_m: float) -> str:
    """The carry label text: the reported carry in the configured unit."""
    if not (math.isfinite(carry_m) and carry_m >= 0.0):
        raise ValueError(f"carry_m must be finite and not negative, got {carry_m}")
    carry = carry_m * marker.carry_label_scale
    return marker.carry_label_format.format(carry=carry, unit=marker.carry_label_unit)


def paint_track(
    tile: Image8,
    camera: Any,
    points_m: Array,
    style: TracerSettings,
    occlusion: flight_depth.Occlusion | None,
    landed: bool,
) -> Image8:
    """``tile`` with the ground track of ``points_m`` and, in flight, the drop line.

    ``points_m`` are the tracer's world points (last = the ball), so the track
    is revealed in step with the tracer. ``occlusion`` (scene depth of the tile,
    or ``None``) hides what is behind the scene, as for the tracer.
    """
    m, near = style.landing_marker, style.near_clip_m
    if m is None or not (m.ground_track or m.drop_line):
        return tile
    occlusion = occlusion if m.occlude_track else None
    image = Image.fromarray(tile)
    pen = ImageDraw.Draw(image, "RGBA")
    ground = ground_track_points(points_m, m.ground_z_m)
    if m.ground_track and len(ground) > 1:
        colour, width = m.ground_track_color_rgba, m.ground_track_width_px
        flight_depth.draw_line(pen, camera, ground, colour, width, near, occlusion)
    if m.drop_line and not landed:
        drop = np.vstack([np.asarray(points_m, dtype=float)[-1:], ground[-1:]])
        colour, width = m.drop_line_color_rgba, m.drop_line_width_px
        flight_depth.draw_line(pen, camera, drop, colour, width, near, occlusion)
    return np.asarray(image, dtype=np.uint8)


@lru_cache(maxsize=8)
def _font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size_px)


def paint_label(
    tile: Image8,
    camera: Any,
    landing_m: Array,
    style: TracerSettings,
    carry_m: float,
    grid: bool = False,
) -> Image8:
    """``tile`` with the carry label centred beneath the landing ring.

    The label is kept inside the tile. The scene never hides it, like the ring.
    In a ``grid`` tile it also keeps ``grid_carry_label_margin_px`` clear of the
    tile edges and is left out when it would cover the tile's central subject
    (``grid_carry_label_keep_out``): the HUD and other tiles still state it.
    """
    m, near = style.landing_marker, style.near_clip_m
    if m is None or not m.carry_label:
        return tile
    (u, v), depth = (
        a[0] for a in project(camera, _ground(landing_m, m.ground_z_m), near)
    )
    if not np.isfinite(u):
        return tile
    ring_r = max(float(m.min_radius_px), camera.matrix[1, 1] * m.radius_m / depth)
    image = Image.fromarray(tile)
    pen = ImageDraw.Draw(image, "RGBA")
    text, font, stroke = (
        carry_text(m, carry_m),
        _font(m.carry_label_size_px),
        m.carry_label_outline_px,
    )
    left, top, right, bottom = pen.textbbox(
        (0, 0), text, font=font, stroke_width=stroke
    )
    gap = m.grid_carry_label_margin_px if grid else 0
    width, height = right - left, bottom - top
    x = float(np.clip(u - width / 2.0, gap, max(gap, image.width - width - gap)))
    y = v + ring_r * m.min_aspect + m.carry_label_gap_px
    y = float(np.clip(y, gap, max(gap, image.height - height - gap)))
    if grid and _covers_subject(m, (x, y, x + width, y + height), image.size):
        return tile
    pen.text(
        (x - left, y - top),
        text,
        font=font,
        fill=m.carry_label_color_rgba,
        stroke_width=stroke,
        stroke_fill=m.carry_label_outline_rgba,
    )
    return np.asarray(image, dtype=np.uint8)


def _covers_subject(
    m: LandingMarkerSettings,
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> bool:
    """Whether the label ``box`` meets the keep-out rectangle of a tile of ``size``."""
    keep = m.grid_carry_label_keep_out
    if keep is None:
        return False
    kx0, ky0, kx1, ky1 = (
        keep[0] * size[0],
        keep[1] * size[1],
        keep[2] * size[0],
        keep[3] * size[1],
    )
    return box[0] < kx1 and kx0 < box[2] and box[1] < ky1 and ky0 < box[3]
