"""Landing marker and ground track of the shot tracer (#15150).

A ground ring at the landing point puts a far landing on the turf instead of
the horizon. The ring has a minimum screen size: a marker
size, not the landing area. Options: ``flight_tracer_settings.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

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

    def __post_init__(self) -> None:
        if self.when not in WHEN:
            raise ValueError(f"when must be one of {WHEN}, got {self.when!r}")
        for name in ("radius_m", "min_aspect"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("min_radius_px", "line_width_px"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


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
