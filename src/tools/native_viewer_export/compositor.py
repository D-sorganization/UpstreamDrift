"""Labelled multi-view grid and HUD drawing for native exports (NV-6, #11679).

Text is drawn with Pillow: OpenCV ``putText`` advances differently for the
outline and the fill pass and garbles the labels. Pure numpy and Pillow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

Image8 = NDArray[np.uint8]
_FONTS: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
LABEL_SIZE_PX = 18
HUD_SIZE_PX = 16
_WHITE = (255, 255, 255)
_YELLOW = (255, 235, 90)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font = _FONTS.get(size)
    if font is None:
        font = _FONTS.setdefault(size, ImageFont.load_default(size=size))
    return font


def _check_image(img: object) -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")


@dataclass(frozen=True)
class GridGeometry:
    """Pixel layout of a tile grid."""

    rows: int
    cols: int
    tile_h: int
    tile_w: int

    @property
    def height(self) -> int:
        return self.rows * self.tile_h

    @property
    def width(self) -> int:
        return self.cols * self.tile_w

    def origin(self, index: int) -> tuple[int, int]:
        """``(x, y)`` of tile ``index`` in reading order."""
        if not 0 <= index < self.rows * self.cols:
            raise IndexError(f"tile index {index} outside the grid")
        row, col = divmod(index, self.cols)
        return col * self.tile_w, row * self.tile_h


def grid_geometry(
    n_tiles: int, tile_h: int, tile_w: int, cols: int = 2
) -> GridGeometry:
    """Geometry for ``n_tiles`` tiles laid out ``cols`` per row."""
    if n_tiles < 1 or cols < 1 or tile_h < 1 or tile_w < 1:
        raise ValueError("n_tiles, cols, tile_h and tile_w must be positive")
    return GridGeometry(math.ceil(n_tiles / cols), cols, tile_h, tile_w)


def compose_grid(tiles: Sequence[Image8], cols: int = 2) -> Image8:
    """Tile equally sized images in reading order (unused cells stay black)."""
    if len(tiles) == 0:
        raise ValueError("compose_grid needs at least one tile")
    for tile in tiles:
        _check_image(tile)
    if len({t.shape for t in tiles}) != 1:
        raise ValueError("all tiles must have the same shape")
    h, w = tiles[0].shape[:2]
    geo = grid_geometry(len(tiles), h, w, min(cols, len(tiles)))
    out = np.zeros((geo.height, geo.width, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        x, y = geo.origin(i)
        out[y : y + h, x : x + w] = tile
    return out


def draw_label(
    img: Image8,
    text: str,
    xy: tuple[int, int],
    *,
    size: int = LABEL_SIZE_PX,
    fill: tuple[int, int, int] = _WHITE,
) -> Image8:
    """Return a copy of ``img`` with outlined ``text`` at ``xy`` (Pillow)."""
    _check_image(img)
    out = img.copy()
    if not text:
        return out
    pil = Image.fromarray(out)
    ImageDraw.Draw(pil).text(
        xy, text, font=_font(size), fill=fill, stroke_width=2, stroke_fill=(0, 0, 0)
    )
    return np.asarray(pil).copy()


@dataclass(frozen=True)
class HudInfo:
    """Heads-up display content: time, engine, club and overlay legend."""

    time_s: float
    engine: str
    club: str
    legend: str
    speed: float | None = None
    ms_from_impact: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        if self.speed is not None and not (
            math.isfinite(self.speed) and self.speed > 0
        ):
            raise ValueError("speed must be positive and finite")
        if self.ms_from_impact is not None and not math.isfinite(self.ms_from_impact):
            raise ValueError("ms_from_impact must be finite")

    def lines(self) -> list[str]:
        """Text lines, the legend only when present.

        The first line carries the swing time, and when known the playback
        speed (``0.5x``) and the real time from impact (``-12.0 ms``).
        """
        head = f"{self.engine} | {self.club} | t = {self.time_s:.3f} s"
        if self.speed is not None:
            head += f" | {self.speed:g}x"
        if self.ms_from_impact is not None:
            head += f" | {self.ms_from_impact:+.1f} ms from impact"
        out = [head]
        if self.legend:
            out.append(self.legend)
        return out


def draw_hud(img: Image8, hud: HudInfo, *, size: int = HUD_SIZE_PX) -> Image8:
    """Draw the HUD lines along the bottom edge of ``img``."""
    _check_image(img)
    lines = hud.lines()
    pad = 6
    line_h = size + 4
    y = img.shape[0] - pad - line_h * len(lines)
    out = img
    for i, line in enumerate(lines):
        out = draw_label(
            out,
            line,
            (pad, y + i * line_h),
            size=size,
            fill=_WHITE if i == 0 else _YELLOW,
        )
    return out
