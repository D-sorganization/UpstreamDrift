"""Labelled multi-view grid and HUD drawing for native exports (NV-6, #11679).

Text is drawn with Pillow: OpenCV ``putText`` advances differently for the
outline and the fill pass and garbles the labels. Pure numpy and Pillow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import textwrap
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

Image8 = NDArray[np.uint8]
_FONTS: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
LABEL_SIZE_PX = 18
HUD_SIZE_PX = 16
_WHITE = (255, 255, 255)
_YELLOW = (255, 235, 90)


LAYOUT_PATH = Path(__file__).with_name("hud_layout_settings.json")


@dataclass(frozen=True)
class HudLayout:
    """Validated ``hud_layout_settings.json`` (``ValueError`` when invalid)."""

    stroke_px: int
    line_gap_px: int
    edge_margin_px: int
    label_origin_px: tuple[int, int]
    panel_gap_px: int

    def __post_init__(self) -> None:
        values = [self.stroke_px, self.line_gap_px, self.edge_margin_px]
        values += [*self.label_origin_px, self.panel_gap_px]
        if not all(isinstance(v, int) and v >= 0 for v in values):
            raise ValueError("HUD layout values must be integers, not negative")


def load_hud_layout(path: Path | None = None) -> HudLayout:
    """Read the HUD layout settings (``KeyError`` for a missing key)."""
    raw = json.loads((path or LAYOUT_PATH).read_text(encoding="utf-8"))
    keys = ("stroke_px", "line_gap_px", "edge_margin_px", "panel_gap_px")
    origin = tuple(raw["label_origin_px"])
    return HudLayout(
        **{k: raw[k] for k in keys},
        label_origin_px=origin,  # type: ignore[arg-type]
    )


HUD_LAYOUT = load_hud_layout()
#: separator of the entries of one HUD line; long lines wrap at it
HUD_SEPARATOR = " | "


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


def _line_extent(size: int) -> tuple[int, int]:
    """Top and bottom of an outlined text line below its origin (``Ag|``)."""
    box = _font(size).getbbox("Ag|", stroke_width=HUD_LAYOUT.stroke_px)
    return math.floor(box[1]), math.ceil(box[3])


def text_box(text: str, xy: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` of outlined ``text`` drawn at ``xy``.

    The height is that of the tallest and lowest glyphs (``Ag|``), so every line
    of one size is as tall as the others and boxes can be stacked.
    """
    top, bottom = _line_extent(size)
    box = _font(size).getbbox(text, stroke_width=HUD_LAYOUT.stroke_px)
    x0, x1 = math.floor(xy[0] + box[0]), math.ceil(xy[0] + box[2])
    return (x0, xy[1] + top, x1, xy[1] + bottom)


def view_label_box(shape: tuple[int, ...], text: str) -> tuple[int, int, int, int]:
    """Pixel box of the view label ``text`` in a frame of ``shape``."""
    x0, y0, x1, y1 = text_box(text, HUD_LAYOUT.label_origin_px, LABEL_SIZE_PX)
    return (x0, y0, min(x1, int(shape[1])), y1)


def draw_view_label(img: Image8, text: str) -> Image8:
    """Copy of ``img`` with the view label ``text`` at its layout position."""
    return draw_label(img, text, HUD_LAYOUT.label_origin_px)


def _hud_origins(
    shape: tuple[int, ...], count: int, size: int
) -> list[tuple[int, int]]:
    """Text origins of ``count`` HUD lines stacked up from the bottom edge."""
    top, bottom = _line_extent(size)
    pitch = bottom - top + HUD_LAYOUT.line_gap_px
    last = int(shape[0]) - HUD_LAYOUT.edge_margin_px - bottom
    x = HUD_LAYOUT.edge_margin_px
    return [(x, last - pitch * (count - 1 - i)) for i in range(count)]


def hud_line_boxes(
    shape: tuple[int, ...], lines: Sequence[str], size: int
) -> list[tuple[int, int, int, int]]:
    """Boxes of HUD ``lines``, top line first, as :func:`draw_hud` draws them.

    Postcondition: consecutive boxes are ``line_gap_px`` apart, so none
    overlaps another.
    """
    origins = _hud_origins(shape, len(lines), size)
    return [text_box(t, xy, size) for t, xy in zip(lines, origins, strict=True)]


def _fits(text: str, width_px: int, size: int) -> bool:
    margin = HUD_LAYOUT.edge_margin_px
    return text_box(text, (margin, 0), size)[2] <= width_px - margin


def _fill(parts: list[str], sep: str, width_px: int, size: int) -> list[str]:
    """Greedily join ``parts`` with ``sep`` into rows that fit ``width_px``."""
    rows: list[str] = []
    row = ""
    for part in parts:
        longer = part if not row else row + sep + part
        if row and not _fits(longer, width_px, size):
            rows.append(row)
            row = part
        else:
            row = longer
    return [*rows, row]


def wrap_hud_line(text: str, width_px: int, size: int) -> list[str]:
    """``text`` as rows that fit a frame ``width_px`` wide, split at ``HUD_SEPARATOR``.

    Rows are filled in order and joined by the separator they were split at.
    An entry wider than the frame on its own is broken at spaces instead of
    running off the edge; a single word wider than the frame keeps its own
    row. Postcondition: the rows hold the words of ``text`` in order.
    """
    rows: list[str] = []
    for row in _fill(text.split(HUD_SEPARATOR), HUD_SEPARATOR, width_px, size):
        if _fits(row, width_px, size):
            rows.append(row)
        else:
            rows.extend(_fill(row.split(" "), " ", width_px, size))
    return rows


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
        xy,
        text,
        font=_font(size),
        fill=fill,
        stroke_width=HUD_LAYOUT.stroke_px,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(pil).copy()


_EVENT_DISPLAY_NAMES: dict[str, str] = {
    "address": "address",
    "pelvis_reversal": "pelvis reversal",
    "top": "top",
    "impact": "impact",
    "finish": "finish",
}


def event_caption_for_time(
    time_s: float,
    events: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the active anatomical event caption name at ``time_s`` (#13175).

    Accepts either an ``anatomical_event_table`` document (with an ``"events"``
    mapping) or a flat mapping of event names to ``{"time_s": ...}`` / float
    timestamps. Raises ``ValueError`` when ``time_s`` is not finite.
    """
    if not math.isfinite(time_s):
        raise ValueError("time_s must be finite")
    if events is None or not isinstance(events, Mapping):
        return "address"
    raw_events = events.get("events", events)
    if not isinstance(raw_events, Mapping) or not raw_events:
        return "address"
    parsed: list[tuple[float, int, str]] = []
    for idx, (name, entry) in enumerate(raw_events.items()):
        t_val: float | None = None
        if isinstance(entry, Mapping) and "time_s" in entry:
            t_val = float(entry["time_s"])
        elif isinstance(entry, (int, float)):
            t_val = float(entry)
        if t_val is not None and math.isfinite(t_val):
            parsed.append((t_val, idx, str(name)))
    if not parsed:
        return "address"
    parsed.sort(key=lambda item: (item[0], item[1]))
    chosen = parsed[0][2]
    for event_time_s, _idx, name in parsed:
        if event_time_s <= time_s + 1e-9:
            chosen = name
        else:
            break
    return _EVENT_DISPLAY_NAMES.get(chosen, chosen.replace("_", " "))


@dataclass(frozen=True)
class HudInfo:
    """Heads-up display content: time, engine, club and overlay legend."""

    time_s: float
    engine: str
    club: str
    legend: str
    speed: float | None = None
    ms_from_impact: float | None = None
    #: tension/compression shading key line (its own line when non-empty)
    shading_legend: str = ""
    #: optional anatomical swing event caption (HRG-E4, #13175); off by default
    event_caption: str = ""
    #: optional capability/mobility HUD lines (HCM-3, #14004, #14038)
    extra_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        if self.speed is not None and not (
            math.isfinite(self.speed) and self.speed > 0
        ):
            raise ValueError("speed must be positive and finite")
        if self.ms_from_impact is not None and not math.isfinite(self.ms_from_impact):
            raise ValueError("ms_from_impact must be finite")
        if not isinstance(self.event_caption, str):
            raise TypeError("event_caption must be a str")

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
        if self.shading_legend:
            out.append(self.shading_legend)
        if self.event_caption:
            caption = (
                self.event_caption
                if self.event_caption.startswith("Event:")
                else f"Event: {self.event_caption}"
            )
            out.append(caption)
        out.extend(self.extra_lines)
        return out


_CARD_POSITIONS = ("right", "left")
_CARD_TITLE = "Impact Parameters"
_CARD_GREY = (190, 190, 190)
_CARD_ALPHA = 0.62
_CARD_REASON_CHARS = 46


def _short(reason: str) -> str:
    """``reason`` limited to the panel width, marked with ``...`` when cut."""
    if len(reason) <= _CARD_REASON_CHARS:
        return reason
    return reason[: _CARD_REASON_CHARS - 3].rstrip() + "..."


def _wrap_reason(reason: str, indent: str = "") -> list[str]:
    """``reason`` as lines of at most ``_CARD_REASON_CHARS`` characters.

    Lines break between words (a word wider than a line keeps its own line),
    each led by ``indent``; the text is never cut or abbreviated.
    """
    width = max(1, _CARD_REASON_CHARS - len(indent))
    words = textwrap.wrap(
        reason, width=width, break_long_words=False, break_on_hyphens=False
    )
    return [indent + line for line in words or [""]]


def _card_value_text(row: Any) -> str:
    """Value text of a card row; ``None`` is shown as unavailable, never as 0."""
    if row.value is None:
        return "unavailable"
    unit = f" {row.unit}" if row.unit else ""
    if row.unit == "m":
        text = f"{row.value:+.3f}{unit}"
    elif row.unit in ("deg", "mm"):
        text = f"{row.value:+.1f}{unit}"
    elif row.unit:
        text = f"{row.value:.1f}{unit}"
    else:
        text = f"{row.value:.2f}"
    return f"{text} ({row.note})" if row.note else text


def _card_lines(card: Any) -> list[tuple[str, str, bool]]:
    """``(label, value, unavailable)`` lines of ``card`` (reasons as sub-lines)."""
    if not card.available or not card.rows:
        reason = str(card.reason or "no rows")
        return [("Unavailable", "", True)] + [
            (line, "", True) for line in _wrap_reason(reason)
        ]
    lines: list[tuple[str, str, bool]] = []
    for row in card.rows:
        lines.append((row.label, _card_value_text(row), row.value is None))
        if row.value is None and row.reason:
            lines.extend(
                (line, "", True) for line in _wrap_reason(str(row.reason), "  ")
            )
    return lines


#: a panel line's value: a text, or the cells of a table row (right-aligned columns)
Cells = str | Sequence[str]


def _cells(value: Cells) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def _column_widths(font: Any, lines: Sequence[tuple[str, Cells, bool]]) -> list[float]:
    rows = [_cells(v) for _, v, _ in lines]
    return [
        max(font.getlength(r[j]) for r in rows if len(r) > j)
        for j in range(max(map(len, rows), default=0))
    ]


def text_panel_box(
    shape: tuple[int, ...],
    title: str,
    lines: Sequence[tuple[str, Cells, bool]],
    position: str = "right",
    size: int | None = None,
) -> tuple[tuple[int, int, int, int], int]:
    """Pixel box and font size of a panel holding ``title`` and ``lines``.

    The box is as wide as the widest ``(label, value, dimmed)`` line, clipped
    to the frame; ``size`` defaults to the frame height over 45 (at least 11).
    """
    if position not in _CARD_POSITIONS:
        raise ValueError(f"position must be one of {_CARD_POSITIONS}, got {position!r}")
    height, width = int(shape[0]), int(shape[1])
    size = max(11, height // 45) if size is None else size
    font = _font(size)
    pad, line_h = size // 2 + 2, size + 6
    cols = _column_widths(font, lines)
    table_w = sum(cols) + pad * (len(cols) - 1)
    wide = [
        table_w if not isinstance(v, str) else font.getlength(v) for _, v, _ in lines
    ]
    text_w = max(
        [font.getlength(title)]
        + [
            font.getlength(ln[0]) + w + 2 * pad
            for ln, w in zip(lines, wide, strict=True)
        ]
    )
    box_w = min(width, int(text_w + 2 * pad))
    box_h = min(height, int(2 * pad + line_h * (len(lines) + 1)))
    margin = max(6, width // 80)
    x0 = width - margin - box_w if position == "right" else margin
    y0 = max(margin, view_label_box(shape, "")[3] + HUD_LAYOUT.panel_gap_px)
    return (x0, y0, x0 + box_w, min(height, y0 + box_h)), size


class GridLayout(NamedTuple):
    """Placement of a text panel in the composed 2x2 grid (see ``grid_anchor``,
    ``grid_font_height_divisor`` and ``grid_min_font_px`` in
    ``impact_factors_settings.json``)."""

    anchor: str
    font_height_divisor: int
    min_font_px: int


def grid_text_panel_box(
    shape: tuple[int, ...],
    title: str,
    lines: Sequence[tuple[str, Cells, bool]],
    layout: GridLayout,
) -> tuple[tuple[int, int, int, int], int]:
    """Pixel box and font size of a panel in the ``layout.anchor`` tile corner.

    The font is the larger of ``layout.min_font_px`` and the frame height over
    ``layout.font_height_divisor``; it shrinks to the minimum until the box
    fits inside its tile, and the box is clipped to that tile.
    """
    height, width = int(shape[0]), int(shape[1])
    tile_w, tile_h = width // 2, height // 2
    top = max(layout.min_font_px, height // layout.font_height_divisor)
    for size in range(top, layout.min_font_px - 1, -1):
        x0, y0, x1, y1 = text_panel_box(shape, title, lines, "right", size)[0]
        margin = width - x1
        if x1 - x0 <= tile_w - 2 * margin and y1 - y0 <= tile_h - 2 * margin:
            break
    box_w = min(x1 - x0, tile_w - 2 * margin)
    box_h = min(y1 - y0, tile_h - 2 * margin)
    left = width - margin - box_w if layout.anchor.endswith("right") else margin
    up = height - margin - box_h if layout.anchor.startswith("bottom") else y0
    return (left, up, left + box_w, up + box_h), size


def _card_metrics(
    shape: tuple[int, ...], card: Any, position: str, grid: GridLayout | None = None
) -> tuple[tuple[int, int, int, int], int, list[tuple[str, str, bool]]]:
    lines = _card_lines(card)
    if grid is not None:
        return (*grid_text_panel_box(shape, _CARD_TITLE, lines, grid), lines)
    return (*text_panel_box(shape, _CARD_TITLE, lines, position), lines)


def impact_card_bbox(
    shape: tuple[int, ...],
    card: Any,
    *,
    position: str = "right",
    grid: GridLayout | None = None,
) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` the card panel occupies in a frame of ``shape``.

    ``grid`` places it in a tile corner of a composed 2x2 frame instead."""
    return _card_metrics(shape, card, position, grid)[0]


def draw_impact_card(
    img: Image8,
    card: Any,
    *,
    position: str = "right",
    grid: GridLayout | None = None,
) -> Image8:
    """Return a copy of ``img`` with the GCV-17 impact card panel drawn on it.

    ``card`` is the shared :class:`ImpactCard`; rows whose ``value`` is ``None``
    are listed as ``unavailable`` with their reason, never as zero. The panel is
    semi-transparent and sits on the right (or left) edge so the golfer stays
    visible; only pixels inside :func:`impact_card_bbox` change.
    """
    _check_image(img)
    if not hasattr(card, "rows") or not hasattr(card, "available"):
        raise TypeError("card must be an ImpactCard")
    box, size, lines = _card_metrics(img.shape, card, position, grid)
    return draw_text_panel(img, box, size, _CARD_TITLE, lines)


def draw_text_panel(
    img: Image8,
    box: tuple[int, int, int, int],
    size: int,
    title: str,
    lines: Sequence[tuple[str, Cells, bool]],
    alpha: float = _CARD_ALPHA,
    label_colours: Sequence[tuple[int, int, int] | None] = (),
) -> Image8:
    """Copy of ``img`` with a darkened panel in ``box``: ``title``, then one
    ``(label, value, dimmed)`` line each. Only pixels inside ``box`` change.
    A value of cells is drawn as columns; ``label_colours`` colour the labels."""
    _check_image(img)
    (x0, y0, x1, y1), out = box, img.copy()
    region = out[y0:y1, x0:x1].astype(np.float32) * (1.0 - alpha)
    pil = Image.fromarray(region.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    font = _font(size)
    pad, line_h = size // 2 + 2, size + 6
    draw.text((pad, pad), title, font=font, fill=_YELLOW)
    rights, edge = [], float(x1 - x0 - pad)
    for width in reversed(_column_widths(font, lines)):
        rights.append(edge)
        edge -= width + pad
    for i, (label, value, unavailable) in enumerate(lines, start=1):
        y = pad + i * line_h
        colour = _CARD_GREY if unavailable else _WHITE
        colour = next(iter(label_colours[i - 1 : i]), None) or colour
        draw.text((pad, y), label, font=font, fill=colour)
        for cell, right in zip(reversed(_cells(value)), rights, strict=False):
            draw.text(
                (right - font.getlength(cell), y),
                cell,
                font=font,
                fill=_YELLOW if unavailable else _WHITE,
            )
    out[y0:y1, x0:x1] = np.asarray(pil)
    return out


def draw_hud(img: Image8, hud: HudInfo, *, size: int = HUD_SIZE_PX) -> Image8:
    """Draw the HUD lines along the bottom edge of ``img``.

    A line wider than the frame continues on the next row (:func:`wrap_hud_line`).
    """
    _check_image(img)
    width = int(img.shape[1])
    rows = [
        (row, i == 0)
        for i, line in enumerate(hud.lines())
        for row in wrap_hud_line(line, width, size)
    ]
    out = img
    origins = _hud_origins(img.shape, len(rows), size)
    for (row, head), xy in zip(rows, origins, strict=True):
        out = draw_label(out, row, xy, size=size, fill=_WHITE if head else _YELLOW)
    return out
