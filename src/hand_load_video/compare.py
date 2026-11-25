"""Baseline versus least-force frames on a ZTCF, ZVCF and Total grid."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

UNAVAILABLE = "UNAVAILABLE"
STRATEGIES = ("baseline", "least-force")
COMPONENTS = ("ztcf", "zvcf", "total")
BADGES = ("allocation-only", "forward-replayed")
OBJECTIVE = "J_F: least squared force; C: opposing-force cancellation"
_COLUMN = {"ztcf": "ZTCF", "zvcf": "ZVCF", "total": "Total"}
_GRID = tuple(
    (strategy, component) for strategy in STRATEGIES for component in COMPONENTS
)
_FONT = ImageFont.load_default()
_Box = tuple[int, int, int, int]


class PanelSource(Protocol):
    """Local frames. A later catalog adapter implements ``sequences``."""

    def sequences(self) -> tuple[PanelSequence, ...]:
        """One sequence per strategy and component."""


@dataclass(frozen=True)
class GridSpec:
    cell_w: int = 80
    cell_h: int = 36
    gutter: int = 4
    banner_h: int = 14
    footer_h: int = 16
    label_w: int = 100
    rows = STRATEGIES
    columns = COMPONENTS
    yellow, white = (240, 228, 66), (255, 255, 255)
    black, purple = (0, 0, 0), (204, 121, 167)
    component_rgb = {
        "ztcf": (0, 114, 178),
        "zvcf": (213, 94, 0),
        "total": (242, 242, 242),
    }
    badge_rgb = {"allocation-only": (230, 159, 0), "forward-replayed": (0, 158, 115)}

    def __post_init__(self) -> None:
        for name in ("cell_w", "cell_h", "gutter", "banner_h", "footer_h", "label_w"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive length")

    def _at(self, col: int, row: int) -> tuple[int, int]:
        x = self.label_w + self.gutter + col * (self.cell_w + self.gutter)
        step = self.banner_h + self.cell_h + self.gutter
        return x, 2 * self.banner_h + self.gutter + row * step

    @property
    def width(self) -> int:
        n = len(COMPONENTS)
        return self.label_w + n * self.cell_w + (n + 1) * self.gutter

    @property
    def height(self) -> int:
        band = self.gutter + self.banner_h + self.cell_h
        return 2 * self.banner_h + len(STRATEGIES) * band + self.gutter + self.footer_h

    def cell_box(self, row: int, col: int) -> _Box:
        x, y = self._at(col, row)
        return (x, y + self.banner_h, self.cell_w, self.cell_h)

    def header_box(self, col: int) -> _Box:
        return (self._at(col, 0)[0], self.banner_h, self.cell_w, self.banner_h)

    def badge_box(self, row: int) -> _Box:
        return (0, self._at(0, row)[1], self.label_w, self.banner_h)

    def text_origin(self, slot: str) -> tuple[int, int]:
        if slot == "objective":
            return (2, 1)
        if slot == "footer":
            return (2, self.height - self.footer_h + 1)
        raise ValueError(f"unknown text slot {slot!r}")

    def badge_text_origin(self, row: int) -> tuple[int, int]:
        return (2, self.badge_box(row)[1] + 1)

    def unavailable_origin(self, row: int, col: int) -> tuple[int, int]:
        x, y, _, _ = self.cell_box(row, col)
        return (x + 1, y + self.banner_h)

    def _mid_edge(self, box: _Box) -> tuple[int, int]:
        x, y, width, height = box
        return y + height // 2, x + width - 2

    def header_sample(self, col: int) -> tuple[int, int]:
        return self._mid_edge(self.header_box(col))

    def badge_sample(self, row: int) -> tuple[int, int]:
        return self._mid_edge(self.badge_box(row))

    def content_sample(self, row: int, col: int) -> tuple[int, int]:
        x, y, width, height = self.cell_box(row, col)
        return y + height - 3, x + width // 2


@dataclass(frozen=True)
class Provenance:
    source_id: str
    model_id: str
    git_sha: str

    def __post_init__(self) -> None:
        for name in ("source_id", "model_id", "git_sha"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be a non-empty string")

    def footer(self) -> str:
        return f"source={self.source_id} model={self.model_id} sha={self.git_sha}"


@dataclass(frozen=True)
class PanelSequence:
    strategy: str
    component: str
    times_s: tuple[float, ...]
    frames: tuple[np.ndarray | None, ...]
    badge: str
    units: str

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("strategy", self.strategy, STRATEGIES),
            ("component", self.component, COMPONENTS),
            ("badge", self.badge, BADGES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if not str(self.units).strip():
            raise ValueError("units must be a non-empty string")
        if len(self.times_s) != len(self.frames):
            raise ValueError("each frame needs one time")
        object.__setattr__(self, "times_s", _times(self.times_s, "frame times"))


@dataclass(frozen=True)
class Cell:
    strategy: str
    component: str
    time_s: float | None
    pixels: np.ndarray | None
    units: str
    badge: str
    missing: bool


@dataclass(frozen=True)
class SyncedFrame:
    time_s: float
    cells: tuple[Cell, ...]

    def cell(self, strategy: str, component: str) -> Cell:
        for item in self.cells:
            if (item.strategy, item.component) == (strategy, component):
                return item
        raise KeyError((strategy, component))


@dataclass(frozen=True)
class FrameStamp:
    objective: str
    footer: str
    badges: tuple[str, str]
    units: tuple[str, ...]
    missing: tuple[bool, ...]


def panel_from_arrays(
    strategy: str,
    component: str,
    times_s: Sequence[float],
    frames: Sequence[Any],
    *,
    badge: str,
    units: str,
) -> PanelSequence:
    return PanelSequence(
        strategy, component, tuple(times_s), tuple(map(_as_frame, frames)), badge, units
    )


def load_source(source: PanelSource) -> tuple[PanelSequence, ...]:
    panels = tuple(source.sequences())
    _index(panels)
    return panels


def align(
    panels: Sequence[PanelSequence], clock_s: Sequence[float], *, tolerance_s: float
) -> tuple[SyncedFrame, ...]:
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    indexed = _index(panels)
    clock = _times(clock_s, "clock times", nonempty=True)
    frames = []
    for time_s in clock:
        cells = tuple(
            _cell_at(indexed.get(key), *key, time_s, tolerance_s) for key in _GRID
        )
        frames.append(SyncedFrame(time_s, cells))
    return tuple(frames)


def compose(
    frame: SyncedFrame, provenance: Provenance, spec: GridSpec | None = None
) -> tuple[np.ndarray, FrameStamp]:
    layout = spec or GridSpec()
    badges = (_row_badge(frame, STRATEGIES[0]), _row_badge(frame, STRATEGIES[1]))
    units = tuple(frame.cell(*key).units for key in _GRID)
    missing = tuple(frame.cell(*key).missing for key in _GRID)
    stamp = FrameStamp(OBJECTIVE, provenance.footer(), badges, units, missing)
    canvas = Image.new("RGB", (layout.width, layout.height), layout.black)
    _paint(canvas, ImageDraw.Draw(canvas), frame, layout, stamp)
    return np.array(canvas), stamp


def _times(values, label, *, nonempty=False):
    times = tuple(float(item) for item in values)
    if (nonempty and not times) or any(not math.isfinite(item) for item in times):
        suffix = " and non-empty" if nonempty else ""
        raise ValueError(f"{label} must be finite{suffix}")
    if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
        raise ValueError(f"{label} must be strictly increasing")
    return times


def _as_frame(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.array(value, copy=True)
    shape = arr.shape
    bad = arr.dtype != np.uint8 or arr.ndim != 3 or shape[2] != 3 or min(shape[:2]) < 1
    if bad:
        raise ValueError("a frame must be uint8 HxWx3 or None")
    return arr


def _index(panels: Sequence[PanelSequence]) -> dict[tuple[str, str], PanelSequence]:
    if not panels:
        raise ValueError("a comparison needs at least one panel")
    indexed: dict[tuple[str, str], PanelSequence] = {}
    badges: dict[str, str] = {}
    for panel in panels:
        key = (panel.strategy, panel.component)
        if key in indexed:
            raise ValueError(f"duplicate panel {key}")
        prior = badges.get(panel.strategy)
        if prior not in (None, panel.badge):
            raise ValueError(f"badge mismatch for {panel.strategy}")
        badges[panel.strategy] = panel.badge
        indexed[key] = panel
    return indexed


def _cell_at(panel, strategy, component, time_s, tolerance_s) -> Cell:
    if panel is None:
        return Cell(strategy, component, None, None, UNAVAILABLE, UNAVAILABLE, True)
    best: int | None = None
    best_gap = tolerance_s
    for index, stamp in enumerate(panel.times_s):
        gap = abs(stamp - time_s)
        if gap <= tolerance_s and (best is None or gap < best_gap):
            best, best_gap = index, gap
    if best is None:
        return Cell(strategy, component, None, None, panel.units, panel.badge, True)
    pixels = panel.frames[best]
    when, units, badge = panel.times_s[best], panel.units, panel.badge
    return Cell(strategy, component, when, pixels, units, badge, pixels is None)


def _row_badge(frame: SyncedFrame, strategy: str) -> str:
    found = [frame.cell(strategy, name).badge for name in COMPONENTS]
    return next((badge for badge in found if badge != UNAVAILABLE), UNAVAILABLE)


def _text(image, xy, text, fill) -> None:
    mask = _FONT.getmask(text, mode="1")
    glyph = Image.frombytes("L", mask.size, bytes(mask))
    image.paste(Image.new("RGB", mask.size, fill), xy, glyph)


def _paint(canvas, draw, frame, spec, stamp) -> None:
    _text(canvas, spec.text_origin("objective"), stamp.objective, spec.yellow)
    for col, name in enumerate(COMPONENTS):
        x, y, width, height = spec.header_box(col)
        ink = spec.black if name == "total" else spec.yellow
        draw.rectangle(
            [x, y, x + width - 1, y + height - 1], fill=spec.component_rgb[name]
        )
        _text(canvas, (x + 2, y + 1), _COLUMN[name], ink)
    for row, strategy in enumerate(STRATEGIES):
        badge = stamp.badges[row]
        x, y, width, height = spec.badge_box(row)
        color = spec.badge_rgb.get(badge, spec.purple)
        draw.rectangle([x, y, x + width - 1, y + height - 1], fill=color)
        _text(canvas, spec.badge_text_origin(row), badge, spec.black)
        _text(canvas, (2, spec.cell_box(row, 0)[1] + 1), strategy, spec.yellow)
        for col, name in enumerate(COMPONENTS):
            _paint_cell(canvas, frame.cell(strategy, name), spec, row, col)
    _text(canvas, spec.text_origin("footer"), stamp.footer, spec.white)


def _paint_cell(canvas, cell, spec, row, col) -> None:
    x, y, width, height = spec.cell_box(row, col)
    if not cell.missing and cell.pixels is not None:
        shot = Image.fromarray(cell.pixels, mode="RGB")
        tile = shot.resize((width, height), Image.Resampling.NEAREST)
    else:
        raw = np.zeros((height, width, 3), dtype=np.uint8)
        yy, xx = np.indices((height, width))
        raw[((xx + yy) // 3) % 2 == 0] = spec.purple
        tile = Image.fromarray(raw, mode="RGB")
    canvas.paste(tile, (x, y))
    _text(canvas, (x + 1, y + 1), cell.units, spec.yellow)
    if cell.missing or cell.pixels is None:
        _text(canvas, spec.unavailable_origin(row, col), UNAVAILABLE, spec.yellow)
