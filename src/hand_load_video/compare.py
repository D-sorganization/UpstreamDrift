"""Baseline versus least-force frames on one clock.

Rows are the baseline null-space allocation and the least-force strategy.
Columns are ZTCF, ZVCF and Total. ``J_F`` is least squared force and ``C`` is
opposing-force cancellation. Each frame carries Okabe-Ito case colors, units,
that objective, a provenance footer and an ``allocation-only`` or
``forward-replayed`` badge. Missing panels are painted ``UNAVAILABLE``; a real
zero-force frame stays dark. Callers pass arrays through :class:`PanelSource`.
The swing-report catalog is not imported. The seven phase ids match that
catalog's phase settings; a phase outside tolerance is reported, not snapped.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

UNAVAILABLE = "UNAVAILABLE"
STRATEGIES = ("baseline", "least-force")
COMPONENTS = ("ztcf", "zvcf", "total")
BADGES = ("allocation-only", "forward-replayed")
OBJECTIVE = "J_F: least squared force; C: opposing-force cancellation"
PHASES = (
    "address",
    "shaft_parallel_back",
    "top",
    "lead_arm_parallel",
    "shaft_parallel_down",
    "impact",
    "follow_through",
)
PHASE_LABEL = {
    "address": "Address",
    "shaft_parallel_back": "Shaft Parallel (Backswing)",
    "top": "Top of Backswing",
    "lead_arm_parallel": "Lead Arm Parallel (Downswing)",
    "shaft_parallel_down": "Shaft Parallel (Downswing)",
    "impact": "Impact",
    "follow_through": "Follow-Through (Shaft Parallel)",
}
_COLUMN = {"ztcf": "ZTCF", "zvcf": "ZVCF", "total": "Total"}
_GRID = tuple((row, col) for row in STRATEGIES for col in COMPONENTS)
_FONT = ImageFont.load_default()
_Box = tuple[int, int, int, int]


class VideoWriterUnavailable(RuntimeError):
    """imageio's ffmpeg writer is not installed."""


class PanelSource(Protocol):
    """Local frames. A later catalog adapter implements ``sequences``."""

    def sequences(self) -> tuple[PanelSequence, ...]:
        """One sequence per strategy and component."""


@dataclass(frozen=True)
class GridSpec:
    """Pixel geometry. Every length is a setting."""

    cell_w: int = 80
    cell_h: int = 36
    gutter: int = 4
    banner_h: int = 14
    footer_h: int = 16
    label_w: int = 100
    yellow = (240, 228, 66)
    white = (255, 255, 255)
    black = (0, 0, 0)
    purple = (204, 121, 167)
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

    @property
    def rows(self) -> tuple[str, ...]:
        return STRATEGIES

    @property
    def columns(self) -> tuple[str, ...]:
        return COMPONENTS

    @property
    def width(self) -> int:
        cols = len(COMPONENTS)
        return self.label_w + cols * self.cell_w + (cols + 1) * self.gutter

    @property
    def height(self) -> int:
        row = self.gutter + self.banner_h + self.cell_h
        return 2 * self.banner_h + len(STRATEGIES) * row + self.gutter + self.footer_h

    def _x(self, col: int) -> int:
        return self.label_w + self.gutter + col * (self.cell_w + self.gutter)

    def _row_y(self, row: int) -> int:
        step = self.banner_h + self.cell_h + self.gutter
        return 2 * self.banner_h + self.gutter + row * step

    def cell_box(self, row: int, col: int) -> _Box:
        """``(x, y, w, h)`` of the picture, below its row banner."""
        return (
            self._x(col),
            self._row_y(row) + self.banner_h,
            self.cell_w,
            self.cell_h,
        )

    def header_box(self, col: int) -> _Box:
        return (self._x(col), self.banner_h, self.cell_w, self.banner_h)

    def badge_box(self, row: int) -> _Box:
        return (0, self._row_y(row), self.label_w, self.banner_h)

    def text_origin(self, slot: str) -> tuple[int, int]:
        if slot == "objective":
            return (2, 1)
        if slot == "footer":
            return (2, self.height - self.footer_h + 1)
        raise ValueError(f"unknown text slot {slot!r}")

    def badge_text_origin(self, row: int) -> tuple[int, int]:
        x, y, _, _ = self.badge_box(row)
        return (x + 2, y + 1)

    def unavailable_origin(self, row: int, col: int) -> tuple[int, int]:
        x, y, _, _ = self.cell_box(row, col)
        return (x + 1, y + self.banner_h)

    def header_sample(self, col: int) -> tuple[int, int]:
        x, y, width, height = self.header_box(col)
        return y + height // 2, x + width - 2

    def badge_sample(self, row: int) -> tuple[int, int]:
        x, y, width, height = self.badge_box(row)
        return y + height // 2, x + width - 2

    def content_sample(self, row: int, col: int) -> tuple[int, int]:
        x, y, width, height = self.cell_box(row, col)
        return y + height - 3, x + width // 2


@dataclass(frozen=True)
class Provenance:
    """Footer identity of the run that produced the frames."""

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
    """One strategy and component, in time order."""

    strategy: str
    component: str
    times_s: tuple[float, ...]
    frames: tuple[np.ndarray | None, ...]
    badge: str
    units: str

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.component not in COMPONENTS:
            raise ValueError(f"component must be one of {COMPONENTS}")
        if self.badge not in BADGES:
            raise ValueError(f"badge must be one of {BADGES}")
        if not str(self.units).strip():
            raise ValueError("units must be a non-empty string")
        if len(self.times_s) != len(self.frames):
            raise ValueError("each frame needs one time")
        if any(not math.isfinite(t) for t in self.times_s):
            raise ValueError("frame times must be finite")
        pairs = zip(self.times_s, self.times_s[1:], strict=False)
        if any(later <= earlier for earlier, later in pairs):
            raise ValueError("frame times must be strictly increasing")


@dataclass(frozen=True)
class Cell:
    """One grid cell at one clock time."""

    strategy: str
    component: str
    time_s: float | None
    pixels: np.ndarray | None
    units: str
    badge: str
    missing: bool


@dataclass(frozen=True)
class SyncedFrame:
    """Every panel at one shared clock time."""

    time_s: float
    cells: tuple[Cell, ...]

    def cell(self, strategy: str, component: str) -> Cell:
        for item in self.cells:
            if item.strategy == strategy and item.component == component:
                return item
        raise KeyError((strategy, component))


@dataclass(frozen=True)
class PhasePick:
    """A phase and the synced frame at its time, when one is inside tolerance."""

    name: str
    time_s: float | None
    frame: SyncedFrame | None


@dataclass(frozen=True)
class FrameStamp:
    """Words painted on one comparison frame."""

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
    """Copy uint8 HxWx3 frames. ``None`` means that instant is unavailable."""
    copied = tuple(_as_frame(frame) for frame in frames)
    return PanelSequence(
        strategy, component, tuple(float(t) for t in times_s), copied, badge, units
    )


def load_source(source: PanelSource) -> tuple[PanelSequence, ...]:
    """Read a :class:`PanelSource`. Post: no duplicate strategy/component pair."""
    panels = tuple(source.sequences())
    _index(panels)
    return panels


def align(
    panels: Sequence[PanelSequence],
    clock_s: Sequence[float],
    *,
    tolerance_s: float,
) -> tuple[SyncedFrame, ...]:
    """Place every panel on ``clock_s`` by time, never by stored index.

    Post: synced times equal the clock, and a present cell is within
    ``tolerance_s`` of that time. Further frames and missing channels are
    unavailable.
    """
    _tolerance(tolerance_s)
    indexed = _index(panels)
    return tuple(_synced(indexed, t, tolerance_s) for t in _clock(clock_s))


def select_phases(
    frames: Sequence[SyncedFrame],
    phase_times: Mapping[str, float | None],
    *,
    tolerance_s: float,
) -> tuple[PhasePick, ...]:
    """One pick per swing phase. Omitted, ``None``, or out-of-tolerance times stay missing."""
    _tolerance(tolerance_s)
    unknown = sorted(set(phase_times) - set(PHASES))
    if unknown:
        raise ValueError(f"unknown swing phase {unknown}")
    return tuple(
        _pick(name, phase_times.get(name), frames, tolerance_s) for name in PHASES
    )


def compose(
    frame: SyncedFrame, provenance: Provenance, spec: GridSpec | None = None
) -> tuple[np.ndarray, FrameStamp]:
    """Paint ``frame``. Post: the stamp lists the words drawn on the image."""
    layout = spec or GridSpec()
    stamp = _stamp(frame, provenance)
    canvas = Image.new("RGB", (layout.width, layout.height), layout.black)
    _paint(canvas, ImageDraw.Draw(canvas), frame, layout, stamp)
    return np.array(canvas), stamp


def phase_sheets(
    picks: Sequence[PhasePick], provenance: Provenance, spec: GridSpec | None = None
) -> dict[str, np.ndarray]:
    """One sheet per swing phase. A missing phase is titled unavailable."""
    layout = spec or GridSpec()
    by_name = {pick.name: pick for pick in picks}
    return {
        name: _sheet(by_name.get(name), name, provenance, layout) for name in PHASES
    }


def stack_sheets(sheets: Sequence[np.ndarray], *, gutter: int = 2) -> np.ndarray:
    """Stack phase sheets top to bottom with a ``gutter``-pixel gap."""
    if not sheets or gutter < 0:
        raise ValueError("sheets must be non-empty and gutter non-negative")
    gap = np.zeros((gutter, sheets[0].shape[1], 3), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for index, sheet in enumerate(sheets):
        if index:
            parts.append(gap)
        parts.append(np.asarray(sheet))
    return np.vstack(parts)


def write_video(frames: Sequence[np.ndarray], path: str | Path, *, fps: float) -> None:
    """Write ``frames`` with imageio's ffmpeg writer, when that extra is installed."""
    if not frames:
        raise ValueError("no frames to write")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    imageio = _load_imageio()
    try:
        writer = imageio.get_writer(
            Path(path), fps=fps, codec="libx264", format="FFMPEG"
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise VideoWriterUnavailable("ffmpeg writer is unavailable") from exc
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def _load_imageio() -> Any:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise VideoWriterUnavailable("imageio is not installed") from exc
    return imageio


def _as_frame(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    shape = arr.shape
    if arr.dtype != np.uint8 or arr.ndim != 3 or shape[2] != 3 or min(shape[:2]) < 1:
        raise ValueError("a frame must be uint8 HxWx3 or None")
    return np.array(arr, copy=True)


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
        if prior is not None and prior != panel.badge:
            raise ValueError(
                f"{panel.strategy} badge {prior!r} disagrees with {panel.badge!r}"
            )
        badges[panel.strategy] = panel.badge
        indexed[key] = panel
    return indexed


def _tolerance(tolerance_s: float) -> None:
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")


def _clock(clock_s: Sequence[float]) -> tuple[float, ...]:
    clock = tuple(float(t) for t in clock_s)
    if not clock or any(not math.isfinite(t) for t in clock):
        raise ValueError("clock times must be finite and non-empty")
    pairs = zip(clock, clock[1:], strict=False)
    if any(later <= earlier for earlier, later in pairs):
        raise ValueError("clock times must be strictly increasing")
    return clock


def _nearest(times: Sequence[float], time_s: float, tolerance_s: float) -> int | None:
    best: int | None = None
    best_gap = tolerance_s
    for index, stamp in enumerate(times):
        gap = abs(stamp - time_s)
        if gap <= tolerance_s and (best is None or gap < best_gap):
            best, best_gap = index, gap
    return best


def _cell_at(
    panel: PanelSequence | None,
    strategy: str,
    component: str,
    time_s: float,
    tolerance_s: float,
) -> Cell:
    if panel is None:
        return Cell(strategy, component, None, None, UNAVAILABLE, UNAVAILABLE, True)
    index = _nearest(panel.times_s, time_s, tolerance_s)
    if index is None:
        return Cell(strategy, component, None, None, panel.units, panel.badge, True)
    pixels = panel.frames[index]
    return Cell(
        strategy,
        component,
        panel.times_s[index],
        pixels,
        panel.units,
        panel.badge,
        pixels is None,
    )


def _synced(
    indexed: Mapping[tuple[str, str], PanelSequence], time_s: float, tolerance_s: float
) -> SyncedFrame:
    cells = tuple(
        _cell_at(indexed.get(key), key[0], key[1], time_s, tolerance_s) for key in _GRID
    )
    return SyncedFrame(time_s, cells)


def _pick(
    name: str, raw: float | None, frames: Sequence[SyncedFrame], tolerance_s: float
) -> PhasePick:
    if raw is None:
        return PhasePick(name, None, None)
    time_s = float(raw)
    if not math.isfinite(time_s):
        raise ValueError(f"phase {name} time must be finite")
    chosen: SyncedFrame | None = None
    gap = tolerance_s
    for frame in frames:
        delta = abs(frame.time_s - time_s)
        if delta <= tolerance_s and (chosen is None or delta < gap):
            chosen, gap = frame, delta
    return PhasePick(name, time_s, chosen)


def _row_badge(frame: SyncedFrame, strategy: str) -> str:
    for component in COMPONENTS:
        badge = frame.cell(strategy, component).badge
        if badge != UNAVAILABLE:
            return badge
    return UNAVAILABLE


def _stamp(frame: SyncedFrame, provenance: Provenance) -> FrameStamp:
    badges = (_row_badge(frame, STRATEGIES[0]), _row_badge(frame, STRATEGIES[1]))
    units = tuple(frame.cell(*key).units for key in _GRID)
    missing = tuple(frame.cell(*key).missing for key in _GRID)
    return FrameStamp(OBJECTIVE, provenance.footer(), badges, units, missing)


def _text(
    image: Image.Image, xy: tuple[int, int], text: str, fill: tuple[int, int, int]
) -> None:
    """Stamp ``text`` in exact ``fill`` pixels, with no antialiased mix."""
    mask = _FONT.getmask(text, mode="1")
    image.paste(
        Image.new("RGB", mask.size, fill),
        xy,
        Image.frombytes("L", mask.size, bytes(mask)),
    )


def _rect(box: _Box) -> list[int]:
    x, y, width, height = box
    return [x, y, x + width - 1, y + height - 1]


def _picture(spec: GridSpec, cell: Cell, width: int, height: int) -> Image.Image:
    if cell.missing or cell.pixels is None:
        tile = np.zeros((height, width, 3), dtype=np.uint8)
        yy, xx = np.indices((height, width))
        tile[((xx + yy) // 3) % 2 == 0] = spec.purple
        return Image.fromarray(tile, mode="RGB")
    image = Image.fromarray(cell.pixels, mode="RGB")
    return image.resize((width, height), Image.Resampling.NEAREST)


def _paint(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame: SyncedFrame,
    spec: GridSpec,
    stamp: FrameStamp,
) -> None:
    _text(canvas, spec.text_origin("objective"), stamp.objective, spec.yellow)
    for col, component in enumerate(COMPONENTS):
        box = spec.header_box(col)
        draw.rectangle(_rect(box), fill=spec.component_rgb[component])
        ink = spec.black if component == "total" else spec.yellow
        _text(canvas, (box[0] + 2, box[1] + 1), _COLUMN[component], ink)
    for row, strategy in enumerate(STRATEGIES):
        _paint_row(canvas, draw, frame, spec, stamp, row, strategy)
    _text(canvas, spec.text_origin("footer"), stamp.footer, spec.white)


def _paint_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame: SyncedFrame,
    spec: GridSpec,
    stamp: FrameStamp,
    row: int,
    strategy: str,
) -> None:
    badge = stamp.badges[row]
    draw.rectangle(
        _rect(spec.badge_box(row)), fill=spec.badge_rgb.get(badge, spec.purple)
    )
    _text(canvas, spec.badge_text_origin(row), badge, spec.black)
    _text(canvas, (2, spec.cell_box(row, 0)[1] + 1), strategy, spec.yellow)
    for col, component in enumerate(COMPONENTS):
        _paint_cell(canvas, frame.cell(strategy, component), spec, row, col)


def _paint_cell(
    canvas: Image.Image, cell: Cell, spec: GridSpec, row: int, col: int
) -> None:
    x, y, width, height = spec.cell_box(row, col)
    canvas.paste(_picture(spec, cell, width, height), (x, y))
    _text(canvas, (x + 1, y + 1), cell.units, spec.yellow)
    if cell.missing or cell.pixels is None:
        _text(canvas, spec.unavailable_origin(row, col), UNAVAILABLE, spec.yellow)


def _blank() -> SyncedFrame:
    cells = tuple(
        Cell(strategy, component, None, None, UNAVAILABLE, UNAVAILABLE, True)
        for strategy, component in _GRID
    )
    return SyncedFrame(0.0, cells)


def _sheet(
    pick: PhasePick | None, name: str, provenance: Provenance, spec: GridSpec
) -> np.ndarray:
    label = PHASE_LABEL[name]
    held = None if pick is None else pick.frame
    body, _painted = compose(held if held is not None else _blank(), provenance, spec)
    title = label if held is not None else f"{label} {UNAVAILABLE}"
    band = Image.new("RGB", (spec.width, spec.banner_h), spec.black)
    _text(band, (2, 1), title, spec.yellow)
    return np.vstack([np.array(band), body])
