"""Multiview layout model and frame compositor for the Capture Rig (#9810).

One versioned :class:`LayoutSpec` (JSON schema ``rig-layout/1.0.0``) says
which source goes in which cell of an up-to-4x4 grid and how it is shown
(rotation, flips, crop, fit, label); one :func:`compose` renders any
mapping of BGR frames through it. The live preview, playback and the
composite video export all draw through this module, so it is pure numpy +
cv2: no Qt, no I/O, no hidden state, and pixel-testable.

Colours are never literal here: :class:`Palette` carries BGR tuples the
caller derives from the theme (``Palette.from_hex(**get_current_colors())``)
and :data:`DEFAULT_PALETTE` is a neutral fallback.

``layout.py`` next door is the dock/pane layout of the tile window; this is
the *picture* layout.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

SCHEMA_VERSION = "rig-layout/1.0.0"
MAX_GRID = 4
ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)
FITS: tuple[str, ...] = ("fit", "fill", "stretch")
SOURCE_KINDS: tuple[str, ...] = ("live", "recorded", "overlay", "empty")
DEFAULT_CANVAS = (1280, 720)
LABEL_PAD_PX = 6
LABEL_FONT_SCALE = 0.5

BGR = tuple[int, int, int]
Frame = npt.NDArray[np.uint8]
Rect = tuple[int, int, int, int]


def _check(condition: bool, name: str, detail: str, value: Any = None) -> None:
    """Raise ``ValueError`` naming the offending field (always on)."""
    if not condition:
        got = "" if value is None else f" (got {value!r})"
        raise ValueError(f"{name}: {detail}{got}")


def _int(
    payload: Mapping[str, Any], key: str, default: int | None = None, name: str = ""
) -> int:
    value = payload.get(key, default)
    _check(
        isinstance(value, int) and not isinstance(value, bool),
        name or key,
        "must be an integer",
        value,
    )
    return int(value)


def _bool(payload: Mapping[str, Any], name: str, default: bool) -> bool:
    value = payload.get(name, default)
    _check(isinstance(value, bool), name, "must be true or false", value)
    return bool(value)


def _num(payload: Mapping[str, Any], name: str, default: float) -> float:
    value = payload.get(name, default)
    _check(
        isinstance(value, int | float) and not isinstance(value, bool),
        name,
        "must be a number",
        value,
    )
    return float(value)


def _bgr(value: Any, name: str) -> BGR:
    _check(
        isinstance(value, list | tuple) and len(value) == 3,
        name,
        "must be three BGR channel values",
        value,
    )
    channels = tuple(int(c) for c in value)
    _check(all(0 <= c <= 255 for c in channels), name, "channels must be 0..255", value)
    return channels[0], channels[1], channels[2]


# ------------------------------------------------------------------ model


@dataclass(frozen=True)
class SourceRef:
    """What a tile shows: a live view, a recording, an overlay render or nothing.

    ``key`` is the name the compositor looks up in its frames mapping, so
    producers and the layout agree without sharing objects.
    """

    kind: str
    view: str = ""
    variants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check(
            self.kind in SOURCE_KINDS,
            "source.kind",
            f"one of {SOURCE_KINDS}",
            self.kind,
        )
        if self.kind == "empty":
            _check(
                self.view == "", "source.view", "empty source has no view", self.view
            )
        else:
            _check(self.view.strip() != "", "source.view", "view name required")
        _check(
            isinstance(self.variants, tuple)
            and all(isinstance(v, str) and v != "" for v in self.variants),
            "source.variants",
            "must be a list of variant names",
            self.variants,
        )

    @classmethod
    def empty(cls) -> SourceRef:
        return cls(kind="empty")

    @property
    def is_empty(self) -> bool:
        return self.kind == "empty"

    @property
    def key(self) -> str:
        if self.is_empty:
            return "empty"
        base = f"{self.kind}:{self.view}"
        return f"{base}:{'+'.join(self.variants)}" if self.variants else base

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if not self.is_empty:
            out["view"] = self.view
        if self.variants:
            out["variants"] = list(self.variants)
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceRef:
        """Precondition: ``payload`` is a mapping; raises ``ValueError`` naming a field."""
        _check(isinstance(payload, Mapping), "source", "must be an object", payload)
        kind = payload.get("kind")
        _check(isinstance(kind, str), "source.kind", "must be a string", kind)
        view = payload.get("view", "")
        _check(isinstance(view, str), "source.view", "must be a string", view)
        variants = payload.get("variants", [])
        _check(isinstance(variants, list | tuple), "source.variants", "list", variants)
        return cls(kind=str(kind), view=view, variants=tuple(variants))


@dataclass(frozen=True)
class Cell:
    """Anchor cell plus span; validated against the grid by :class:`LayoutSpec`."""

    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1

    def __post_init__(self) -> None:
        _check(self.row >= 0, "cell.row", "must be >= 0", self.row)
        _check(self.col >= 0, "cell.col", "must be >= 0", self.col)
        _check(self.rowspan >= 1, "cell.rowspan", "must be >= 1", self.rowspan)
        _check(self.colspan >= 1, "cell.colspan", "must be >= 1", self.colspan)

    @property
    def anchor(self) -> tuple[int, int]:
        return self.row, self.col

    def covered(self) -> Iterable[tuple[int, int]]:
        for r in range(self.row, self.row + self.rowspan):
            for c in range(self.col, self.col + self.colspan):
                yield r, c

    def fits(self, rows: int, cols: int) -> bool:
        return self.row + self.rowspan <= rows and self.col + self.colspan <= cols

    def to_dict(self) -> dict[str, int]:
        return {
            "row": self.row,
            "col": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Cell:
        _check(isinstance(payload, Mapping), "cell", "must be an object", payload)
        return cls(
            row=_int(payload, "row", name="cell.row"),
            col=_int(payload, "col", name="cell.col"),
            rowspan=_int(payload, "rowspan", 1, name="cell.rowspan"),
            colspan=_int(payload, "colspan", 1, name="cell.colspan"),
        )


@dataclass(frozen=True)
class Crop:
    """Normalised sub-rectangle of the source frame, all in [0, 1]."""

    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    def __post_init__(self) -> None:
        _check(0.0 <= self.x <= 1.0, "crop.x", "must be in [0, 1]", self.x)
        _check(0.0 <= self.y <= 1.0, "crop.y", "must be in [0, 1]", self.y)
        _check(0.0 < self.w <= 1.0, "crop.w", "must be in (0, 1]", self.w)
        _check(0.0 < self.h <= 1.0, "crop.h", "must be in (0, 1]", self.h)
        _check(
            self.x + self.w <= 1.0 + 1e-9,
            "crop",
            "x + w must be <= 1",
            (self.x, self.w),
        )
        _check(
            self.y + self.h <= 1.0 + 1e-9,
            "crop",
            "y + h must be <= 1",
            (self.y, self.h),
        )

    @property
    def is_full(self) -> bool:
        return self == Crop()

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Crop:
        _check(isinstance(payload, Mapping), "crop", "must be an object", payload)
        return cls(
            x=_num(payload, "x", 0.0),
            y=_num(payload, "y", 0.0),
            w=_num(payload, "w", 1.0),
            h=_num(payload, "h", 1.0),
        )


@dataclass(frozen=True)
class Tile:
    """One picture in the grid and how it is transformed into its cell."""

    source: SourceRef
    cell: Cell
    rotation: int = 0
    flip_h: bool = False
    flip_v: bool = False
    crop: Crop = field(default_factory=Crop)
    fit: str = "fit"
    label: str | None = None
    show_label: bool = True

    def __post_init__(self) -> None:
        _check(
            self.rotation in ROTATIONS, "rotation", f"one of {ROTATIONS}", self.rotation
        )
        _check(self.fit in FITS, "fit", f"one of {FITS}", self.fit)
        _check(
            isinstance(self.flip_h, bool),
            "flip_h",
            "must be true or false",
            self.flip_h,
        )
        _check(
            isinstance(self.flip_v, bool),
            "flip_v",
            "must be true or false",
            self.flip_v,
        )
        _check(
            isinstance(self.show_label, bool),
            "show_label",
            "true or false",
            self.show_label,
        )
        _check(
            self.label is None or isinstance(self.label, str),
            "label",
            "string",
            self.label,
        )

    @property
    def caption(self) -> str:
        """The text drawn when ``show_label``: the label, else the source key."""
        return self.label or self.source.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "cell": self.cell.to_dict(),
            "rotation": self.rotation,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "crop": self.crop.to_dict(),
            "fit": self.fit,
            "label": self.label,
            "show_label": self.show_label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Tile:
        _check(isinstance(payload, Mapping), "tile", "must be an object", payload)
        label = payload.get("label")
        _check(
            label is None or isinstance(label, str), "label", "string or null", label
        )
        return cls(
            source=SourceRef.from_dict(payload.get("source", {})),
            cell=Cell.from_dict(payload.get("cell", {})),
            rotation=_int(payload, "rotation", 0),
            flip_h=_bool(payload, "flip_h", False),
            flip_v=_bool(payload, "flip_v", False),
            crop=Crop.from_dict(payload.get("crop", {})),
            fit=str(payload.get("fit", "fit")),
            label=label,
            show_label=_bool(payload, "show_label", True),
        )


@dataclass(frozen=True)
class LayoutSpec:
    """A named ``rows``x``cols`` grid of non-overlapping tiles on a canvas.

    Invariants (checked at construction): 1 <= rows, cols <= 4; every tile
    lies inside the grid; no two tiles cover the same cell. ``background``
    is ``None`` to use the palette's, else an explicit BGR tuple.
    """

    name: str
    rows: int
    cols: int
    tiles: tuple[Tile, ...] = ()
    canvas: tuple[int, int] = DEFAULT_CANVAS
    background: BGR | None = None

    def __post_init__(self) -> None:
        _check(
            isinstance(self.name, str) and self.name.strip() != "", "name", "required"
        )
        _check(1 <= self.rows <= MAX_GRID, "rows", f"must be 1..{MAX_GRID}", self.rows)
        _check(1 <= self.cols <= MAX_GRID, "cols", f"must be 1..{MAX_GRID}", self.cols)
        _check(
            len(self.canvas) == 2 and all(int(v) > 0 for v in self.canvas),
            "canvas",
            "must be a positive [width, height]",
            self.canvas,
        )
        object.__setattr__(self, "tiles", tuple(self.tiles))
        seen: dict[tuple[int, int], int] = {}
        for index, tile in enumerate(self.tiles):
            _check(
                tile.cell.fits(self.rows, self.cols),
                "cell",
                f"tile {index} leaves the {self.rows}x{self.cols} grid",
                tile.cell,
            )
            for rc in tile.cell.covered():
                _check(
                    rc not in seen,
                    "tiles",
                    f"tiles {seen.get(rc)} and {index} overlap",
                    rc,
                )
                seen[rc] = index

    # -- queries -------------------------------------------------------

    def tile_at(self, row: int, col: int) -> Tile | None:
        """The tile whose span covers ``(row, col)``, or ``None``."""
        for tile in self.tiles:
            if (row, col) in tile.cell.covered():
                return tile
        return None

    def source_keys(self) -> tuple[str, ...]:
        """Keys of the non-empty sources, in tile order, without duplicates."""
        out: list[str] = []
        for tile in self.tiles:
            if not tile.source.is_empty and tile.source.key not in out:
                out.append(tile.source.key)
        return tuple(out)

    # -- functional updates ---------------------------------------------

    def renamed(self, name: str) -> LayoutSpec:
        return replace(self, name=name)

    def with_tile(self, tile: Tile) -> LayoutSpec:
        """A copy with ``tile`` added, replacing whatever its cells covered.

        Postcondition: ``result.tile_at(*tile.cell.anchor) == tile``.
        """
        covered = set(tile.cell.covered())
        kept = tuple(
            t for t in self.tiles if not covered.intersection(t.cell.covered())
        )
        return replace(self, tiles=(*kept, tile))

    def without_tile(self, row: int, col: int) -> LayoutSpec:
        """A copy without the tile covering ``(row, col)`` (no-op when none)."""
        target = self.tile_at(row, col)
        return replace(self, tiles=tuple(t for t in self.tiles if t is not target))

    def move_tile(self, source: Cell, target: Cell) -> LayoutSpec:
        """A copy with the tile anchored at ``source`` re-anchored at ``target``.

        A tile anchored at ``target`` takes the vacated anchor (swap).
        Preconditions: a tile is anchored at ``source``; ``target`` is inside
        the grid. Raises ``ValueError`` when the moved span would leave the
        grid or overlap a third tile.
        """
        moving = next((t for t in self.tiles if t.cell.anchor == source.anchor), None)
        require(moving is not None, "no tile anchored at source cell", source)
        assert moving is not None
        require(target.fits(self.rows, self.cols), "target cell outside grid", target)
        other = next(
            (
                t
                for t in self.tiles
                if t.cell.anchor == target.anchor and t is not moving
            ),
            None,
        )
        rest = [t for t in self.tiles if t is not moving and t is not other]
        moved = replace(
            moving, cell=replace(moving.cell, row=target.row, col=target.col)
        )
        rest.append(moved)
        if other is not None:
            rest.append(
                replace(other, cell=replace(other.cell, row=source.row, col=source.col))
            )
        return replace(self, tiles=tuple(rest))

    # -- (de)serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "canvas": list(self.canvas),
            "background": None if self.background is None else list(self.background),
            "tiles": [tile.to_dict() for tile in self.tiles],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LayoutSpec:
        """Parse and validate; raises ``ValueError`` naming the offending field.

        Unknown keys (``provenance`` among them) are ignored.
        """
        _check(isinstance(payload, Mapping), "layout", "must be an object", payload)
        version = payload.get("schema_version", SCHEMA_VERSION)
        _check(
            version == SCHEMA_VERSION,
            "schema_version",
            f"expected {SCHEMA_VERSION}",
            version,
        )
        name = payload.get("name")
        _check(isinstance(name, str), "name", "must be a string", name)
        canvas = payload.get("canvas", list(DEFAULT_CANVAS))
        _check(
            isinstance(canvas, list | tuple)
            and len(canvas) == 2
            and all(isinstance(v, int) and v > 0 for v in canvas),
            "canvas",
            "must be a positive [width, height]",
            canvas,
        )
        background = payload.get("background")
        tiles = payload.get("tiles", [])
        _check(isinstance(tiles, list | tuple), "tiles", "must be a list", tiles)
        return cls(
            name=str(name),
            rows=_int(payload, "rows"),
            cols=_int(payload, "cols"),
            tiles=tuple(Tile.from_dict(t) for t in tiles),
            canvas=(int(canvas[0]), int(canvas[1])),
            background=None if background is None else _bgr(background, "background"),
        )


# ---------------------------------------------------------------- presets


def _grid(name: str, rows: int, cols: int, sources: Sequence[SourceRef]) -> LayoutSpec:
    refs = list(sources)
    tiles = []
    for index, (r, c) in enumerate((r, c) for r in range(rows) for c in range(cols)):
        source = refs[index] if index < len(refs) else SourceRef.empty()
        tiles.append(Tile(source=source, cell=Cell(r, c)))
    return LayoutSpec(name=name, rows=rows, cols=cols, tiles=tuple(tiles))


def _primary_plus_strip(sources: Sequence[SourceRef]) -> LayoutSpec:
    """One large tile (3 rows x 4 cols) above a strip of four small ones."""
    refs = list(sources)

    def pick(index: int) -> SourceRef:
        return refs[index] if index < len(refs) else SourceRef.empty()

    tiles = [Tile(source=pick(0), cell=Cell(0, 0, rowspan=3, colspan=4))]
    tiles.extend(Tile(source=pick(i + 1), cell=Cell(3, i)) for i in range(4))
    return LayoutSpec(name="primary_plus_strip", rows=4, cols=4, tiles=tuple(tiles))


_PRESETS: dict[str, Callable[[Sequence[SourceRef]], LayoutSpec]] = {
    "single": lambda s: _grid("single", 1, 1, s),
    "side_by_side": lambda s: _grid("side_by_side", 1, 2, s),
    "three_across": lambda s: _grid("three_across", 1, 3, s),
    "two_by_two": lambda s: _grid("two_by_two", 2, 2, s),
    "three_by_three": lambda s: _grid("three_by_three", 3, 3, s),
    "four_by_four": lambda s: _grid("four_by_four", 4, 4, s),
    "primary_plus_strip": _primary_plus_strip,
}
PRESET_NAMES: tuple[str, ...] = tuple(_PRESETS)


def preset(name: str, sources: Sequence[SourceRef] = ()) -> LayoutSpec:
    """The built-in layout ``name`` with ``sources`` filled in tile order.

    Cells beyond the given sources hold empty tiles, so every preset covers
    its whole grid. Raises ``ValueError`` for an unknown preset name.
    """
    _check(name in _PRESETS, "preset", f"one of {PRESET_NAMES}", name)
    return _PRESETS[name](sources)


# ------------------------------------------------------------- compositor


@dataclass(frozen=True)
class Palette:
    """BGR colours the compositor draws with; derive from the theme via ``from_hex``."""

    background: BGR = (16, 16, 16)
    placeholder: BGR = (48, 48, 48)
    placeholder_text: BGR = (160, 160, 160)
    label_text: BGR = (240, 240, 240)
    label_box: BGR = (0, 0, 0)

    @classmethod
    def from_hex(cls, **colours: str) -> Palette:
        """Override any field from ``#rrggbb`` strings; unknown names are ignored.

        Raises ``ValueError`` naming the field for a value that is not hex.
        """
        names = {f.name for f in fields(cls)}
        out: dict[str, BGR] = {}
        for name, value in colours.items():
            if name not in names:
                continue
            text = value.strip().lstrip("#")
            _check(
                len(text) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in text),
                name,
                "must be #rrggbb",
                value,
            )
            r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
            out[name] = (b, g, r)
        return cls(**out)


DEFAULT_PALETTE = Palette()


def cell_rect(spec: LayoutSpec, cell: Cell, size: tuple[int, int]) -> Rect:
    """Pixel rectangle ``(x0, y0, x1, y1)`` of ``cell`` on a ``size`` (w, h) canvas.

    Integer partition, so adjacent cells share edges and cover the canvas
    exactly. Precondition: the cell fits the grid.
    """
    require(cell.fits(spec.rows, spec.cols), "cell must fit the grid", cell)
    width, height = size
    x0 = cell.col * width // spec.cols
    x1 = (cell.col + cell.colspan) * width // spec.cols
    y0 = cell.row * height // spec.rows
    y1 = (cell.row + cell.rowspan) * height // spec.rows
    return x0, y0, x1, y1


def _as_bgr(frame: npt.NDArray[Any]) -> Frame:
    require(frame.dtype == np.uint8, "frames must be uint8", frame.dtype)
    require(frame.ndim in (2, 3), "frames must be HxW or HxWx3", frame.shape)
    if frame.ndim == 2:
        return np.ascontiguousarray(np.repeat(frame[:, :, None], 3, axis=2))
    require(frame.shape[2] == 3, "colour frames must have 3 channels", frame.shape)
    return frame


def _crop(frame: Frame, crop: Crop) -> Frame:
    if crop.is_full:
        return frame
    h, w = frame.shape[:2]
    x0, y0 = int(round(crop.x * w)), int(round(crop.y * h))
    x1 = max(x0 + 1, int(round((crop.x + crop.w) * w)))
    y1 = max(y0 + 1, int(round((crop.y + crop.h) * h)))
    return frame[y0 : min(y1, h), x0 : min(x1, w)]


def _orient(frame: Frame, tile: Tile) -> Frame:
    """Rotate clockwise by ``tile.rotation`` then apply the flips."""
    out = frame
    if tile.rotation:
        out = np.rot90(out, k=-(tile.rotation // 90))
    if tile.flip_h:
        out = out[:, ::-1]
    if tile.flip_v:
        out = out[::-1]
    return out


def _resize(frame: Frame, width: int, height: int) -> Frame:
    import cv2

    out: Frame = np.asarray(
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), dtype=np.uint8
    )
    return out


def _fit(frame: Frame, rect_w: int, rect_h: int, fit: str) -> tuple[Frame, int, int]:
    """Resize (and for ``fill`` centre-crop) a frame into ``rect_w`` x ``rect_h``.

    Returns the image plus its offset inside the rectangle (letterbox margin).
    """
    h, w = frame.shape[:2]
    if fit == "stretch":
        return _resize(frame, rect_w, rect_h), 0, 0
    scale = (min if fit == "fit" else max)(rect_w / w, rect_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = _resize(frame, new_w, new_h)
    if fit == "fit":
        return resized, (rect_w - new_w) // 2, (rect_h - new_h) // 2
    x0, y0 = (new_w - rect_w) // 2, (new_h - rect_h) // 2
    return resized[y0 : y0 + rect_h, x0 : x0 + rect_w], 0, 0


def _draw_label(canvas: Frame, rect: Rect, text: str, palette: Palette) -> None:
    import cv2

    x0, y0, x1, y1 = rect
    scale = LABEL_FONT_SCALE * max(0.6, min(1.5, (y1 - y0) / 360.0))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = LABEL_PAD_PX
    bx1 = min(x1, x0 + tw + 2 * pad)
    by1 = min(y1, y0 + th + baseline + 2 * pad)
    cv2.rectangle(canvas, (x0, y0), (bx1, by1), palette.label_box, -1)
    cv2.putText(
        canvas,
        text,
        (x0 + pad, y0 + pad + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        palette.label_text,
        1,
        cv2.LINE_AA,
    )


def _render_tile(
    canvas: Frame, tile: Tile, rect: Rect, frame: Frame | None, palette: Palette
) -> None:
    x0, y0, x1, y1 = rect
    rect_w, rect_h = x1 - x0, y1 - y0
    if rect_w <= 0 or rect_h <= 0:
        return
    if frame is None:
        canvas[y0:y1, x0:x1] = palette.placeholder
        if tile.show_label:
            _draw_label(canvas, rect, tile.caption, palette)
        return
    image = _orient(_crop(_as_bgr(frame), tile.crop), tile)
    fitted, dx, dy = _fit(np.ascontiguousarray(image), rect_w, rect_h, tile.fit)
    fh, fw = fitted.shape[:2]
    canvas[y0 + dy : y0 + dy + fh, x0 + dx : x0 + dx + fw] = fitted
    if tile.show_label:
        _draw_label(canvas, rect, tile.caption, palette)


def compose(
    frames: Mapping[str, npt.NDArray[Any]],
    spec: LayoutSpec,
    size: tuple[int, int] | None = None,
    palette: Palette = DEFAULT_PALETTE,
) -> Frame:
    """Render ``frames`` (keyed by ``SourceRef.key``) through ``spec``.

    Each tile's frame is cropped, rotated, flipped and fitted into its cell
    (letterboxed for ``fit``, centre-cropped for ``fill``); a missing or
    empty source shows the palette's placeholder; labels are drawn when the
    tile asks for them. Preconditions: ``size`` (w, h) positive when given;
    every used frame is uint8 HxW or HxWx3. Postconditions: the result is a
    new ``(h, w, 3)`` uint8 array, inputs are not mutated, and the same
    inputs always give the same pixels.
    """
    width, height = size or spec.canvas
    require(width > 0 and height > 0, "canvas size must be positive", (width, height))
    canvas: Frame = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = spec.background if spec.background is not None else palette.background
    for tile in spec.tiles:
        frame = None if tile.source.is_empty else frames.get(tile.source.key)
        _render_tile(
            canvas, tile, cell_rect(spec, tile.cell, (width, height)), frame, palette
        )
    return canvas
