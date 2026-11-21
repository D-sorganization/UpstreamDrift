"""Compact side panel of the export ellipsoids (#15150).

A small table drawn by the impact factors panel's renderer replaces the wrapped
HUD text rows that covered the clubhead and ball: a row per point (largest
mobility and effective-mass radius, condition number), a units row and, with
colour-by-kind, a legend. It sits opposite the impact factors panel, stacks
below it on a shared edge and stays above the bottom HUD rows. Options:
``ellipsoid_panel_settings.json`` (layout ``"text"`` restores the HUD rows).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.force_overlay.palette import ellipsoid_kind_rgba
from src.tools.native_viewer_export import compositor

if TYPE_CHECKING:
    from src.tools.native_viewer_export.core import ClipOverlays, ExportSettings
    from src.tools.native_viewer_export.impact_factors import FactorsPanel

SETTINGS_PATH = Path(__file__).with_name("ellipsoid_panel_settings.json")
LAYOUTS, POSITIONS = ("panel", "text"), ("auto", "left", "right")
#: kind -> the :class:`PanelRow` field its column shows
KIND_FIELDS = {"mobility": "mobility", "effective-mass": "effective_mass"}
Line = tuple[str, tuple[str, ...] | str, bool]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PanelSettings:
    """Validated ``ellipsoid_panel_settings.json`` (``ValueError``)."""

    layout: str
    position: str
    opacity: float
    font_height_divisor: int
    min_font_px: int
    title: str
    units_label: str
    units: dict[str, str]
    headers: dict[str, str]
    legend: dict[str, str]
    radius_decimals: int
    condition_decimals: int

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS or self.position not in POSITIONS:
            raise ValueError(f"layout must be in {LAYOUTS}, position in {POSITIONS}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must lie in [0, 1]")
        if min(self.font_height_divisor, self.min_font_px) < 1:
            raise ValueError("font sizes must be positive")
        if any(
            set(KIND_FIELDS) - set(t) for t in (self.headers, self.legend, self.units)
        ):
            raise ValueError("headers, legend and units must name every kind")


def load_panel_settings() -> PanelSettings:
    """Read ``ellipsoid_panel_settings.json`` beside this module."""
    raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    raw.pop("sources", None)
    return PanelSettings(**raw)


@dataclass(frozen=True)
class PanelRow:
    """One table point: its largest radii and condition number."""

    label: str
    mobility: float
    effective_mass: float
    condition: float

    def __post_init__(self) -> None:
        if not (self.label and math.isfinite(self.mobility + self.effective_mass)):
            raise ValueError("a panel row needs a label and finite radii")


def _text(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}" if math.isfinite(value) else "inf"


@dataclass(frozen=True)
class EllipsoidPanel:
    """The panel of one export: ``kinds`` as columns, beside ``factors`` (the
    impact factors panel, ``None`` when off)."""

    settings: PanelSettings
    kinds: tuple[str, ...]
    by_kind: bool
    factors: FactorsPanel | None = field(default=None, compare=False)

    @property
    def position(self) -> str:
        """``left`` or ``right``: the setting, or opposite the factors panel."""
        if self.settings.position != "auto":
            return self.settings.position
        other = self.factors is not None and self.factors.settings.position == "left"
        return "right" if other else "left"

    def lines(self, rows: tuple[PanelRow, ...]) -> list[Line]:
        """Header, units, a line per row, then (colour-by-kind) the legend."""
        cfg = self.settings
        head = (*(cfg.headers[k] for k in self.kinds), cfg.headers["condition"])
        units = (*(cfg.units[k] for k in self.kinds), "1")
        out: list[Line] = [("", head, True), (cfg.units_label, units, True)]
        for row in rows:
            radii = (
                _text(getattr(row, KIND_FIELDS[k]), cfg.radius_decimals)
                for k in self.kinds
            )
            cond = _text(row.condition, cfg.condition_decimals)
            out.append((row.label.replace("_", " ").title(), (*radii, cond), False))
        if self.by_kind:
            out += [(cfg.legend[k], "", False) for k in self.kinds]
        return out

    def _box(self, shape: tuple[int, ...], lines: list[Line], size: int) -> Box:
        """Box of ``lines``, lowered below the factors panel on a shared edge."""
        x0, y0, x1, y1 = compositor.text_panel_box(
            shape, self.settings.title, lines, self.position, size
        )[0]
        other = self.factors.bbox(shape) if self.factors is not None else None
        if other is not None and other[0] < x1 and x0 < other[2]:
            shift = other[3] + compositor.HUD_LAYOUT.panel_gap_px - y0
            y0, y1 = y0 + shift, y1 + shift
        return x0, y0, x1, y1

    def fit(
        self, shape: tuple[int, ...], rows: tuple[PanelRow, ...], limit_y: int
    ) -> tuple[Box, int, list[Line]]:
        """First layout ending above ``limit_y``: the font shrinks, then the last
        line (legend first, then rows) is dropped."""
        cfg, lines = self.settings, self.lines(rows)
        top = max(cfg.min_font_px, int(shape[0]) // cfg.font_height_divisor)
        while True:
            for size in range(top, cfg.min_font_px - 1, -1):
                box = self._box(shape, lines, size)
                if box[3] <= limit_y or not lines:
                    return box, size, lines
            lines = lines[:-1]

    def draw(
        self, img: compositor.Image8, rows: tuple[PanelRow, ...], limit_y: int
    ) -> compositor.Image8:
        """Copy of ``img`` with the panel; only pixels of :meth:`box` change."""
        box, size, lines = self.fit(img.shape, rows, limit_y)
        if not rows or len(lines) <= 2:  # no point fits above the HUD rows
            return img
        rgb = {
            self.settings.legend[k]: tuple(
                int(c * 255) for c in ellipsoid_kind_rgba(k)[:3]
            )
            for k in self.kinds
        }
        colours: list[Any] = [rgb.get(label) for label, _, _ in lines]
        return compositor.draw_text_panel(
            img, box, size, self.settings.title, lines, self.settings.opacity, colours
        )


def with_panel(
    overlays: ClipOverlays | None,
    export: ExportSettings,
    feed: Any,
    factors: FactorsPanel | None,
) -> ClipOverlays | None:
    """``overlays`` with the export's panel; unchanged (HUD text rows stay) when
    the layout is ``text``, the HUD is off, or the feed has no rows or kinds."""
    cfg = export.ellipsoid_panel or load_panel_settings()
    kinds = tuple(k for k in KIND_FIELDS if k in export.interface_ellipsoids)
    has_rows = getattr(feed, "hand_panel_rows", None) is not None
    if cfg.layout != "panel" or not (export.hud and has_rows and kinds):
        return overlays
    from src.tools.native_viewer_export.core import ClipOverlays

    panel = EllipsoidPanel(cfg, kinds, export.ellipsoid_color_by == "kind", factors)
    return replace(overlays or ClipOverlays(), ellipsoids=panel)


def hud_lines(overlays: ClipOverlays | None, feed: Any, index: int) -> tuple[str, ...]:
    """HUD text lines of frame ``index``: none while the panel shows them."""
    shown = overlays is not None and overlays.ellipsoids is not None
    return () if feed is None or shown else feed.hud_lines_at(index)  # type: ignore[no-any-return]


def hud_top_px(shape: tuple[int, ...], hud: compositor.HudInfo) -> int:
    """Top edge of the first bottom HUD row :func:`compositor.draw_hud` draws."""
    size, width = compositor.HUD_SIZE_PX, int(shape[1])
    rows = [r for t in hud.lines() for r in compositor.wrap_hud_line(t, width, size)]
    return compositor.hud_line_boxes(shape, rows, size)[0][1]


def draw_all(
    overlays: ClipOverlays | None,
    feed: Any,
    index: int,
    hud: compositor.HudInfo,
    outputs: dict[str, compositor.Image8],
) -> dict[str, compositor.Image8]:
    """``outputs`` with the panel of frame ``index`` drawn above the HUD rows."""
    panel = None if overlays is None else overlays.ellipsoids
    if panel is None or feed is None:
        return outputs
    rows = feed.hand_panel_rows(index)
    return {
        n: panel.draw(f, rows, hud_top_px(f.shape, hud)) for n, f in outputs.items()
    }
