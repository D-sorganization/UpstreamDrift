"""Low-level DXF rendering primitives.

This module provides the foundational drawing functions used by all other
modules: text, boxes, arrows, layers, and SVG export.

Preconditions:
    - All ``msp`` arguments must be an ezdxf modelspace object.
    - Numeric arguments are coerced via ``to_float`` for robustness.

Postconditions:
    - Functions add entities to the modelspace on the specified layer.
    - ``ensure_layer`` / ``ensure_layers`` are idempotent.
"""

from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
except ImportError:  # pragma: no cover — optional heavy dep
    ezdxf = None  # type: ignore[assignment]
    TextEntityAlignment = None  # type: ignore[assignment,misc]
from programmatic_pid.geometry import to_float
from programmatic_pid.spec_loader import get_drawing, get_layer_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer management
# ---------------------------------------------------------------------------


def ensure_layer(
    doc: Any, name: str, color: int = 7, linetype: str = "CONTINUOUS"
) -> None:
    """Create a layer if it doesn't already exist.

    Preconditions:
        - ``doc`` is a valid ezdxf Drawing.
        - ``name`` is a non-empty string.

    Postconditions:
        - ``name`` exists in ``doc.layers``.
    """
    if name is None:
        raise ValueError("name must be provided")
    if not name:
        return
    if name in doc.layers:
        return
    attrs = {"color": int(color), "linetype": str(linetype)}
    try:
        doc.layers.new(name=name, dxfattribs=attrs)
    except ezdxf.DXFValueError:
        attrs["linetype"] = "CONTINUOUS"
        doc.layers.new(name=name, dxfattribs=attrs)


def ensure_layers(doc: Any, spec: dict[str, Any]) -> None:
    """Create all layers declared in the spec plus standard defaults."""
    if spec is None:
        raise ValueError("spec must be provided")
    for name, cfg in get_layer_config(spec).items():
        cfg = cfg or {}
        ensure_layer(
            doc,
            name,
            color=cfg.get("color", 7),
            linetype=cfg.get("linetype", "CONTINUOUS"),
        )

    for name, color in (
        ("TEXT", 7),
        ("NOTES", 3),
        ("LEADERS", 8),
        ("EQUIPMENT", 7),
        ("INSTRUMENTS", 2),
        ("PROCESS", 5),
    ):
        ltype = "DASHED" if name == "LEADERS" else "CONTINUOUS"
        ensure_layer(doc, name, color=color, linetype=ltype)


def layer_name(
    layer_index: dict[str, str],
    *candidates: str,
    default: str = "0",
) -> str:
    """Resolve a layer name from a case-insensitive index.

    Tries each *candidate* in order.  Returns the first match found in
    ``layer_index`` or *default*.
    """
    for candidate in candidates:
        if not candidate:
            continue
        actual = layer_index.get(str(candidate).lower())
        if actual:
            return actual
    return default


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def parse_alignment(align: Any) -> TextEntityAlignment:
    """Convert a string alignment name to an ezdxf ``TextEntityAlignment``."""
    if isinstance(align, TextEntityAlignment):
        return align
    key = str(align or "MIDDLE_CENTER").upper()
    return getattr(TextEntityAlignment, key, TextEntityAlignment.MIDDLE_CENTER)


def wrap_text_lines(text: str, width: int) -> list[str]:
    """Wrap *text* to *width* characters, returning a list of lines."""
    if text is None:
        raise ValueError("text must be provided")
    chunks = textwrap.wrap(
        str(text),
        width=max(int(width), 12),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return chunks if chunks else [str(text)]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def add_text(
    msp: Any,
    text: str,
    x: float,
    y: float,
    h: float,
    layer: str = "TEXT",
    align: str = "MIDDLE_CENTER",
) -> Any:
    """Add a text entity to the modelspace."""
    if text is None:
        raise ValueError("text must be provided")
    t = msp.add_text(
        str(text),
        dxfattribs={
            "height": max(to_float(h, 1.0), 0.1),
            "layer": layer,
        },
    )
    t.set_placement((to_float(x), to_float(y)), align=parse_alignment(align))
    return t


def add_text_panel(
    msp: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Sequence[str | None],
    text_h: float,
    text_layer: str,
    border_layer: str,
    max_chars: int = 42,
) -> None:
    """Draw a bordered panel with a title and wrapped text lines."""
    if x is None:
        raise ValueError("x must be provided")
    add_box(msp, x, y, w, h, border_layer)
    inset_x = x + 1.1
    inset_top = y + h - 1.0
    add_text(
        msp,
        title,
        inset_x,
        inset_top,
        text_h * 1.05,
        layer=text_layer,
        align="TOP_LEFT",
    )

    step = max(text_h * 1.16, 0.9)
    available = max(int((h - 2.6) / step), 1)
    out: list[str] = []
    for line in lines:
        if line is None:
            out.append("")
            continue
        out.extend(wrap_text_lines(line, max_chars))
    out = out[:available]

    cy = inset_top - max(text_h * 1.55, 1.1)
    for line in out:
        add_text(msp, line, inset_x, cy, text_h, layer=text_layer, align="TOP_LEFT")
        cy -= step


def add_box(msp: Any, x: float, y: float, w: float, h: float, layer: str) -> None:
    """Draw a rectangular outline on the given layer."""
    if x is None:
        raise ValueError("x must be provided")
    x = to_float(x)
    y = to_float(y)
    w = to_float(w)
    h = to_float(h)
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})


# ---------------------------------------------------------------------------
# Arrow primitives
# ---------------------------------------------------------------------------


def add_arrow_head(
    msp: Any,
    s: Sequence[float],
    e: Sequence[float],
    layer: str,
    color: int | None = None,
    arrow_size: float = 1.6,
) -> None:
    """Draw a filled arrowhead at the end of a line from *s* to *e*."""
    if s is None:
        raise ValueError("s must be provided")
    sx, sy = to_float(s[0]), to_float(s[1])
    ex, ey = to_float(e[0]), to_float(e[1])
    attrs: dict[str, Any] = {"layer": layer}
    if color is not None:
        attrs["color"] = int(color)

    ang = math.atan2(ey - sy, ex - sx)
    ah = max(to_float(arrow_size, 1.6), 0.2)
    aw = ah * 0.45
    p1 = (ex, ey)
    p2 = (
        ex - ah * math.cos(ang) + aw * math.sin(ang),
        ey - ah * math.sin(ang) - aw * math.cos(ang),
    )
    p3 = (
        ex - ah * math.cos(ang) - aw * math.sin(ang),
        ey - ah * math.sin(ang) + aw * math.cos(ang),
    )
    msp.add_solid([p1, p2, p3, p3], dxfattribs=attrs)


def add_arrow(
    msp: Any,
    s: Sequence[float],
    e: Sequence[float],
    layer: str,
    color: int | None = None,
    arrow_size: float = 1.6,
) -> None:
    """Draw a line with an arrowhead at the end."""
    if s is None:
        raise ValueError("s must be provided")
    sx, sy = to_float(s[0]), to_float(s[1])
    ex, ey = to_float(e[0]), to_float(e[1])
    attrs: dict[str, Any] = {"layer": layer}
    if color is not None:
        attrs["color"] = int(color)
    msp.add_line((sx, sy), (ex, ey), dxfattribs=attrs)
    add_arrow_head(
        msp, (sx, sy), (ex, ey), layer=layer, color=color, arrow_size=arrow_size
    )


def add_poly_arrow(
    msp: Any,
    verts: Sequence[Sequence[float]],
    layer: str,
    color: int | None = None,
    arrow_size: float = 1.6,
) -> None:
    """Draw a polyline with an arrowhead on the last segment."""
    if verts is None:
        raise ValueError("verts must be provided")
    points = [(to_float(v[0]), to_float(v[1])) for v in verts if len(v) >= 2]
    if len(points) < 2:
        return
    attrs: dict[str, Any] = {"layer": layer}
    if color is not None:
        attrs["color"] = int(color)
    msp.add_lwpolyline(points, dxfattribs=attrs)
    add_arrow_head(
        msp, points[-2], points[-1], layer, color=color, arrow_size=arrow_size
    )


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------


def export_svg_from_dxf(
    spec: dict[str, Any],
    dxf_path: str | Path,
    svg_path: str | Path | None,
    fallback_extent: tuple[float, float, float, float],
) -> None:
    """Render a DXF file to SVG using ezdxf's drawing backend.

    Postconditions:
        - If *svg_path* is not ``None`` and export succeeds, the SVG file exists.
        - On failure, a warning is logged (never raises).
    """
    if spec is None:
        raise ValueError("spec must be provided")
    if not svg_path:
        return
    x_min, y_min, x_max, y_max = fallback_extent
    try:
        from ezdxf import recover
        from ezdxf.addons.drawing import Frontend, RenderContext, layout, svg

        audit_doc, auditor = recover.readfile(str(dxf_path))
        ctx = RenderContext(audit_doc)
        backend = svg.SVGBackend()
        Frontend(ctx, backend).draw_layout(audit_doc.modelspace(), finalize=True)

        paper = get_drawing(spec).get("paper", {})
        page_width = to_float(paper.get("width"), max(x_max - x_min, 100.0))
        page_height = to_float(paper.get("height"), max(y_max - y_min, 100.0))
        unit_name = str(paper.get("units", "mm")).lower()
        unit_map = {
            "mm": layout.Units.mm,
            "cm": layout.Units.cm,
            "inch": layout.Units.inch,
            "in": layout.Units.inch,
            "pt": layout.Units.pt,
            "px": layout.Units.px,
        }
        page = layout.Page(
            page_width, page_height, units=unit_map.get(unit_name, layout.Units.mm)
        )
        Path(svg_path).write_text(backend.get_string(page), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DXF created, but SVG export failed: %s", exc)
