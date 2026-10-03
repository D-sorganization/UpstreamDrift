"""Renderer-neutral 2D glyph projection for video overlays (FTO-29, #11314).

Projects 3D world GlyphSet coordinates to 2D image pixels through an ImageProjector,
producing a ProjectedGlyphSet with pixel polyline shafts and triangle arrowhead polygons.
Shared between OpenCV video renderer and Web/SVG video force overlay (DRY).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import numpy.typing as npt

from src.shared.python.force_overlay.palette import FORCE_KIND_PALETTE

if TYPE_CHECKING:
    from src.shared.python.force_overlay.glyphs import (
        ArrowGlyph,
        GlyphSet,
        LegendSpec,
        TorqueArcGlyph,
    )
    from src.shared.python.force_overlay.renderers.opencv_glyphs import (
        ImageProjector,
        VideoGlyphReceipt,
        VideoGlyphStyle,
    )

__all__ = [
    "ProjectedArrowGlyph",
    "ProjectedGlyphSet",
    "ProjectedTorqueArcGlyph",
    "project_glyphs",
]


def _rgba_to_hex(rgba: tuple[float, float, float, float] | Sequence[float]) -> str:
    """Convert float RGBA to #RRGGBB hex string."""
    r = min(255, max(0, int(round(rgba[0] * 255.0))))
    g = min(255, max(0, int(round(rgba[1] * 255.0))))
    b = min(255, max(0, int(round(rgba[2] * 255.0))))
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass(frozen=True)
class ProjectedArrowGlyph:
    """Projected 2D arrow in image pixel space."""

    start_px: tuple[float, float]
    end_px: tuple[float, float]
    polyline_px: tuple[tuple[float, float], ...]
    head_poly_px: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    rgba: tuple[float, float, float, float]
    color_hex: str
    kind: str
    label: str
    magnitude: float
    units: str
    shaft_width_px: float
    halo_width_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_px": [float(self.start_px[0]), float(self.start_px[1])],
            "end_px": [float(self.end_px[0]), float(self.end_px[1])],
            "polyline_px": [[float(p[0]), float(p[1])] for p in self.polyline_px],
            "head_poly_px": [[float(p[0]), float(p[1])] for p in self.head_poly_px],
            "rgba": [float(c) for c in self.rgba],
            "color_hex": self.color_hex,
            "kind": self.kind,
            "label": self.label,
            "magnitude": float(self.magnitude),
            "units": self.units,
            "shaft_width_px": float(self.shaft_width_px),
            "halo_width_px": float(self.halo_width_px),
        }


@dataclass(frozen=True)
class ProjectedTorqueArcGlyph:
    """Projected 2D torque arc in image pixel space."""

    polyline_px: tuple[tuple[float, float], ...]
    head_poly_px: tuple[tuple[float, float], ...] | None
    rgba: tuple[float, float, float, float]
    color_hex: str
    kind: str
    label: str
    magnitude: float
    units: str
    shaft_width_px: float
    halo_width_px: float

    def to_dict(self) -> dict[str, Any]:
        head = (
            [[float(p[0]), float(p[1])] for p in self.head_poly_px]
            if self.head_poly_px
            else None
        )
        return {
            "polyline_px": [[float(p[0]), float(p[1])] for p in self.polyline_px],
            "head_poly_px": head,
            "rgba": [float(c) for c in self.rgba],
            "color_hex": self.color_hex,
            "kind": self.kind,
            "label": self.label,
            "magnitude": float(self.magnitude),
            "units": self.units,
            "shaft_width_px": float(self.shaft_width_px),
            "halo_width_px": float(self.halo_width_px),
        }


@dataclass(frozen=True)
class ProjectedGlyphSet:
    """Container of 2D projected glyphs for a video frame."""

    time_s: float
    image_size_px: tuple[int, int]
    arrows: tuple[ProjectedArrowGlyph, ...]
    torque_arcs: tuple[ProjectedTorqueArcGlyph, ...]
    legend: LegendSpec
    receipt: VideoGlyphReceipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": float(self.time_s),
            "image_size_px": [int(self.image_size_px[0]), int(self.image_size_px[1])],
            "arrows": [a.to_dict() for a in self.arrows],
            "torque_arcs": [a.to_dict() for a in self.torque_arcs],
            "legend": {
                "engine": self.legend.engine,
                "force_reference_n": self.legend.force_reference_n,
                "torque_reference_nm": self.legend.torque_reference_nm,
                "kinds_present": list(self.legend.kinds_present),
                "unavailable_labels": list(self.legend.unavailable_labels),
                "source_labels": list(self.legend.source_labels),
            },
            "receipt": self.receipt.to_dict(),
        }


def _resolve_image_size(
    projector: ImageProjector, image_size_px: tuple[int, int] | None
) -> tuple[int, int]:
    """Resolve image size in pixels from parameter or projector camera."""
    if image_size_px is not None:
        return (int(image_size_px[0]), int(image_size_px[1]))
    if hasattr(projector, "camera") and hasattr(projector.camera, "image_size_px"):
        cam_size = projector.camera.image_size_px
        return (int(cam_size[0]), int(cam_size[1]))
    return (1920, 1080)


def _compute_triangle_head(
    tip_px: npt.NDArray[np.float64],
    base_px: npt.NDArray[np.float64],
    head_px: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute 2D arrowhead triangle vertices (tip, base+norm, base-norm)."""
    v = tip_px - base_px
    vl = float(math.hypot(float(v[0]), float(v[1])))
    u = v / vl if vl > 1e-4 else np.array([0.0, 1.0])
    norm = np.array([-u[1], u[0]]) * (head_px / 2.0)
    p_tip = (float(tip_px[0]), float(tip_px[1]))
    p_b1 = (float(base_px[0] + norm[0]), float(base_px[1] + norm[1]))
    p_b2 = (float(base_px[0] - norm[0]), float(base_px[1] - norm[1]))
    return (p_tip, p_b1, p_b2)


def _compute_outcode(x: float, y: float, w: int, h: int) -> int:
    code = 0
    if x < 0:
        code |= 1
    elif x >= w:
        code |= 2
    if y < 0:
        code |= 4
    elif y >= h:
        code |= 8
    return code


def _segment_intersects_rect(
    p1: tuple[float, float], p2: tuple[float, float], w: int, h: int
) -> bool:
    """Check if line segment intersects [0, w) x [0, h) rectangle."""
    x0, y0 = p1
    x1, y1 = p2
    c0 = _compute_outcode(x0, y0, w, h)
    c1 = _compute_outcode(x1, y1, w, h)
    for _ in range(4):
        if not (c0 | c1):
            return True
        if c0 & c1:
            return False
        out = c0 or c1
        if out & 8:
            x = x0 + (x1 - x0) * (h - 1 - y0) / (y1 - y0) if y1 != y0 else x0
            y = float(h - 1)
        elif out & 4:
            x = x0 + (x1 - x0) * (-y0) / (y1 - y0) if y1 != y0 else x0
            y = 0.0
        elif out & 2:
            y = y0 + (y1 - y0) * (w - 1 - x0) / (x1 - x0) if x1 != x0 else y0
            x = float(w - 1)
        else:
            y = y0 + (y1 - y0) * (-x0) / (x1 - x0) if x1 != x0 else y0
            x = 0.0
        if out == c0:
            x0, y0, c0 = x, y, _compute_outcode(x, y, w, h)
        else:
            x1, y1, c1 = x, y, _compute_outcode(x, y, w, h)
    return False


def _arrow_in_frame(
    tail_pt: tuple[float, float],
    base_pt: tuple[float, float],
    head_poly: tuple[tuple[float, float], ...],
    w: int,
    h: int,
) -> bool:
    """True if shaft intersects frame or any head vertex is inside frame."""
    if _segment_intersects_rect(tail_pt, base_pt, w, h):
        return True
    return any(0 <= pt[0] < w and 0 <= pt[1] < h for pt in head_poly)


def _project_single_arrow(
    arrow: ArrowGlyph,
    projector: ImageProjector,
    head_px: float,
    line_px: float,
    halo_px: float,
    w: int,
    h: int,
) -> tuple[ProjectedArrowGlyph | None, bool, bool]:
    """Project a single arrow glyph. Returns (glyph_or_none, is_behind, is_out)."""
    pts = np.array([arrow.tail_m, arrow.tip_m, arrow.head_base_m], float)
    pix, val = projector.project(pts)
    if not (val[0] and val[1]):
        return None, True, False

    base = pix[2] if val[2] else pix[1]
    head_poly = _compute_triangle_head(pix[1], base, head_px)
    tail_pt = (float(pix[0][0]), float(pix[0][1]))
    base_pt, tip_pt = (
        (float(base[0]), float(base[1])),
        (float(pix[1][0]), float(pix[1][1])),
    )

    if not _arrow_in_frame(tail_pt, base_pt, head_poly, w, h):
        return None, False, True

    color_hex = FORCE_KIND_PALETTE.get(arrow.kind, _rgba_to_hex(arrow.rgba))
    glyph = ProjectedArrowGlyph(
        start_px=tail_pt,
        end_px=tip_pt,
        polyline_px=(tail_pt, base_pt),
        head_poly_px=head_poly,
        rgba=arrow.rgba,
        color_hex=color_hex,
        kind=str(arrow.kind),
        label=str(arrow.label),
        magnitude=float(arrow.magnitude),
        units=str(arrow.units),
        shaft_width_px=float(line_px),
        halo_width_px=float(halo_px),
    )
    return glyph, False, False


def _project_single_arc(
    arc: TorqueArcGlyph,
    projector: ImageProjector,
    head_px: float,
    line_px: float,
    halo_px: float,
    w: int,
    h: int,
) -> tuple[ProjectedTorqueArcGlyph | None, bool, bool]:
    """Project a single torque arc glyph. Returns (glyph_or_none, is_behind, is_out)."""
    if not arc.polyline_m:
        return None, False, True

    n = len(arc.polyline_m)
    pts = np.vstack(
        [
            np.array(arc.polyline_m, float),
            np.array([arc.head_tip_m, arc.head_base_m], float),
        ]
    )
    pix, val = projector.project(pts)
    if not (val[0] and val[n - 1]):
        return None, True, False

    segs = [(float(pix[i][0]), float(pix[i][1])) for i in range(n) if val[i]]
    head_poly = (
        _compute_triangle_head(pix[n], pix[n + 1], head_px)
        if (val[n] and val[n + 1])
        else None
    )

    in_frame = any(
        _segment_intersects_rect(segs[i], segs[i + 1], w, h)
        for i in range(len(segs) - 1)
    ) or (
        head_poly is not None
        and any(0 <= p[0] < w and 0 <= p[1] < h for p in head_poly)
    )
    if not in_frame:
        return None, False, True

    color_hex = FORCE_KIND_PALETTE.get(arc.kind, _rgba_to_hex(arc.rgba))
    glyph = ProjectedTorqueArcGlyph(
        polyline_px=tuple(segs),
        head_poly_px=head_poly,
        rgba=arc.rgba,
        color_hex=color_hex,
        kind=str(arc.kind),
        label=str(arc.label),
        magnitude=float(arc.magnitude),
        units=str(arc.units),
        shaft_width_px=float(line_px),
        halo_width_px=float(halo_px),
    )
    return glyph, False, False


def project_glyphs(
    glyphs: GlyphSet,
    projector: ImageProjector,
    *,
    style: VideoGlyphStyle | None = None,
    image_size_px: tuple[int, int] | None = None,
) -> ProjectedGlyphSet:
    """Project a GlyphSet into 2D pixel space for video overlay rendering.

    Args:
        glyphs: Input 3D GlyphSet in camera world coordinates (ADR-0041).
        projector: Calibrated pinhole or hypothesis camera projector.
        style: Optional visual style specifying line and head pixel sizes.
        image_size_px: Optional (width, height) override in pixels.

    Returns:
        ProjectedGlyphSet containing 2D projected arrows and torque arcs.
    """
    from src.shared.python.force_overlay.renderers.opencv_glyphs import (
        VideoGlyphReceipt,
        VideoGlyphStyle,
    )

    w, h = _resolve_image_size(projector, image_size_px)
    s = style or VideoGlyphStyle()
    line_px = s.resolve_line_px(h)
    halo_px = s.resolve_halo_px(line_px)
    head_px = s.resolve_head_px(line_px)

    arrows: list[ProjectedArrowGlyph] = []
    skipped_behind = 0
    skipped_out = 0

    for arrow in glyphs.arrows:
        glyph, is_behind, is_out = _project_single_arrow(
            arrow, projector, float(head_px), float(line_px), float(halo_px), w, h
        )
        if is_behind:
            skipped_behind += 1
        elif is_out:
            skipped_out += 1
        elif glyph is not None:
            arrows.append(glyph)

    arcs: list[ProjectedTorqueArcGlyph] = []
    for arc in glyphs.torque_arcs:
        arc_glyph, is_behind, is_out = _project_single_arc(
            arc, projector, float(head_px), float(line_px), float(halo_px), w, h
        )
        if is_behind:
            skipped_behind += 1
        elif is_out:
            skipped_out += 1
        elif arc_glyph is not None:
            arcs.append(arc_glyph)

    receipt = VideoGlyphReceipt(
        drawn=len(arrows) + len(arcs),
        skipped_behind_camera=skipped_behind,
        skipped_out_of_frame=skipped_out,
        unavailable_labels=tuple(glyphs.legend.unavailable_labels),
    )

    return ProjectedGlyphSet(
        time_s=glyphs.time_s,
        image_size_px=(w, h),
        arrows=tuple(arrows),
        torque_arcs=tuple(arcs),
        legend=glyphs.legend,
        receipt=receipt,
    )
