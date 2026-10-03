"""Canonical Matplotlib 3D force and torque glyph renderer (ADR-0052, #11292)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
)
from src.shared.python.motion_matching.diagnostics._skeleton_render import (
    equalize_3d_axes,
)
from src.shared.python.plot_style import FORCE_KIND_PALETTE

__all__ = [
    "draw_glyphs_3d",
    "draw_legend",
    "equalize_3d_axes",
]


def _build_cone_facets(
    base_center: Sequence[float],
    apex: Sequence[float],
    radius: float,
    num_facets: int = 12,
) -> list[list[np.ndarray]]:
    """Build triangular facets and base disc for a 3D cone."""
    c = np.asarray(base_center, dtype=float)
    a = np.asarray(apex, dtype=float)
    axis = a - c
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or radius <= 0.0:
        return []

    u = axis / norm
    v = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v, u)
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm < 1e-9:
        return []
    e1 = e1 / e1_norm
    e2 = np.cross(u, e1)

    angles = np.linspace(0.0, 2.0 * math.pi, num_facets, endpoint=False)
    circle_pts = [c + radius * (math.cos(th) * e1 + math.sin(th) * e2) for th in angles]

    facets: list[list[np.ndarray]] = []
    for k in range(num_facets):
        next_k = (k + 1) % num_facets
        facets.append([circle_pts[k], circle_pts[next_k], a])
    facets.append(circle_pts)
    return facets


def draw_glyphs_3d(
    ax: Any,
    glyphs: GlyphSet,
    *,
    linewidth_pt: float = 2.0,
    halo: bool = True,
) -> list[Artist]:
    """Render 3D force arrows and torque arcs on a Matplotlib 3D axes.

    Parameters
    ----------
    ax
        Matplotlib 3D axes (projection='3d').
    glyphs
        Deterministic glyph set to draw.
    linewidth_pt
        Stroke width for shafts and arcs in points.
    halo
        If True, renders a 1.5x width dark #202020 underlay with alpha 0.6.

    Returns
    -------
    list[Artist]
        Created matplotlib artists supporting .remove().
    """
    if getattr(ax, "name", "") != "3d" and not hasattr(ax, "set_zlim"):
        raise ValueError("ax must be a matplotlib 3D axes (projection='3d')")
    if not isinstance(glyphs, GlyphSet):
        raise TypeError(f"glyphs must be GlyphSet, got {type(glyphs)}")

    artists: list[Artist] = []

    # 1. Force arrows
    for arrow in glyphs.arrows:
        tail = np.asarray(arrow.tail_m, dtype=float)
        head_base = np.asarray(arrow.head_base_m, dtype=float)
        tip = np.asarray(arrow.tip_m, dtype=float)

        if halo:
            halo_shaft = ax.plot(
                [tail[0], head_base[0]],
                [tail[1], head_base[1]],
                [tail[2], head_base[2]],
                color="#202020",
                linewidth=linewidth_pt * 1.5,
                alpha=0.6,
                solid_capstyle="round",
            )[0]
            artists.append(halo_shaft)

        shaft = ax.plot(
            [tail[0], head_base[0]],
            [tail[1], head_base[1]],
            [tail[2], head_base[2]],
            color=arrow.rgba,
            linewidth=linewidth_pt,
            solid_capstyle="round",
        )[0]
        artists.append(shaft)

        cone_facets = _build_cone_facets(head_base, tip, arrow.head_radius_m, 12)
        if cone_facets:
            cone = Poly3DCollection(
                cone_facets,
                facecolors=arrow.rgba,
                edgecolors=arrow.rgba,
                alpha=arrow.rgba[3] if len(arrow.rgba) > 3 else 1.0,
            )
            ax.add_collection3d(cone)
            artists.append(cone)

    # 2. Torque arcs
    for arc in glyphs.torque_arcs:
        pts = np.asarray(arc.polyline_m, dtype=float)
        if len(pts) >= 2:
            if halo:
                halo_arc = ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color="#202020",
                    linewidth=linewidth_pt * 1.5,
                    alpha=0.6,
                    solid_capstyle="round",
                )[0]
                artists.append(halo_arc)

            fg_arc = ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=arc.rgba,
                linewidth=linewidth_pt,
                solid_capstyle="round",
            )[0]
            artists.append(fg_arc)

        hb = np.asarray(arc.head_base_m, dtype=float)
        ht = np.asarray(arc.head_tip_m, dtype=float)
        head_len = float(np.linalg.norm(ht - hb))
        cone_facets = _build_cone_facets(hb, ht, max(1e-4, head_len * 0.35), 12)
        if cone_facets:
            cone = Poly3DCollection(
                cone_facets,
                facecolors=arc.rgba,
                edgecolors=arc.rgba,
                alpha=arc.rgba[3] if len(arc.rgba) > 3 else 1.0,
            )
            ax.add_collection3d(cone)
            artists.append(cone)

    return artists


def draw_legend(ax: Axes, legend: LegendSpec) -> Axes:
    """Render an inset legend with reference scales, kinds, and unavailable notes.

    Parameters
    ----------
    ax
        Host Matplotlib axes.
    legend
        Legend specification containing reference scales and kinds.

    Returns
    -------
    Axes
        The inset axes containing the legend items.
    """
    inset = ax.inset_axes([0.02, 0.02, 0.36, 0.30])
    inset.set_facecolor("#181818")
    inset.patch.set_alpha(0.85)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#404040")

    lines_text: list[str] = []
    if legend.force_reference_n is not None:
        lines_text.append(f"Force Ref: {legend.force_reference_n:g} N")
    if legend.torque_reference_nm is not None:
        lines_text.append(f"Torque Ref: {legend.torque_reference_nm:g} N·m")

    y = 0.88
    for line in lines_text:
        inset.text(0.06, y, line, color="#ffffff", fontsize=8, weight="bold")
        y -= 0.22

    # Kind swatches
    if legend.kinds_present:
        swatch_x = 0.06
        for kind in legend.kinds_present:
            col = FORCE_KIND_PALETTE.get(kind, "#888888")
            inset.plot([swatch_x, swatch_x + 0.12], [y, y], color=col, linewidth=3.0)
            kind_name = kind.value if isinstance(kind, WrenchKind) else str(kind)
            inset.text(
                swatch_x + 0.15, y - 0.03, kind_name[:8], color="#cccccc", fontsize=7
            )
            swatch_x += 0.45
            if swatch_x > 0.8:
                swatch_x = 0.06
                y -= 0.18

    if legend.unavailable_labels:
        inset.text(
            0.06,
            0.10,
            f"Unavailable: {', '.join(legend.unavailable_labels)}",
            color="#ff8888",
            fontsize=7,
        )

    return inset
