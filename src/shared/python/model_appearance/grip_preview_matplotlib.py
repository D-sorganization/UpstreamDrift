"""Equal-aspect matplotlib previews for grip and hand meshes (visual evidence only)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GripMeshPreviewStyle:
    """Lambert-shaded mesh colours for grip/hand evidence stills."""

    base_rgb: tuple[float, float, float] = (0.72, 0.58, 0.48)
    ambient: float = 0.35
    diffuse: float = 0.55
    specular: float = 0.1
    light_direction: tuple[float, float, float] = (0.35, 0.45, 1.0)
    edge_rgba: tuple[float, float, float, float] = (0.95, 0.92, 0.88, 0.28)
    edge_linewidth: float = 0.18
    alpha: float = 1.0
    margin_fraction: float = 1.08
    min_half_width_m: float = 0.045


_DEFAULT_STYLE = GripMeshPreviewStyle()


def _unit(vec: Array) -> Array:
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return vec / n


def equal_half_width_from_points(
    points: Array,
    *,
    margin_fraction: float = _DEFAULT_STYLE.margin_fraction,
    min_half_width_m: float = _DEFAULT_STYLE.min_half_width_m,
    centre: Array | None = None,
) -> tuple[Array, float]:
    """Return (centre, half_width) using one span for all three axes.

      Preconditions: ``points`` is an ``(N, 3)`` finite array with ``N >= 1``.
    Postcondition: ``half_width`` is at least ``min_half_width_m`` and equals the
      maximum axis half-range times ``margin_fraction``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")
    if centre is None:
        centre = pts.mean(axis=0)
    else:
        centre = np.asarray(centre, dtype=float)
        if centre.shape != (3,):
            raise ValueError("centre must be length 3")
    half_ranges = 0.5 * np.ptp(pts, axis=0)
    half = float(max(half_ranges.max(), min_half_width_m) * margin_fraction)
    return centre, half


def apply_equal_axis_limits(
    ax: Axes,
    points: Array,
    *,
    centre: Array | None = None,
    style: GripMeshPreviewStyle = _DEFAULT_STYLE,
) -> float:
    """Crop all three axes to the same half-width around ``centre``.

    Postcondition: each axis span equals ``2 * half_width`` (within float
    round-trip). Returns ``half_width`` for tests and callers.
    """
    mid, half = equal_half_width_from_points(
        points,
        margin_fraction=style.margin_fraction,
        min_half_width_m=style.min_half_width_m,
        centre=centre,
    )
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))
    return half


def axis_half_widths(ax: Axes) -> tuple[float, float, float]:
    """Read back axis half-widths from a matplotlib 3D axes."""
    spans = []
    for getter in (ax.get_xlim, ax.get_ylim, ax.get_zlim):
        lo, hi = getter()
        spans.append(0.5 * float(hi - lo))
    return spans[0], spans[1], spans[2]


def face_lambert_colours(
    vertices: Array,
    faces: Array,
    style: GripMeshPreviewStyle = _DEFAULT_STYLE,
) -> Array:
    """RGB face colours from triangle normals and a single light direction."""
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(e1, e2)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-12)
    normals = normals / lens
    light = _unit(np.asarray(style.light_direction, dtype=float))
    ndotl = np.clip(normals @ light, 0.0, 1.0)
    shade = style.ambient + style.diffuse * ndotl + style.specular * ndotl**8
    shade = np.clip(shade, 0.0, 1.0)[:, None]
    base = np.asarray(style.base_rgb, dtype=float)
    return np.clip(base * shade, 0.0, 1.0)


def add_mesh_preview(
    ax: Axes,
    mesh: Mesh,
    *,
    style: GripMeshPreviewStyle = _DEFAULT_STYLE,
) -> Poly3DCollection:
    """Draw ``mesh`` with Lambert shading and light edges on a 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri = mesh.vertices[mesh.faces]
    colours = face_lambert_colours(mesh.vertices, mesh.faces, style=style)
    collection = Poly3DCollection(
        tri,
        facecolors=colours,
        edgecolors=[style.edge_rgba] * len(tri),
        linewidths=style.edge_linewidth,
        alpha=style.alpha,
    )
    ax.add_collection3d(collection)
    return collection


def preview_figure_size(half_width_m: float, *, dpi: int = 120) -> tuple[float, float]:
    """Figure size in inches for a square-framed hand/grip crop."""
    span_in = max(2.4, min(5.5, half_width_m * 2.0 / 0.065))
    return span_in, span_in
