"""MeshCat 3D glyph renderer for force arrows and torque arcs (FTO-5, #11290).

Renders a GlyphSet into a MeshCat visualizer tree using real 3D geometry:
  - Arrows: a cylinder shaft plus a cone head (radius_top=0).
  - Torque arcs: a polyline chain of cylinder segments plus a cone head.

Implements the MeshcatSink protocol so both meshcat-python and Drake MeshCat
are supported without code duplication.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.shared.python.force_overlay.glyphs import ArrowGlyph, GlyphSet, TorqueArcGlyph


@runtime_checkable
class MeshcatSink(Protocol):
    """Minimal sink interface abstracting MeshCat visualizer implementations."""

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        """Create or update a cylinder or cone geometry at path."""
        ...

    def set_transform(self, path: str, matrix4x4: npt.NDArray[np.float64]) -> None:
        """Set the 4x4 affine transform of the node at path."""
        ...

    def delete(self, path: str) -> None:
        """Delete the node and all its children at path."""
        ...


def align_y_to(
    direction: Sequence[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute 3x3 rotation matrix mapping local +y (0, 1, 0) to unit vector direction.

    Uses Rodrigues' rotation formula. Explicitly handles identity (+y) and
    antiparallel singularity (-y). Always returns a proper rotation (det R = +1).
    """
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    norm_d = float(np.linalg.norm(d))
    if norm_d < 1e-12:
        return np.eye(3, dtype=np.float64)
    d = d / norm_d

    d_x, d_y, d_z = float(d[0]), float(d[1]), float(d[2])

    # Case 1: Already aligned with +y
    if d_y >= 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)

    # Case 2: Antiparallel (-y singularity) -> 180 deg rotation about X
    if d_y <= -1.0 + 1e-8:
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )

    # Case 3: General Rodrigues rotation from y=(0,1,0) to d
    # Axis v = y x d = (d_z, 0, -d_x)
    # v_cross = [[0, d_x, 0], [-d_x, 0, -d_z], [0, d_z, 0]]
    v_cross = np.array(
        [
            [0.0, d_x, 0.0],
            [-d_x, 0.0, -d_z],
            [0.0, d_z, 0.0],
        ],
        dtype=np.float64,
    )
    r = np.eye(3, dtype=np.float64) + v_cross + (v_cross @ v_cross) / (1.0 + d_y)
    return r


def _make_segment_transform(
    p_start: Sequence[float], p_end: Sequence[float]
) -> tuple[float, npt.NDArray[np.float64]]:
    """Compute segment length and 4x4 matrix mapping centered +y cylinder to segment."""
    p0 = np.asarray(p_start, dtype=np.float64)
    p1 = np.asarray(p_end, dtype=np.float64)
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-12:
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = p0
        return 0.0, t

    rot = align_y_to(diff / length)
    mid = 0.5 * (p0 + p1)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rot
    t[:3, 3] = mid
    return length, t


class MeshcatGlyphRenderer:
    """Renderer mapping GlyphSet to real 3D cylinder and cone arrows in MeshCat."""

    def __init__(self, sink: MeshcatSink, root: str = "/force_overlay") -> None:
        if not isinstance(sink, MeshcatSink):
            raise TypeError(
                f"sink must implement MeshcatSink protocol, got {type(sink).__name__}"
            )
        self._sink = sink
        self._root = root.rstrip("/")
        self._active_paths_by_label: dict[str, set[str]] = {}
        self._geometry_cache: dict[
            str, tuple[float, float, float, tuple[float, float, float, float]]
        ] = {}

    def update(self, glyphs: GlyphSet) -> None:
        """Update MeshCat scene with current glyph geometry, caching unchanged shapes."""
        current_labels: set[str] = set()
        new_active_paths_by_label: dict[str, set[str]] = {}

        # 1. Render Force Arrows
        for arrow in glyphs.arrows:
            label = arrow.label
            current_labels.add(label)
            label_paths: set[str] = set()

            # Shaft cylinder
            shaft_path = f"{self._root}/{label}/shaft"
            shaft_len, shaft_tf = _make_segment_transform(
                arrow.tail_m, arrow.head_base_m
            )
            if shaft_len > 1e-6:
                label_paths.add(shaft_path)
                geom_key = (
                    round(shaft_len, 6),
                    round(arrow.shaft_radius_m, 6),
                    round(arrow.shaft_radius_m, 6),
                    arrow.rgba,
                )
                if self._geometry_cache.get(shaft_path) != geom_key:
                    self._sink.set_cylinder(
                        shaft_path,
                        shaft_len,
                        arrow.shaft_radius_m,
                        arrow.shaft_radius_m,
                        arrow.rgba,
                    )
                    self._geometry_cache[shaft_path] = geom_key
                self._sink.set_transform(shaft_path, shaft_tf)

            # Head cone (radius_top = 0.0)
            head_path = f"{self._root}/{label}/head"
            head_len, head_tf = _make_segment_transform(arrow.head_base_m, arrow.tip_m)
            if head_len > 1e-6:
                label_paths.add(head_path)
                geom_key = (
                    round(head_len, 6),
                    0.0,
                    round(arrow.head_radius_m, 6),
                    arrow.rgba,
                )
                if self._geometry_cache.get(head_path) != geom_key:
                    self._sink.set_cylinder(
                        head_path,
                        head_len,
                        0.0,
                        arrow.head_radius_m,
                        arrow.rgba,
                    )
                    self._geometry_cache[head_path] = geom_key
                self._sink.set_transform(head_path, head_tf)

            new_active_paths_by_label[label] = label_paths

        # 2. Render Torque Arcs
        for arc in glyphs.torque_arcs:
            label = arc.label
            current_labels.add(label)
            label_paths = new_active_paths_by_label.get(label, set())

            poly = arc.polyline_m
            head_len_est = float(
                np.linalg.norm(np.array(arc.head_tip_m) - np.array(arc.head_base_m))
            )
            arc_tube_radius = (
                max(0.002, head_len_est * 0.15)
                if head_len_est > 0
                else max(0.002, arc.radius_m * 0.04)
            )

            for i in range(len(poly) - 1):
                seg_path = f"{self._root}/{label}/arc/{i}"
                seg_len, seg_tf = _make_segment_transform(poly[i], poly[i + 1])
                if seg_len > 1e-6:
                    label_paths.add(seg_path)
                    geom_key = (
                        round(seg_len, 6),
                        round(arc_tube_radius, 6),
                        round(arc_tube_radius, 6),
                        arc.rgba,
                    )
                    if self._geometry_cache.get(seg_path) != geom_key:
                        self._sink.set_cylinder(
                            seg_path,
                            seg_len,
                            arc_tube_radius,
                            arc_tube_radius,
                            arc.rgba,
                        )
                        self._geometry_cache[seg_path] = geom_key
                    self._sink.set_transform(seg_path, seg_tf)

            # Arc cone head
            head_path = f"{self._root}/{label}/head"
            head_len, head_tf = _make_segment_transform(arc.head_base_m, arc.head_tip_m)
            if head_len > 1e-6:
                label_paths.add(head_path)
                cone_base_r = arc_tube_radius * 2.5
                geom_key = (
                    round(head_len, 6),
                    0.0,
                    round(cone_base_r, 6),
                    arc.rgba,
                )
                if self._geometry_cache.get(head_path) != geom_key:
                    self._sink.set_cylinder(
                        head_path,
                        head_len,
                        0.0,
                        cone_base_r,
                        arc.rgba,
                    )
                    self._geometry_cache[head_path] = geom_key
                self._sink.set_transform(head_path, head_tf)

            new_active_paths_by_label[label] = label_paths

        # 3. Clean up disappeared labels and unused paths
        for old_label, old_paths in self._active_paths_by_label.items():
            current_paths = new_active_paths_by_label.get(old_label, set())
            dead_paths = old_paths - current_paths
            for dead_path in dead_paths:
                self._sink.delete(dead_path)
                self._geometry_cache.pop(dead_path, None)

        self._active_paths_by_label = new_active_paths_by_label

    def clear(self) -> None:
        """Clear all force overlay nodes from MeshCat."""
        self._sink.delete(self._root)
        self._active_paths_by_label.clear()
        self._geometry_cache.clear()


class MeshcatPythonSink:
    """Sink adapting meshcat-python (meshcat.Visualizer) to MeshcatSink protocol."""

    def __init__(self, visualizer: Any) -> None:
        self._vis = visualizer

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        import meshcat.geometry as mcg

        clean_path = path.strip("/")
        node = self._vis[clean_path]
        r, g, b, a = rgba
        rgb_int = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
        material = mcg.MeshLambertMaterial(
            color=rgb_int,
            opacity=float(a),
            transparent=(a < 1.0),
        )
        geom = mcg.Cylinder(
            height=length_m,
            radiusTop=radius_top_m,
            radiusBottom=radius_bottom_m,
        )
        node.set_object(geom, material)

    def set_transform(self, path: str, matrix4x4: npt.NDArray[np.float64]) -> None:
        clean_path = path.strip("/")
        self._vis[clean_path].set_transform(matrix4x4)

    def delete(self, path: str) -> None:
        clean_path = path.strip("/")
        self._vis[clean_path].delete()


def legend_text(glyphs: GlyphSet) -> str:
    """Format single-line summary text for host GUI status bars."""
    leg = glyphs.legend
    parts: list[str] = []
    if leg.force_reference_n is not None and leg.force_reference_length_m is not None:
        parts.append(
            f"Ref Force: {leg.force_reference_n:.0f} N ({leg.force_reference_length_m:.2f} m)"
        )
    if (
        leg.torque_reference_nm is not None
        and leg.torque_reference_radius_m is not None
    ):
        parts.append(
            f"Ref Torque: {leg.torque_reference_nm:.0f} N*m (r={leg.torque_reference_radius_m:.2f} m)"
        )
    if leg.engine:
        parts.append(f"Engine: {leg.engine}")
    if leg.kinds_present:
        parts.append(f"Kinds: {', '.join(leg.kinds_present)}")
    return " | ".join(parts) if parts else "Force Overlay"
