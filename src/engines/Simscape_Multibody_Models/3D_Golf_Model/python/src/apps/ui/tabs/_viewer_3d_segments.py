"""User-defined body-segment rendering for :mod:`viewer_3d_tab`.

This private controller owns shape construction, fitting, artist lifecycle,
and per-frame visibility updates.  Keeping those responsibilities together
lets :class:`Viewer3DTab` remain a thin UI/playback facade while preserving its
existing public segment API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.shared.python.body_part_viz import SegmentVizSpec, ShapeTheme
from src.shared.python.body_part_viz.asset_library import ShapeLibrary
from src.shared.python.body_part_viz.fitters import (
    BetweenTwoMarkersFitter,
    ClusterKabschFitter,
    ProcrustesAnisotropicFitter,
)
from src.shared.python.body_part_viz.renderers import MatplotlibRenderer
from src.shared.python.body_part_viz.shapes import (
    CapsuleShape,
    CylinderShape,
    EllipsoidShape,
    LineShape,
    MeshShape,
)
from src.shared.python.logging_pkg.logging_config import get_logger

from ...core.models import C3DDataModel
from ...services.segment_set_io import SegmentSpec, spec_v1_to_v2

_LOGGER = get_logger(__name__)

_GROUP_COLORS: dict[str, tuple[float, float, float, float]] = {
    "pelvis": (0.20, 0.55, 0.85, 1.0),
    "torso": (0.45, 0.30, 0.75, 1.0),
    "head": (0.85, 0.60, 0.20, 1.0),
    "left_arm": (0.30, 0.75, 0.40, 1.0),
    "right_arm": (0.20, 0.50, 0.30, 1.0),
    "left_leg": (0.85, 0.30, 0.30, 1.0),
    "right_leg": (0.65, 0.20, 0.20, 1.0),
    "auto": (0.30, 0.30, 0.30, 1.0),
    "default": (0.30, 0.30, 0.30, 1.0),
}


def _rgba_to_hex(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _a = rgba
    rh = int(round(r * 255))
    gh = int(round(g * 255))
    bh = int(round(b * 255))
    return f"#{rh:02x}{gh:02x}{bh:02x}"


def _build_shape_from_spec(
    spec: SegmentVizSpec,
    *,
    library: ShapeLibrary | None,
) -> Any:
    """Construct a body-part shape, or return ``None`` if unavailable."""
    kind = spec.shape_kind
    params = spec.shape_params
    if kind == "line":
        return LineShape(length=float(params.get("length", 1.0)))
    if kind == "cylinder":
        return CylinderShape(
            length=float(params.get("length", 1.0)),
            radius=float(params.get("radius", 0.015)),
            n_facets=int(params.get("n_facets", 16)),
        )
    if kind == "ellipsoid":
        return EllipsoidShape(
            a=float(params["a"]),
            b=float(params["b"]),
            c=float(params["c"]),
            n_lon=int(params.get("n_lon", 16)),
            n_lat=int(params.get("n_lat", 8)),
        )
    if kind == "capsule":
        return CapsuleShape(
            length=float(params.get("length", 1.0)),
            radius=float(params.get("radius", 0.015)),
            n_facets=int(params.get("n_facets", 16)),
            n_lat=int(params.get("n_lat", 8)),
        )
    if kind == "mesh_file":
        try:
            return MeshShape.load(
                str(params["path"]),
                max_vertices=int(params.get("max_vertices", 5000)),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            _LOGGER.warning("could not load mesh %s: %s", params.get("path"), exc)
            return None
    if kind == "library_shape":
        if library is None:
            return None
        try:
            return library.get(str(params["shape_id"]))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            _LOGGER.warning(
                "library shape %s not available: %s", params.get("shape_id"), exc
            )
            return None
    return None


def _fitter_for_kind(kind: str) -> Any:
    if kind == "between_two":
        return BetweenTwoMarkersFitter()
    if kind == "cluster_kabsch":
        return ClusterKabschFitter()
    if kind == "procrustes_anisotropic":
        return ProcrustesAnisotropicFitter()
    raise ValueError(f"unknown fitter_kind {kind!r}")


class UserSegmentRenderer:
    """Own the user-defined segment specs and their Matplotlib artists."""

    def __init__(
        self,
        transform_positions: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        if not callable(transform_positions):
            raise TypeError("transform_positions must be callable")
        self._transform_positions = transform_positions
        self._segments: tuple[SegmentVizSpec, ...] = ()
        self._renderer: MatplotlibRenderer | None = None
        self._render_entries: list[tuple[str | None, str, np.ndarray | None]] = []
        self._shape_library: ShapeLibrary | None = None

    def set_segments(
        self,
        segments: tuple[SegmentSpec | SegmentVizSpec, ...],
    ) -> None:
        """Validate and store segment specs, converting legacy v1 entries."""
        if segments is None:
            raise ValueError("segments must be provided (use () for an empty set)")
        viz_specs: list[SegmentVizSpec] = []
        for spec in segments:
            if isinstance(spec, SegmentVizSpec):
                viz_specs.append(spec)
            elif isinstance(spec, SegmentSpec):
                viz_specs.append(spec_v1_to_v2(spec))
            else:
                raise TypeError(
                    "segments entries must be SegmentSpec or SegmentVizSpec; "
                    f"got {type(spec).__name__}"
                )
        self._segments = tuple(viz_specs)

    @property
    def cylinder_count(self) -> int:
        """Return the number of allocated non-line segment artists."""
        return sum(
            1
            for handle, kind, _valid in self._render_entries
            if handle is not None and kind != "line"
        )

    @property
    def line_segment_count(self) -> int:
        """Return the number of allocated line segment artists."""
        return sum(
            1
            for handle, kind, _valid in self._render_entries
            if handle is not None and kind == "line"
        )

    def clear(self) -> None:
        """Clear all allocated artists while retaining the segment specs."""
        if self._renderer is not None:
            self._renderer.clear()
        self._renderer = None
        self._render_entries = []

    def rebuild(self, ax: Any, model: C3DDataModel | None, n_frames: int) -> None:
        """Rebuild user-segment artists for the active axes and model."""
        if n_frames < 0:
            raise ValueError(f"n_frames must be non-negative, got {n_frames}")
        self.clear()
        if ax is None:
            return
        self._renderer = MatplotlibRenderer(ax)
        if not self._segments or model is None or n_frames <= 0:
            return

        library = self._resolve_library()
        for spec in self._segments:
            entry = self._build_render_entry(spec, model, n_frames, library)
            self._render_entries.append(entry)

    def update_frame(self, frame: int, n_frames: int) -> None:
        """Update visible user-segment artists for one validated frame."""
        if self._renderer is None or not self._segments or n_frames <= 0:
            return
        if not 0 <= frame < n_frames:
            return
        for entry, spec in zip(self._render_entries, self._segments, strict=False):
            handle, _kind, valid_mask = entry
            if handle is None:
                continue
            frame_valid = True
            if valid_mask is not None and frame < len(valid_mask):
                frame_valid = bool(valid_mask[frame])
            visible = bool(spec.visible) and frame_valid
            try:
                self._renderer.set_visible(handle, visible)
                if visible:
                    self._renderer.update_frame(handle, frame)
            except (KeyError, IndexError, TypeError) as exc:
                _LOGGER.warning("renderer.update_frame failed: %s", exc)

    def _resolve_library(self) -> ShapeLibrary | None:
        if self._shape_library is None:
            try:
                self._shape_library = ShapeLibrary.default()
            except (FileNotFoundError, ValueError) as exc:
                _LOGGER.warning("default shape library unavailable: %s", exc)
                return None
        return self._shape_library

    def _markers_xyz(
        self,
        names: tuple[str, ...],
        model: C3DDataModel,
        n_frames: int,
    ) -> dict[str, np.ndarray] | None:
        out: dict[str, np.ndarray] = {}
        for name in names:
            marker = model.markers.get(name)
            if marker is None or marker.position.size == 0:
                return None
            positions = np.asarray(marker.position, dtype=float)
            if positions.shape[0] < n_frames:
                padded: np.ndarray = np.full((n_frames, 3), np.nan, dtype=float)
                padded[: positions.shape[0]] = positions
                positions = padded
            elif positions.shape[0] > n_frames:
                positions = positions[:n_frames]
            out[name] = self._transform_positions(positions)
        return out

    @staticmethod
    def _theme_for_spec(spec: SegmentVizSpec) -> ShapeTheme:
        rgba = _GROUP_COLORS.get(spec.theme.group, _GROUP_COLORS["auto"])
        if spec.theme.color == "#1f77b4" and spec.theme.group in _GROUP_COLORS:
            color = _rgba_to_hex(rgba)
            return ShapeTheme(
                color=color,
                opacity=spec.theme.opacity,
                edge_color=color,
                edge_width=spec.theme.edge_width,
                flat_shaded=spec.theme.flat_shaded,
                group=spec.theme.group,
            )
        return spec.theme

    def _build_render_entry(
        self,
        spec: SegmentVizSpec,
        model: C3DDataModel,
        n_frames: int,
        library: ShapeLibrary | None,
    ) -> tuple[str | None, str, np.ndarray | None]:
        shape = _build_shape_from_spec(spec, library=library)
        if shape is None:
            return (None, spec.shape_kind, None)
        markers_xyz = self._markers_xyz(spec.binding.marker_names, model, n_frames)
        if markers_xyz is None:
            return (None, spec.shape_kind, None)
        try:
            fitter = _fitter_for_kind(spec.fitter_kind)
        except ValueError:
            return (None, spec.shape_kind, None)
        try:
            effective_binding = spec.binding
            if not effective_binding.rest_dimensions and shape.rest_dimensions:
                effective_binding = type(spec.binding)(
                    kind=spec.binding.kind,
                    marker_names=spec.binding.marker_names,
                    rest_dimensions=(float(shape.rest_dimensions[0]),),
                    rest_orientation_quat=spec.binding.rest_orientation_quat,
                )
            fitted = fitter.fit(shape, effective_binding, markers_xyz)
        except (KeyError, TypeError, ValueError) as exc:
            _LOGGER.warning(
                "fit failed for segment %s: %s", spec.binding.marker_names, exc
            )
            return (None, spec.shape_kind, None)

        renderer = self._renderer
        assert renderer is not None
        try:
            handle = renderer.add_shape(shape, fitted, self._theme_for_spec(spec))
        except (TypeError, ValueError) as exc:
            _LOGGER.warning(
                "renderer.add_shape failed for %s: %s", spec.shape_kind, exc
            )
            return (None, spec.shape_kind, None)
        if not spec.visible:
            renderer.set_visible(handle, False)
        valid_mask = getattr(fitted, "valid_mask", None)
        return (handle, spec.shape_kind, valid_mask)
