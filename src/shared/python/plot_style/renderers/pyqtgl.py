"""GPU-accelerated marker renderer using ``pyqtgraph.opengl``.

This module implements
:class:`~src.shared.python.plot_style.contracts.MarkerRenderer` against
``pyqtgraph.opengl``. Built-in 0-D shapes (``SPHERE``, ``POINT``,
``PLUS``) are rendered with a single :class:`GLScatterPlotItem` per
``add_markers`` call; 3-D shapes (``CUBE``, ``CROSS``, ``STAR``,
``DIAMOND``, ``CUSTOM_MESH``) are rendered with one
:class:`GLMeshItem` per marker, instantiated from the unit-radius mesh
provided by :mod:`plot_style.shapes`.

Optional dependency policy
--------------------------
``pyqtgraph`` and its ``opengl`` submodule are imported lazily so
``import plot_style`` never pulls them in. Install the optional extra
to enable this backend::

    pip install upstream-drift[body-part-viz-gl]
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from ..colors import StaticColor
from ..markers import CustomMeshSpec, MarkerShape, MarkerStyle
from ..shapes import (
    CrossMarker,
    CubeMarker,
    DiamondMarker,
    SphereMarker,
    StarMarker,
)

__all__ = ["PyQtGLMarkerRenderer"]


# Shapes rendered as point primitives via GLScatterPlotItem.
_SCATTER_SHAPES: frozenset[MarkerShape] = frozenset(
    {MarkerShape.SPHERE, MarkerShape.POINT, MarkerShape.PLUS}
)


def _import_pyqtgraph_opengl() -> Any:
    """Lazily import ``pyqtgraph.opengl``.

    Raises
    ------
    ImportError
        With an actionable hint pointing at the ``body-part-viz-gl``
        extra if the optional dependency is missing.
    """
    try:
        import pyqtgraph.opengl as gl  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "pyqtgraph.opengl is required for PyQtGLMarkerRenderer. "
            "Install the optional extra: pip install upstream-drift[body-part-viz-gl]"
        ) from exc
    return gl


def _mesh_for_shape(style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` for a non-scatter built-in shape.

    The returned vertices are scaled to ``size_px / 2``.
    """
    shape = style.shape
    if shape is MarkerShape.CUBE:
        return CubeMarker().mesh(style)
    if shape is MarkerShape.CROSS:
        return CrossMarker().mesh(style)
    if shape is MarkerShape.STAR:
        return StarMarker().mesh(style)
    if shape is MarkerShape.DIAMOND:
        return DiamondMarker().mesh(style)
    if shape is MarkerShape.SPHERE:
        return SphereMarker().mesh(style)
    raise ValueError(f"no mesh available for shape {shape!r}")


def _custom_mesh_to_arrays(
    spec: CustomMeshSpec, size_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a :class:`CustomMeshSpec` to GL-ready vertex/face arrays.

    The spec's vertices are scaled by ``size_px / 2`` so that the
    bounding sphere convention (diameter == ``size_px``) is preserved.
    """
    radius = float(size_px) / 2.0
    verts = np.asarray(spec.vertices, dtype=np.float64) * radius
    faces = np.asarray(spec.faces, dtype=np.int64)
    return verts, faces


def _resolve_color_from_style(style: MarkerStyle) -> tuple[float, float, float, float]:
    """Resolve the style's ``fill_color`` to a single RGBA tuple.

    Only :class:`StaticColor` is supported here — palette and
    data-driven colors require per-marker resolution and must be
    supplied via the ``colors`` argument to :meth:`draw` /
    :meth:`add_markers`.
    """
    fill = style.fill_color
    rgba = fill.resolve(0, None)
    a = float(rgba[3]) * float(style.opacity)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), max(0.0, min(1.0, a)))


def _coerce_positions(positions: np.ndarray) -> np.ndarray:
    """Validate / normalise a positions ndarray to ``(T, M, 3)``.

    Accepts ``(T, 3)`` (single-marker trajectory across ``T`` frames, as
    documented by the :class:`MarkerRenderer` contract). Returns a
    ``(T, M, 3)`` ``float32`` view; a 2-D ``(T, 3)`` input becomes
    ``(T, 1, 3)`` so callers can still call ``update_frame`` for every
    frame in the trajectory.
    """
    if not isinstance(positions, np.ndarray):
        raise TypeError(
            f"positions must be numpy.ndarray; got {type(positions).__name__}"
        )
    if positions.ndim == 2:
        if positions.shape[1] != 3:
            raise ValueError(
                f"2-D positions must have shape (T, 3); got {positions.shape}"
            )
        # Per the MarkerRenderer contract a 2-D (T, 3) input is a
        # single-marker trajectory of length T, NOT a single frame with
        # T markers. Reshape to (T, 1, 3) so update_frame(handle, t)
        # works for every t in [0, T).
        arr = positions.reshape(positions.shape[0], 1, 3)
    elif positions.ndim == 3:
        if positions.shape[2] != 3:
            raise ValueError(
                f"3-D positions must have shape (T, M, 3); got {positions.shape}"
            )
        arr = positions
    else:
        raise ValueError(f"positions must have ndim 2 or 3; got ndim={positions.ndim}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _coerce_colors_for_n(
    colors: np.ndarray | None, n: int, style: MarkerStyle
) -> np.ndarray:
    """Build an ``(N, 4)`` RGBA float32 array.

    If ``colors`` is ``None``, broadcast the style's static fill color.
    """
    if colors is None:
        rgba = _resolve_color_from_style(style)
        out = np.tile(np.asarray(rgba, dtype=np.float32), (n, 1))
        return out
    arr = np.asarray(colors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"colors must have shape (N, 4); got {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"colors length {arr.shape[0]} != number of markers {n}")
    return np.ascontiguousarray(arr)


class _MarkerEntry:
    """Internal bookkeeping for one ``add_markers`` registration."""

    __slots__ = ("style", "positions", "colors", "items", "is_scatter")

    def __init__(
        self,
        style: MarkerStyle,
        positions: np.ndarray,
        colors: np.ndarray,
        items: list[Any],
        is_scatter: bool,
    ) -> None:
        self.style = style
        self.positions = positions  # (T, M, 3) float32
        self.colors = colors  # (M, 4) float32
        self.items = items  # 1 GLScatterPlotItem, or M GLMeshItems
        self.is_scatter = is_scatter


class PyQtGLMarkerRenderer:
    """GPU-accelerated marker renderer implementing :class:`MarkerRenderer`.

    Parameters
    ----------
    gl_view_widget:
        A ``pyqtgraph.opengl.GLViewWidget`` instance. The widget does
        not need to be visible — instantiating it in headless mode
        (``QT_QPA_PLATFORM=offscreen``) is sufficient for tests.
    """

    def __init__(self, gl_view_widget: Any) -> None:
        if gl_view_widget is None:
            raise TypeError("gl_view_widget must not be None")
        self._gl = _import_pyqtgraph_opengl()
        self._widget = gl_view_widget
        self._entries: dict[str, _MarkerEntry] = {}
        self._handle_counter = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_handle(self, prefix: str) -> str:
        self._handle_counter += 1
        return f"{prefix}#{self._handle_counter}"

    def _build_scatter_item(
        self,
        positions_frame: np.ndarray,
        style: MarkerStyle,
        colors: np.ndarray,
    ) -> Any:
        """Create a ``GLScatterPlotItem`` for the given frame."""
        return self._gl.GLScatterPlotItem(
            pos=positions_frame,
            color=colors,
            size=float(style.size_px),
            pxMode=True,
        )

    def _build_mesh_items(
        self,
        positions_frame: np.ndarray,
        style: MarkerStyle,
        colors: np.ndarray,
    ) -> list[Any]:
        """Create one ``GLMeshItem`` per marker for the given frame."""
        if style.shape is MarkerShape.CUSTOM_MESH:
            assert style.custom_mesh is not None  # validated by MarkerStyle
            unit_verts, faces = _custom_mesh_to_arrays(style.custom_mesh, style.size_px)
        else:
            unit_verts, faces = _mesh_for_shape(style)

        verts_f32 = np.ascontiguousarray(unit_verts, dtype=np.float32)
        faces_i32 = np.ascontiguousarray(faces, dtype=np.int32)

        items: list[Any] = []
        for i, pos in enumerate(positions_frame):
            translated = verts_f32 + np.asarray(pos, dtype=np.float32)
            mesh_data = self._gl.MeshData(vertexes=translated, faces=faces_i32)
            color = tuple(float(c) for c in colors[i])
            item = self._gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=False,
                color=color,
                shader="shaded",
                drawEdges=style.edge_width > 0.0,
            )
            items.append(item)
        return items

    # ------------------------------------------------------------------
    # Convenience: stateless one-shot draw
    # ------------------------------------------------------------------

    def draw(
        self,
        view: Any,
        positions: np.ndarray,
        style: MarkerStyle,
        colors: np.ndarray | None = None,
    ) -> Any | list[Any]:
        """Render ``positions`` once into ``view`` and return the GL items.

        Parameters
        ----------
        view:
            Target ``GLViewWidget``. Must equal the widget passed to
            ``__init__``.
        positions:
            ``(N, 3)`` ndarray of marker world positions.
        style:
            Visual specification for these markers.
        colors:
            Optional ``(N, 4)`` RGBA float array (each component in
            ``[0, 1]``). If ``None``, the style's ``fill_color`` is
            broadcast to all markers (only valid for
            :class:`StaticColor`).

        Returns
        -------
        item_or_items:
            The single ``GLScatterPlotItem`` for scatter-friendly
            shapes, or a list of ``GLMeshItem`` (one per marker) for
            mesh-based shapes.
        """
        if view is not self._widget:
            raise ValueError(
                "draw(view=...) must be the same GLViewWidget passed to __init__"
            )
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"draw() expects positions of shape (N, 3); got {positions.shape}"
            )

        n = int(positions.shape[0])
        col = _coerce_colors_for_n(colors, n, style)
        pos_f32 = np.ascontiguousarray(positions, dtype=np.float32)

        if style.shape in _SCATTER_SHAPES:
            item = self._build_scatter_item(pos_f32, style, col)
            view.addItem(item)
            return item
        items = self._build_mesh_items(pos_f32, style, col)
        for it in items:
            view.addItem(it)
        return items

    # ------------------------------------------------------------------
    # MarkerRenderer Protocol
    # ------------------------------------------------------------------

    def add_markers(
        self,
        positions: np.ndarray,
        style: MarkerStyle,
        label: str = "",
    ) -> str:
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        if not isinstance(label, str):
            raise TypeError(f"label must be str; got {type(label).__name__}")

        pos = _coerce_positions(positions)  # (T, M, 3) float32
        n_markers = int(pos.shape[1])

        # Reject palette / data-driven without explicit per-marker colors.
        if not isinstance(style.fill_color, StaticColor):
            raise NotImplementedError(
                "PyQtGLMarkerRenderer.add_markers requires a StaticColor "
                "fill_color; use draw(..., colors=...) for palette / "
                "data-driven scales."
            )
        colors = _coerce_colors_for_n(None, n_markers, style)

        is_scatter = style.shape in _SCATTER_SHAPES
        first_frame = pos[0]
        if is_scatter:
            item = self._build_scatter_item(first_frame, style, colors)
            self._widget.addItem(item)
            items = [item]
        else:
            items = self._build_mesh_items(first_frame, style, colors)
            for it in items:
                self._widget.addItem(it)

        prefix = label or style.shape.value
        handle = self._next_handle(prefix)
        self._entries[handle] = _MarkerEntry(style, pos, colors, items, is_scatter)
        return handle

    def update_frame(self, handle: str, frame_idx: int) -> None:
        entry = self._entries.get(handle)
        if entry is None:
            raise KeyError(f"Unknown handle: {handle!r}")
        if not isinstance(frame_idx, (int, np.integer)) or isinstance(frame_idx, bool):
            raise TypeError(f"frame_idx must be int; got {type(frame_idx).__name__}")
        n_frames = int(entry.positions.shape[0])
        if frame_idx < 0 or frame_idx >= n_frames:
            raise IndexError(
                f"frame_idx {frame_idx} out of range for {n_frames} frames"
            )

        positions_frame = entry.positions[frame_idx]
        if entry.is_scatter:
            entry.items[0].setData(
                pos=positions_frame,
                color=entry.colors,
                size=float(entry.style.size_px),
            )
        else:
            # Translate the unit mesh per marker.
            if entry.style.shape is MarkerShape.CUSTOM_MESH:
                assert entry.style.custom_mesh is not None
                unit_verts, faces = _custom_mesh_to_arrays(
                    entry.style.custom_mesh, entry.style.size_px
                )
            else:
                unit_verts, faces = _mesh_for_shape(entry.style)
            verts_f32 = np.ascontiguousarray(unit_verts, dtype=np.float32)
            faces_i32 = np.ascontiguousarray(faces, dtype=np.int32)
            for i, item in enumerate(entry.items):
                translated = verts_f32 + np.asarray(
                    positions_frame[i], dtype=np.float32
                )
                item.setMeshData(vertexes=translated, faces=faces_i32)

    def update_style(self, handle: str, style: MarkerStyle) -> None:
        entry = self._entries.get(handle)
        if entry is None:
            raise KeyError(f"Unknown handle: {handle!r}")
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        if style.shape is not entry.style.shape:
            raise ValueError(
                "update_style cannot change MarkerShape; remove the handle and re-add."
            )
        if not isinstance(style.fill_color, StaticColor):
            raise NotImplementedError(
                "PyQtGLMarkerRenderer.update_style requires a StaticColor fill_color."
            )
        n_markers = int(entry.positions.shape[1])
        new_colors = _coerce_colors_for_n(None, n_markers, style)

        # Update color/size in place, preserving current frame geometry.
        if entry.is_scatter:
            entry.items[0].setData(
                color=new_colors,
                size=float(style.size_px),
            )
        else:
            for i, item in enumerate(entry.items):
                item.setColor(tuple(float(c) for c in new_colors[i]))

        entry.style = style
        entry.colors = new_colors

    def set_visible(self, handle: str, visible: bool) -> None:
        entry = self._entries.get(handle)
        if entry is None:
            raise KeyError(f"Unknown handle: {handle!r}")
        if not isinstance(visible, bool):
            raise TypeError(f"visible must be bool; got {type(visible).__name__}")
        for item in entry.items:
            item.setVisible(visible)

    def remove(self, handle: str) -> None:
        entry = self._entries.pop(handle, None)
        if entry is None:
            raise KeyError(f"Unknown handle: {handle!r}")
        for item in entry.items:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._widget.removeItem(item)
