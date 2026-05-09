"""GPU-accelerated 3D renderer using ``pyqtgraph.opengl``.

This is the high-performance counterpart to the matplotlib renderer.
It is intended for cases where the matplotlib backend's per-frame
update budget is exceeded (typically: > 30 segments × > 1000 vertices
each).

Optional dependency policy
--------------------------
``pyqtgraph`` and its ``opengl`` submodule are imported lazily inside
this module so that ``import body_part_viz`` never pulls them in. The
package's top-level ``__init__`` does not import this module — callers
must explicitly do ``from .renderers.pyqtgl_renderer import PyQtGLRenderer``.

Install the optional extra to enable this backend::

    pip install upstream-drift[body-part-viz-gl]
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import FittedShape
from ..contracts import BodyPartShape
from ..theme import ShapeTheme

if TYPE_CHECKING:  # pragma: no cover - typing only
    pyqtgraph_opengl_module = Any  # placeholder for type checkers

__all__ = ["PyQtGLRenderer"]


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
            "pyqtgraph.opengl is required for PyQtGLRenderer. "
            "Install the optional extra: pip install upstream-drift[body-part-viz-gl]"
        ) from exc
    return gl


def _hex_or_name_to_rgba(
    color: str, opacity: float
) -> tuple[float, float, float, float]:
    """Convert a matplotlib-recognised colour string to an RGBA tuple."""
    from matplotlib.colors import to_rgba  # noqa: PLC0415 - lightweight, no extra dep

    r, g, b, a = to_rgba(color)
    return (float(r), float(g), float(b), float(a) * float(opacity))


class _ShapeEntry:
    """Internal bookkeeping for one shape registered with the renderer."""

    __slots__ = ("shape", "fitted", "theme", "item", "is_line", "user_visible")

    def __init__(
        self,
        shape: BodyPartShape,
        fitted: FittedShape,
        theme: ShapeTheme,
        item: Any,
        is_line: bool,
    ) -> None:
        self.shape = shape
        self.fitted = fitted
        self.theme = theme
        self.item = item
        self.is_line = is_line
        # User-controlled visibility — preserved across update_frame() calls so
        # an explicit set_visible(handle, False) is not overridden by the next
        # valid frame. See issue #4784.
        self.user_visible = True


class PyQtGLRenderer:
    """GPU-accelerated 3D renderer implementing :class:`ShapeRenderer`.

    One ``GLMeshItem`` (or ``GLLinePlotItem`` for line shapes) is
    created per registered shape; ``update_frame`` mutates that item
    in-place via ``setMeshData`` / ``setData`` rather than allocating
    new GL items each frame.

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
        self._entries: dict[str, _ShapeEntry] = {}
        self._handle_counter = 0

    def _next_handle(self, shape_id: str) -> str:
        self._handle_counter += 1
        return f"{shape_id}#{self._handle_counter}"

    def _vertices_for_frame(self, entry: _ShapeEntry, frame_idx: int) -> np.ndarray:
        """Return ``(V, 3)`` vertices at ``frame_idx`` (single-frame slice)."""
        n_frames = entry.fitted.centroid.shape[0]
        if not isinstance(frame_idx, (int, np.integer)) or isinstance(frame_idx, bool):
            raise TypeError(f"frame_idx must be int; got {type(frame_idx).__name__}")
        if frame_idx < 0 or frame_idx >= n_frames:
            raise IndexError(
                f"frame_idx {frame_idx} out of range for {n_frames} frames"
            )

        # Build a single-frame FittedShape view to reuse shape.transform()
        single = FittedShape(
            shape_id=entry.fitted.shape_id,
            binding=entry.fitted.binding,
            centroid=entry.fitted.centroid[frame_idx : frame_idx + 1],
            rotation_matrix=entry.fitted.rotation_matrix[frame_idx : frame_idx + 1],
            scale=entry.fitted.scale[frame_idx : frame_idx + 1],
            valid_mask=entry.fitted.valid_mask[frame_idx : frame_idx + 1],
        )
        verts = entry.shape.transform(single)
        # transform() returns (T, V, 3); collapse to (V, 3)
        return np.asarray(verts[0], dtype=np.float32)

    # -- ShapeRenderer protocol --------------------------------------

    def add_shape(
        self,
        shape: BodyPartShape,
        fitted: FittedShape,
        theme: ShapeTheme,
    ) -> str:
        if not isinstance(theme, ShapeTheme):
            raise TypeError(f"theme must be a ShapeTheme; got {type(theme).__name__}")
        if shape.shape_id != fitted.shape_id:
            raise ValueError(
                f"shape.shape_id {shape.shape_id!r} does not match "
                f"fitted.shape_id {fitted.shape_id!r}"
            )

        faces = np.asarray(shape.faces())
        is_line = faces.shape[0] == 0
        rgba = _hex_or_name_to_rgba(theme.color, theme.opacity)

        if is_line:
            verts = np.asarray(shape.vertices_at_rest(), dtype=np.float32)
            item = self._gl.GLLinePlotItem(
                pos=verts,
                color=rgba,
                width=max(theme.edge_width, 1.0),
                antialias=True,
                mode="line_strip",
            )
        else:
            verts = np.asarray(shape.vertices_at_rest(), dtype=np.float32)
            faces_i = np.asarray(faces, dtype=np.int32)
            mesh_data = self._gl.MeshData(vertexes=verts, faces=faces_i)
            item = self._gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=not theme.flat_shaded,
                color=rgba,
                shader="shaded",
                drawEdges=theme.edge_width > 0.0,
                edgeColor=_hex_or_name_to_rgba(theme.edge_color, 1.0),
            )

        self._widget.addItem(item)
        handle = self._next_handle(shape.shape_id)
        self._entries[handle] = _ShapeEntry(shape, fitted, theme, item, is_line)
        return handle

    def update_frame(self, handle: str, frame_idx: int) -> None:
        entry = self._entries.get(handle)
        if entry is None:
            raise KeyError(f"Unknown handle: {handle!r}")
        verts = self._vertices_for_frame(entry, frame_idx)

        # NaN frames mean "invalid" — hide the item rather than push NaNs to GL.
        if not np.all(np.isfinite(verts)):
            entry.item.setVisible(False)
            return
        # Respect user-controlled visibility: if set_visible(handle, False)
        # was called, do not silently re-show on the next valid frame.
        entry.item.setVisible(entry.user_visible)
        if not entry.user_visible:
            return

        if entry.is_line:
            entry.item.setData(pos=verts)
        else:
            faces_i = np.asarray(entry.shape.faces(), dtype=np.int32)
            entry.item.setMeshData(vertexes=verts, faces=faces_i)

    def set_visible(self, handle: str, visible: bool) -> None:
        entry = self._entries.get(handle)
        if entry is None:
            raise KeyError(f"Unknown handle: {handle!r}")
        if not isinstance(visible, bool):
            raise TypeError(f"visible must be bool; got {type(visible).__name__}")
        entry.user_visible = visible
        entry.item.setVisible(visible)

    def remove(self, handle: str) -> None:
        entry = self._entries.pop(handle, None)
        if entry is None:
            raise KeyError(f"Unknown handle: {handle!r}")
        with contextlib.suppress(Exception):  # pragma: no cover - defensive teardown
            self._widget.removeItem(entry.item)
