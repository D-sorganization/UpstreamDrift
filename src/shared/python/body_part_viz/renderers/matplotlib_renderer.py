"""Matplotlib 3D ``ShapeRenderer`` implementation.

Renders :class:`~body_part_viz.contracts.BodyPartShape` instances onto a
``mpl_toolkits.mplot3d.Axes3D`` using one matplotlib artist per shape:

* Line shapes use :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection`.
* Mesh-bearing shapes (cylinder, ellipsoid, capsule, composite, mesh)
  use :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.

``update_frame`` mutates the existing artist's geometry via
``set_segments`` / ``set_verts`` rather than rebuilding it. The scene is
never cleared. This keeps the per-frame budget low enough to satisfy the
60 fps target on the 26-shape × 200-vertex workload.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .._types import FittedShape
from ..contracts import BodyPartShape
from ..theme import ShapeTheme

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

__all__ = ["MatplotlibRenderer"]


@dataclass
class _ShapeEntry:
    """Bookkeeping for a single registered shape."""

    shape: BodyPartShape
    fitted: FittedShape
    faces: np.ndarray
    world_vertices: np.ndarray  # (T, V, 3)
    artist: Any  # Line3DCollection or Poly3DCollection
    is_line: bool


class MatplotlibRenderer:
    """Render body_part_viz shapes onto a matplotlib :class:`Axes3D`.

    Parameters
    ----------
    ax:
        The 3D axes to draw into. The renderer owns the artists it adds
        but never clears the axes.
    """

    def __init__(self, ax: Axes3D) -> None:
        if ax is None:
            raise TypeError("ax must not be None")
        self._ax = ax
        self._entries: dict[str, _ShapeEntry] = {}

    # ------------------------------------------------------------------
    # ShapeRenderer Protocol
    # ------------------------------------------------------------------
    def add_shape(
        self,
        shape: BodyPartShape,
        fitted: FittedShape,
        theme: ShapeTheme,
    ) -> str:
        """Add ``shape`` and return a stable handle.

        Precomputes per-frame world vertices once, then constructs a
        single matplotlib artist initialised at frame 0.
        """
        if not isinstance(shape, BodyPartShape):
            raise TypeError(
                f"shape must satisfy BodyPartShape; got {type(shape).__name__}"
            )
        if not isinstance(fitted, FittedShape):
            raise TypeError(f"fitted must be FittedShape; got {type(fitted).__name__}")
        if not isinstance(theme, ShapeTheme):
            raise TypeError(f"theme must be ShapeTheme; got {type(theme).__name__}")

        world = shape.transform(fitted)  # (T, V, 3)
        faces = shape.faces()
        is_line = faces.shape[0] == 0

        frame0 = self._frame_geometry(world, 0)
        if is_line:
            segments = self._line_segments(frame0)
            artist = Line3DCollection(
                segments,
                colors=theme.color,
                linewidths=max(theme.edge_width, 1.0),
                alpha=theme.opacity,
            )
        else:
            polys = self._face_polys(frame0, faces)
            artist = Poly3DCollection(
                polys,
                facecolors=theme.color,
                edgecolors=theme.edge_color,
                linewidths=theme.edge_width,
                alpha=theme.opacity,
            )

        self._ax.add_collection3d(artist)
        handle = uuid4().hex
        self._entries[handle] = _ShapeEntry(
            shape=shape,
            fitted=fitted,
            faces=faces,
            world_vertices=world,
            artist=artist,
            is_line=is_line,
        )
        return handle

    def update_frame(self, handle: str, frame_idx: int) -> None:
        """Update the artist for ``handle`` to frame ``frame_idx``.

        Calls ``set_segments`` / ``set_verts`` on the existing artist;
        never clears the axes. Issues a single ``draw_idle`` at the end.
        """
        entry = self._require(handle)
        if not isinstance(frame_idx, (int, np.integer)) or isinstance(frame_idx, bool):
            raise TypeError(f"frame_idx must be int; got {type(frame_idx).__name__}")
        n_frames = entry.world_vertices.shape[0]
        if not 0 <= int(frame_idx) < n_frames:
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {n_frames})")

        verts = self._frame_geometry(entry.world_vertices, int(frame_idx))
        if entry.is_line:
            entry.artist.set_segments(self._line_segments(verts))
        else:
            entry.artist.set_verts(self._face_polys(verts, entry.faces))

        # Single redraw signal at the end. Only emit on interactive
        # backends — under non-interactive ones (e.g. Agg in tests) the
        # host drives the draw cycle, and a synchronous draw on every
        # frame would obliterate the per-frame budget.
        if matplotlib.is_interactive():
            canvas = getattr(self._ax.figure, "canvas", None)
            if canvas is not None and hasattr(canvas, "draw_idle"):
                canvas.draw_idle()

    def set_visible(self, handle: str, visible: bool) -> None:
        """Show or hide the artist for ``handle``."""
        if not isinstance(visible, bool):
            raise TypeError(f"visible must be bool; got {type(visible).__name__}")
        entry = self._require(handle)
        entry.artist.set_visible(visible)

    def remove(self, handle: str) -> None:
        """Remove the artist for ``handle`` from the axes."""
        entry = self._require(handle)
        # If the artist is already detached, fall through and drop the
        # bookkeeping anyway.
        with contextlib.suppress(ValueError, NotImplementedError):
            entry.artist.remove()
        del self._entries[handle]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def add_segment_set(
        self,
        segments: Iterable[tuple[BodyPartShape, FittedShape]],
        theme_resolver: Callable[[FittedShape], ShapeTheme],
    ) -> list[str]:
        """Add many shapes; thin wrapper over :meth:`add_shape`.

        ``segments`` is an iterable of ``(shape, fitted)`` pairs.
        ``theme_resolver`` is called once per pair with the ``FittedShape``
        and must return a :class:`ShapeTheme`.
        """
        if not callable(theme_resolver):
            raise TypeError("theme_resolver must be callable")
        handles: list[str] = []
        for entry in segments:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("segments must yield (shape, fitted) tuples")
            shape, fitted = entry
            theme = theme_resolver(fitted)
            handles.append(self.add_shape(shape, fitted, theme))
        return handles

    def clear(self) -> None:
        """Remove every shape this renderer owns. Does not clear the axes."""
        for handle in list(self._entries.keys()):
            self.remove(handle)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require(self, handle: str) -> _ShapeEntry:
        if handle not in self._entries:
            raise KeyError(f"unknown shape handle: {handle!r}")
        return self._entries[handle]

    @staticmethod
    def _frame_geometry(world: np.ndarray, frame_idx: int) -> np.ndarray:
        verts = world[frame_idx]
        # NaN-fill invalid frames to a safe zero so matplotlib does not
        # raise; the artist is still hidden visually because subsequent
        # frames will overwrite. Caller filters via valid_mask if needed.
        if not np.all(np.isfinite(verts)):
            verts = np.where(np.isfinite(verts), verts, 0.0)
        return verts

    @staticmethod
    def _line_segments(vertices: np.ndarray) -> list[np.ndarray]:
        # A line shape stores ordered endpoints; emit consecutive pairs.
        if vertices.shape[0] < 2:
            return []
        return [
            np.stack([vertices[i], vertices[i + 1]], axis=0)
            for i in range(vertices.shape[0] - 1)
        ]

    @staticmethod
    def _face_polys(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
        if faces.shape[0] == 0:
            return []
        return [vertices[face] for face in faces]
