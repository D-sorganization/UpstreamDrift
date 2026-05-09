"""Runtime-checkable Protocols for the plot_style package.

These contracts define the abstract surface that every backend
(matplotlib, pyqtgraph, ...) and color-resolver implementation must
satisfy. Nothing in this module imports a rendering library — the
contracts are intentionally backend-agnostic.

* :class:`MarkerRenderer` — adds, updates, and removes markers in a
  scene. Implementations live in
  :mod:`src.shared.python.plot_style.renderers`.
* :class:`ColorResolver` — resolves a :data:`ColorScale` to numeric
  RGBA tuples. Implementations live in
  :mod:`src.shared.python.plot_style.resolvers`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .colors import ColorScale
from .markers import MarkerStyle

__all__ = ["ColorResolver", "MarkerRenderer", "MarkerShapeRenderer"]


@runtime_checkable
class MarkerRenderer(Protocol):
    """Backend-specific marker renderer.

    A renderer maintains a collection of marker primitives identified
    by stable string handles. Concrete subclasses target matplotlib,
    pyqtgraph, plotly, etc.
    """

    def add_markers(
        self,
        positions: np.ndarray,
        style: MarkerStyle,
        label: str = "",
    ) -> str:
        """Add a marker primitive and return a stable handle.

        Parameters
        ----------
        positions:
            ``(T, M, 3)`` or ``(T, 3)`` ndarray of marker world
            positions per frame.
        style:
            Visual specification for these markers.
        label:
            Optional human-readable label.

        Returns
        -------
        handle:
            Non-empty string handle suitable for later
            :meth:`update_frame`, :meth:`update_style`,
            :meth:`set_visible`, and :meth:`remove` calls.
        """
        ...

    def update_frame(self, handle: str, frame_idx: int) -> None:
        """Move the markers identified by ``handle`` to ``frame_idx``."""
        ...

    def update_style(self, handle: str, style: MarkerStyle) -> None:
        """Replace the style applied to ``handle``."""
        ...

    def set_visible(self, handle: str, visible: bool) -> None:
        """Show or hide the markers identified by ``handle``."""
        ...

    def remove(self, handle: str) -> None:
        """Remove the markers identified by ``handle`` from the scene."""
        ...


@runtime_checkable
class MarkerShapeRenderer(Protocol):
    """Backend-agnostic marker-shape primitive.

    A shape renderer turns a :class:`MarkerStyle` into a triangle mesh
    expressed in the marker's local frame. The mesh is centred on the
    origin and scaled so its bounding sphere has radius
    ``style.size_px / 2`` (i.e. the diameter equals ``size_px``). Higher
    layers translate the mesh to per-marker world positions and apply
    per-pixel sizing if the backend works in screen coordinates.
    """

    shape_id: str

    def mesh(self, style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(vertices, faces)`` for ``style``.

        Parameters
        ----------
        style:
            Marker style. ``style.size_px`` controls the linear scale.

        Returns
        -------
        vertices:
            ``(V, 3)`` ``float64`` ndarray of vertex positions in the
            marker-local frame, scaled by ``size_px / 2``.
        faces:
            ``(F, 3)`` ``int64`` ndarray of triangle indices.
        """
        ...


@runtime_checkable
class ColorResolver(Protocol):
    """Resolves a :data:`ColorScale` to numeric RGBA values."""

    def resolve_one(
        self,
        scale: ColorScale,
        frame_idx: int,
        marker_idx: int | None = None,
    ) -> tuple[float, float, float, float]:
        """Resolve a single ``(frame_idx, marker_idx)`` pair.

        Returns
        -------
        rgba:
            ``(r, g, b, a)`` tuple with each component in ``[0, 1]``.
        """
        ...

    def resolve_array(
        self,
        scale: ColorScale,
        n_frames: int,
        n_markers: int | None = None,
    ) -> np.ndarray:
        """Bulk-resolve into an ``(n_frames, [n_markers,] 4)`` ndarray.

        Useful for pre-computing per-frame look-up tables to avoid
        per-frame Python overhead during animation.
        """
        ...
