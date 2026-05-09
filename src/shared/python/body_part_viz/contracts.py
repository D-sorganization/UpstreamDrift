"""Runtime-checkable Protocols for the body_part_viz package.

These contracts define the abstract surface every shape, fitter, and
renderer implementation must satisfy. They are intentionally backend-
agnostic: nothing in this module imports matplotlib, pyqtgraph, or any
other rendering library.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ._types import FittedShape
from .bindings import MarkerBinding
from .theme import ShapeTheme

__all__ = ["BodyPartShape", "ShapeFitter", "ShapeRenderer"]


@runtime_checkable
class BodyPartShape(Protocol):
    """A geometric body-part visualisation.

    Implementations include line, cylinder, ellipsoid, capsule, mesh, and
    composite shapes. Each shape exposes its rest-pose vertices / faces
    and knows how to apply a :class:`FittedShape` transform.
    """

    shape_id: str
    rest_dimensions: tuple[float, ...]

    def vertices_at_rest(self) -> np.ndarray:
        """Return ``(V, 3)`` vertex array in the shape's local frame."""
        ...

    def faces(self) -> np.ndarray:
        """Return ``(F, 3)`` triangle indices, or empty array for line shapes."""
        ...

    def transform(self, fitted: FittedShape) -> np.ndarray:
        """Return ``(V, 3)`` vertices after applying the fitted transform."""
        ...


@runtime_checkable
class ShapeFitter(Protocol):
    """Compute a per-frame transform from markers to a fitted shape."""

    def fit(
        self,
        shape: BodyPartShape,
        binding: MarkerBinding,
        markers_xyz: dict[str, np.ndarray],
    ) -> FittedShape:
        """Fit ``shape`` to ``markers_xyz`` according to ``binding``.

        Parameters
        ----------
        shape:
            The geometric shape being fitted.
        binding:
            Marker binding describing how the shape attaches to markers.
        markers_xyz:
            Mapping ``marker_name -> (T, 3)`` world-frame positions.
        """
        ...


@runtime_checkable
class ShapeRenderer(Protocol):
    """Backend-specific renderer (matplotlib, pyqtgraph, ...).

    The contract intentionally exposes no Qt or matplotlib types so the
    same shape / fitter implementations can target any backend.
    """

    def add_shape(
        self,
        shape: BodyPartShape,
        fitted: FittedShape,
        theme: ShapeTheme,
    ) -> str:
        """Add ``shape`` to the scene and return a stable handle."""
        ...

    def update_frame(self, handle: str, frame_idx: int) -> None:
        """Update the geometry referred to by ``handle`` to ``frame_idx``."""
        ...

    def set_visible(self, handle: str, visible: bool) -> None:
        """Show or hide the geometry referred to by ``handle``."""
        ...

    def remove(self, handle: str) -> None:
        """Remove the geometry referred to by ``handle`` from the scene."""
        ...
