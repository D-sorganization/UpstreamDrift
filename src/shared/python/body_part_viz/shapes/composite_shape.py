"""Composite shape: a tree of child shapes with per-child local transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._types import FittedShape
from ..contracts import BodyPartShape
from ._transform import apply_fitted_to_rest_vertices

__all__ = ["CompositeShape"]


def _validate_local_transform(transform: np.ndarray, idx: int) -> np.ndarray:
    if not isinstance(transform, np.ndarray):
        raise TypeError(
            f"child[{idx}] local_transform must be ndarray; "
            f"got {type(transform).__name__}"
        )
    if transform.shape != (4, 4):
        raise ValueError(
            f"child[{idx}] local_transform must be (4, 4); got {transform.shape}"
        )
    if not bool(np.all(np.isfinite(transform))):
        raise ValueError(f"child[{idx}] local_transform must be finite")
    return transform.astype(np.float64, copy=False)


def _apply_local(transform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return vertices @ rot.T + trans


class CompositeShape:
    """Composite of child :class:`BodyPartShape` instances.

    Each child is paired with a 4x4 ``local_transform`` matrix that
    positions the child within the composite's local frame.
    ``rest_dimensions`` is the concatenation of every child's
    ``rest_dimensions``; ``faces()`` re-indexes child faces into a single
    flat vertex array.
    """

    shape_id: str
    rest_dimensions: tuple[float, ...]

    def __init__(
        self,
        children: Sequence[tuple[BodyPartShape, np.ndarray]],
        *,
        shape_id: str = "composite",
    ) -> None:
        if not isinstance(children, (list, tuple)):
            raise TypeError(
                f"children must be a list/tuple; got {type(children).__name__}"
            )
        if len(children) == 0:
            raise ValueError("children must be non-empty")
        if not isinstance(shape_id, str) or not shape_id:
            raise ValueError(f"shape_id must be non-empty str; got {shape_id!r}")

        validated: list[tuple[BodyPartShape, np.ndarray]] = []
        rest_dims: list[float] = []
        for idx, entry in enumerate(children):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    f"children[{idx}] must be a (shape, local_transform) tuple"
                )
            child_shape, local_transform = entry
            if not isinstance(child_shape, BodyPartShape):
                raise TypeError(
                    f"children[{idx}][0] must satisfy BodyPartShape; "
                    f"got {type(child_shape).__name__}"
                )
            transform = _validate_local_transform(local_transform, idx)
            validated.append((child_shape, transform))
            rest_dims.extend(child_shape.rest_dimensions)

        self.shape_id = shape_id
        self.rest_dimensions = tuple(rest_dims)
        self._children = tuple(validated)
        self._build_combined()

    def _build_combined(self) -> None:
        verts_blocks: list[np.ndarray] = []
        faces_blocks: list[np.ndarray] = []
        offset = 0
        for child_shape, transform in self._children:
            child_verts = child_shape.vertices_at_rest()
            child_faces = child_shape.faces()
            placed = _apply_local(transform, child_verts)
            verts_blocks.append(placed)
            if child_faces.size > 0:
                faces_blocks.append(child_faces.astype(np.int64) + offset)
            offset += placed.shape[0]
        self._vertices = (
            np.concatenate(verts_blocks, axis=0).astype(np.float64)
            if verts_blocks
            else np.zeros((0, 3), dtype=np.float64)
        )
        self._faces = (
            np.concatenate(faces_blocks, axis=0).astype(np.int64)
            if faces_blocks
            else np.zeros((0, 3), dtype=np.int64)
        )

    @property
    def children(self) -> tuple[tuple[BodyPartShape, np.ndarray], ...]:
        return self._children

    def vertices_at_rest(self) -> np.ndarray:
        return self._vertices.copy()

    def faces(self) -> np.ndarray:
        return self._faces.copy()

    def transform(self, fitted: FittedShape) -> np.ndarray:
        return apply_fitted_to_rest_vertices(self._vertices, fitted)
