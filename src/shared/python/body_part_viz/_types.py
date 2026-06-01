"""Shared dataclasses for body_part_viz: per-frame fitted-shape state.

``FittedShape`` is the output of any :class:`ShapeFitter`. It is consumed by
:class:`BodyPartShape.transform` and :class:`ShapeRenderer.add_shape`.

All fields are validated for shape and dtype consistency in
``__post_init__`` (Design-by-Contract).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bindings import MarkerBinding

__all__ = ["FittedShape"]


@dataclass(frozen=True)
class FittedShape:
    """Per-frame placement of a fitted shape.

    Attributes
    ----------
    shape_id:
        Stable, non-empty identifier of the shape this fit refers to.
    binding:
        The marker binding used to compute this fit.
    centroid:
        ``(T, 3)`` world-frame centroid per frame.
    rotation_matrix:
        ``(T, 3, 3)`` rotation matrix per frame.
    scale:
        ``(T, 3)`` anisotropic scale per frame; entries on valid frames
        must be strictly positive.
    valid_mask:
        ``(T,)`` boolean mask. A frame is valid iff all source markers
        contributing to it are finite.
    """

    shape_id: str
    binding: MarkerBinding
    centroid: np.ndarray
    rotation_matrix: np.ndarray
    scale: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.shape_id, str) or not self.shape_id:
            raise ValueError(
                f"shape_id must be a non-empty string; got {self.shape_id!r}"
            )
        if not isinstance(self.binding, MarkerBinding):
            raise TypeError(
                f"binding must be a MarkerBinding; got {type(self.binding).__name__}"
            )

        for name in ("centroid", "rotation_matrix", "scale", "valid_mask"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy.ndarray; got {type(arr).__name__}"
                )

        # Floating-dtype enforcement for numeric fields. The contract and type
        # hints require floating arrays; integer/object dtypes silently
        # propagate into geometry math and cause precision/backend bugs.
        # See issue #4776.
        for name in ("centroid", "rotation_matrix", "scale"):
            arr = getattr(self, name)
            if arr.dtype.kind != "f":
                raise TypeError(f"{name} must have a floating dtype; got {arr.dtype}")

        if self.centroid.ndim != 2 or self.centroid.shape[1] != 3:
            raise ValueError(
                f"centroid must have shape (T, 3); got {self.centroid.shape}"
            )
        n_frames = self.centroid.shape[0]

        expected_rot = (n_frames, 3, 3)
        if self.rotation_matrix.shape != expected_rot:
            raise ValueError(
                f"rotation_matrix must have shape {expected_rot}; "
                f"got {self.rotation_matrix.shape}"
            )

        expected_scale = (n_frames, 3)
        if self.scale.shape != expected_scale:
            raise ValueError(
                f"scale must have shape {expected_scale}; got {self.scale.shape}"
            )

        expected_mask = (n_frames,)
        if self.valid_mask.shape != expected_mask:
            raise ValueError(
                f"valid_mask must have shape {expected_mask}; "
                f"got {self.valid_mask.shape}"
            )

        if self.valid_mask.dtype != np.bool_:
            raise TypeError(
                f"valid_mask must have dtype=bool; got {self.valid_mask.dtype}"
            )

        if n_frames > 0 and bool(self.valid_mask.any()):
            valid_scale = self.scale[self.valid_mask]
            if not bool(np.all(np.isfinite(valid_scale))):
                raise ValueError("scale entries on valid frames must be finite")
            if not bool(np.all(valid_scale > 0.0)):
                raise ValueError(
                    "scale entries on valid frames must be strictly positive"
                )

    @property
    def n_frames(self) -> int:
        """Return the number of frames (T) in the trajectory."""
        return self.centroid.shape[0]
