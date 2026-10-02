"""Explicit coordinate expansion preserves existing Hermite polynomial motion."""

from __future__ import annotations
from collections.abc import Sequence
from numpy.typing import NDArray
from dataclasses import dataclass
import re
import numpy as np
from src.shared.python.estimation import CubicHermiteSplineTrajectory
from .contracts import ImageSplineStart

REFERENCE_POSE_ATOL = 1e-10


def _reference_pose(
    values: Sequence[float] | NDArray[np.float64], size: int
) -> tuple[float, ...]:
    raw = np.asarray(values)
    if (
        raw.shape != (size,)
        or raw.dtype.kind not in "ifu"
        or not np.isfinite(raw).all()
        or any(isinstance(value, (bool, np.bool_)) for value in values)
    ):
        raise ValueError(
            "Reference pose must be a complete finite numeric native-order vector"
        )
    return tuple(float(value) for value in raw)


def _selection(start: ImageSplineStart, names: tuple[str, ...]) -> tuple[str, ...]:
    if (
        not isinstance(names, tuple)
        or not names
        or any(
            not isinstance(name, str) or name not in start.coordinate_order
            for name in names
        )
    ):
        raise ValueError(
            "Expanded free coordinates must be immutable known native names"
        )
    if len(set(names)) != len(names):
        raise ValueError("Expanded free coordinates must be unique")
    if (
        tuple(name for name in names if name in start.free_coordinates)
        != start.free_coordinates
    ):
        raise ValueError("Expansion cannot lose or reorder existing free coordinates")
    return tuple(name for name in names if name not in start.free_coordinates)


@dataclass(frozen=True)
class SplineCoordinateExpansion:
    """Copied expansion receipt; changed free order/hash is not exact parent restart."""

    original_coefficient_sha256: str
    expanded_start: ImageSplineStart
    added_coordinates: tuple[str, ...]
    reference_pose: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.original_coefficient_sha256, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.original_coefficient_sha256)
            is None
        ):
            raise ValueError(
                "Expansion requires original canonical coefficient identity"
            )
        if not isinstance(self.expanded_start, ImageSplineStart):
            raise ValueError("Expansion requires a typed immutable spline start")
        if (
            not isinstance(self.added_coordinates, tuple)
            or any(
                name not in self.expanded_start.free_coordinates
                for name in self.added_coordinates
            )
            or len(set(self.added_coordinates)) != len(self.added_coordinates)
        ):
            raise ValueError("Added coordinates must be unique expanded free names")
        if not isinstance(self.reference_pose, tuple):
            raise ValueError("Expansion reference pose must be immutable")
        object.__setattr__(
            self,
            "reference_pose",
            _reference_pose(
                self.reference_pose, len(self.expanded_start.coordinate_order)
            ),
        )


def expand_image_spline_coordinates(
    start: ImageSplineStart,
    free_coordinates: tuple[str, ...],
    reference_pose: Sequence[float] | NDArray[np.float64],
) -> SplineCoordinateExpansion:
    """Preserve old q/v coefficients, adding constant positions and zero velocities.

    The complete reference pose uses native coordinate order; old free positions
    must match the first knot within REFERENCE_POSE_ATOL (absolute, zero relative
    tolerance). They are validated, never overwritten. The caller must separately
    establish that formerly locked source coordinates equal the reference pose.
    Model identity, knot clock and native order remain unchanged. Added polynomial
    coordinates are constant; canonical evaluation can differ by floating-point
    roundoff after a decision-size change, so this is not a bitwise exact restart.
    Bounds are checked separately through the canonical strict native initializer.
    """
    if not isinstance(start, ImageSplineStart):
        raise ValueError("Coordinate expansion requires a typed spline start")
    added = _selection(start, free_coordinates)
    reference = _reference_pose(reference_pose, len(start.coordinate_order))
    old = CubicHermiteSplineTrajectory(
        np.asarray(start.knot_times), len(start.free_coordinates)
    )
    old_q, old_v = old.unpack(np.asarray(start.spline_coefficients))
    existing = np.array(
        [
            reference[start.coordinate_order.index(name)]
            for name in start.free_coordinates
        ]
    )
    if not np.allclose(existing, old_q[0], rtol=0.0, atol=REFERENCE_POSE_ATOL):
        raise ValueError(
            "Reference pose conflicts with existing initial free positions"
        )
    new = CubicHermiteSplineTrajectory(
        np.asarray(start.knot_times), len(free_coordinates)
    )
    q = np.tile(
        [reference[start.coordinate_order.index(name)] for name in free_coordinates],
        (len(start.knot_times), 1),
    )
    v = np.zeros_like(q)
    indices = [free_coordinates.index(name) for name in start.free_coordinates]
    q[:, indices], v[:, indices] = old_q, old_v
    expanded = ImageSplineStart.from_coefficients(
        start.knot_times,
        new.pack(q, v),
        start.coordinate_order,
        free_coordinates,
        start.model_sha,
    )
    return SplineCoordinateExpansion(
        start.coefficient_sha256, expanded, added, reference
    )
