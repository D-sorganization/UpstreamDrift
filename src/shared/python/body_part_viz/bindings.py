"""Marker-binding dataclass and binding-kind enum for body-part visualisations.

Defines how a geometric shape attaches to mocap markers:

* ``BETWEEN_TWO`` — length / orientation derived from two markers.
* ``CLUSTER`` — rigid Kabsch fit on three or more markers.
* ``ON_MARKER`` — anchored at a single marker (e.g. head sphere).

The dataclass is frozen and validates all invariants in ``__post_init__``
(Design-by-Contract).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["BindingKind", "MarkerBinding"]


class BindingKind(str, Enum):
    """How a shape attaches to mocap markers."""

    BETWEEN_TWO = "between_two"
    CLUSTER = "cluster"
    ON_MARKER = "on_marker"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class MarkerBinding:
    """Rest-pose binding of a shape to one or more mocap markers.

    Attributes
    ----------
    kind:
        Which binding strategy to use.
    marker_names:
        Tuple of marker label strings. Length depends on ``kind``.
    rest_dimensions:
        Optional rest-pose dimensions (lengths / radii). All entries must
        be strictly positive.
    rest_orientation_quat:
        Rest-pose orientation as a unit quaternion ``(w, x, y, z)``.
    """

    kind: BindingKind
    marker_names: tuple[str, ...]
    rest_dimensions: tuple[float, ...] = ()
    rest_orientation_quat: tuple[float, float, float, float] = field(
        default=(1.0, 0.0, 0.0, 0.0)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BindingKind):
            raise TypeError(f"kind must be BindingKind, got {type(self.kind).__name__}")

        # Reject plain strings explicitly: a bare string is iterable and would
        # otherwise be silently treated as a sequence of single-character
        # marker names (e.g. "ab" -> ("a", "b")). See issue #4775.
        if isinstance(self.marker_names, str):
            raise TypeError("marker_names must be a tuple of strings, not a single str")
        # Normalize sequence-like inputs (e.g. list) to an immutable tuple so
        # callers cannot mutate the original after construction.
        if not isinstance(self.marker_names, tuple):
            try:
                normalized = tuple(self.marker_names)
            except TypeError as exc:
                raise TypeError(
                    "marker_names must be a tuple of strings, "
                    f"got {type(self.marker_names).__name__}"
                ) from exc
            object.__setattr__(self, "marker_names", normalized)

        for name in self.marker_names:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"marker_names entries must be non-empty strings; got {name!r}"
                )

        count = len(self.marker_names)
        if self.kind is BindingKind.BETWEEN_TWO and count != 2:
            raise ValueError(
                f"BETWEEN_TWO binding requires exactly 2 markers, got {count}"
            )
        if self.kind is BindingKind.CLUSTER and count < 3:
            raise ValueError(
                f"CLUSTER binding requires at least 3 markers, got {count}"
            )
        if self.kind is BindingKind.ON_MARKER and count != 1:
            raise ValueError(
                f"ON_MARKER binding requires exactly 1 marker, got {count}"
            )

        if not isinstance(self.rest_dimensions, tuple):
            raise TypeError(
                "rest_dimensions must be a tuple of floats, "
                f"got {type(self.rest_dimensions).__name__}"
            )
        for dim in self.rest_dimensions:
            if not isinstance(dim, (int, float)) or isinstance(dim, bool):
                raise TypeError(f"rest_dimensions entries must be numeric, got {dim!r}")
            if not math.isfinite(float(dim)) or float(dim) <= 0.0:
                raise ValueError(
                    f"rest_dimensions entries must be finite and positive; got {dim!r}"
                )

        quat = self.rest_orientation_quat
        if not isinstance(quat, tuple) or len(quat) != 4:
            raise ValueError(
                f"rest_orientation_quat must be a 4-tuple (w, x, y, z); got {quat!r}"
            )
        for component in quat:
            if not isinstance(component, (int, float)) or isinstance(component, bool):
                raise TypeError(
                    f"rest_orientation_quat entries must be numeric; got {component!r}"
                )
            if not math.isfinite(float(component)):
                raise ValueError(
                    f"rest_orientation_quat entries must be finite; got {component!r}"
                )
        norm = math.sqrt(sum(float(c) * float(c) for c in quat))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(
                "rest_orientation_quat must be unit-norm "
                f"(within 1e-6); got norm={norm:.9f}"
            )
