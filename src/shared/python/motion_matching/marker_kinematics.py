"""Optional immutable native marker derivatives without engine state exposure."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be an ordered sequence")
    names = tuple(values)
    if not names or any(
        not isinstance(item, str) or not item.strip() for item in names
    ):
        raise ValueError(f"{name} must contain nonempty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must retain unique identities")
    return names


@dataclass(frozen=True)
class MarkerLinearization:
    """World metre points and derivatives in exact marker/native scalar order."""

    positions: np.ndarray
    jacobian: np.ndarray
    marker_labels: tuple[str, ...]
    coordinate_order: tuple[str, ...]

    def __post_init__(self) -> None:
        labels = _names(self.marker_labels, "Marker labels")
        order = _names(self.coordinate_order, "Coordinate order")
        for name, shape in (
            ("positions", (len(labels), 3)),
            ("jacobian", (len(labels), 3, len(order))),
        ):
            value = np.asarray(getattr(self, name))
            if (
                value.dtype.kind not in "iuf"
                or value.shape != shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"Marker linearization {name} requires finite numeric {shape}"
                )
            copied = np.array(value, dtype=float, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "marker_labels", labels)
        object.__setattr__(self, "coordinate_order", order)


@runtime_checkable
class MarkerLinearizer(Protocol):
    """A fit-owned resource exposing point derivatives through a public method."""

    def marker_linearization(self, q: np.ndarray) -> MarkerLinearization:
        """Linearize world marker points, or explicitly reject unsupported SDKs."""
        ...


@runtime_checkable
class MarkerLinearizationPlant(Protocol):
    """Optional factory; absence preserves existing finite-difference providers."""

    def create_marker_linearizer(
        self, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> MarkerLinearizer:
        """Create one derivative resource for the fit's fixed ordered attachments."""
        ...
