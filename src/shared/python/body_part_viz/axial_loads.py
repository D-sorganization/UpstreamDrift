"""Model-independent proximal-section axial reactions and optional provider seam."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from .force_display import SegmentLoadSeries


def axial_force_from_proximal_reaction(
    force: ArrayLike, proximal: ArrayLike, distal: ArrayLike
) -> float:
    """Return N = -F_parent_on_segment dot unit(proximal→distal), tension positive.

    All inputs must be finite 2D or 3D vectors in the same Cartesian frame and SI
    units. The reaction must act on the segment at its proximal section; a net
    force on a point mass or an unsigned force magnitude is not interchangeable.
    Inputs remain unchanged. Degenerate axes fail explicitly.
    """
    vectors = [np.asarray(value, dtype=float) for value in (force, proximal, distal)]
    if any(v.shape != vectors[0].shape for v in vectors) or vectors[0].shape not in (
        (2,),
        (3,),
    ):
        raise ValueError("force and endpoints must be matching 2D or 3D vectors")
    if not all(np.isfinite(v).all() for v in vectors):
        raise ValueError("force and endpoints must be finite")
    axis = vectors[2] - vectors[1]
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length <= 0:
        raise ValueError("segment axis must have finite positive length")
    result = -float(np.dot(vectors[0], axis / length))
    if not math.isfinite(result):
        raise ValueError("projected axial force must be finite")
    return result


@dataclass(frozen=True)
class AxialLoadFrame:
    """A single immutable, JSON-safe sample using the shared series contract."""

    time_s: float
    values_n: Mapping[str, float | None]
    source: str

    def __post_init__(self) -> None:
        from types import MappingProxyType

        if not isinstance(self.values_n, Mapping):
            raise TypeError("values_n must be a mapping")
        series = SegmentLoadSeries(
            (self.time_s,),
            {name: (value,) for name, value in self.values_n.items()},
            self.source,
        )
        object.__setattr__(self, "time_s", series.time_s[0])
        object.__setattr__(
            self,
            "values_n",
            MappingProxyType(
                {name: values[0] for name, values in series.values_n.items()}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the public wire schema; unavailable values encode as null."""
        return {
            "time_s": self.time_s,
            "values_n": dict(self.values_n),
            "source": self.source,
            "units": "N",
            "sign_convention": "tension-positive",
        }


@runtime_checkable
class AxialLoadProvider(Protocol):
    """Optional engine capability, independent of renderer and engine internals."""

    def get_segment_axial_loads(self) -> AxialLoadFrame | None:
        """Return current proximal-section loads, or None when unqualified."""
        ...


def read_axial_load_frame(provider: object, time_s: float) -> dict[str, Any] | None:
    """Read only declared provider capability; reject stale or untyped results."""
    if not math.isfinite(time_s):
        raise ValueError("time_s must be finite")
    if not isinstance(provider, AxialLoadProvider):
        return None
    frame = provider.get_segment_axial_loads()
    if frame is None:
        return None
    if not isinstance(frame, AxialLoadFrame):
        raise TypeError("provider must return AxialLoadFrame or None")
    if not math.isclose(frame.time_s, time_s, rel_tol=0, abs_tol=1e-12):
        return None
    return frame.to_dict()
