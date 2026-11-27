"""Single-timestep per-segment scalar samples for scene colouring (#15399)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .series_validation import (
    normalize_segment_samples,
    validate_strictly_increasing_times,
)

__all__ = ["SegmentScalarFrame"]


@dataclass(frozen=True)
class SegmentScalarFrame:
    """Immutable scalar sample aligned with body-part segment IDs."""

    time_s: float
    values: Mapping[str, float | None]
    source: str
    units: str
    quantity_label: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a nonempty string")
        if not isinstance(self.quantity_label, str) or not self.quantity_label.strip():
            raise ValueError("quantity_label must be a non-empty string")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping")
        series = normalize_segment_samples(
            validate_strictly_increasing_times((self.time_s,)),
            {name: (value,) for name, value in self.values.items()},
            segment_value_label="scalar",
        )
        object.__setattr__(
            self,
            "values",
            MappingProxyType({name: vals[0] for name, vals in series.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "values": dict(self.values),
            "source": self.source,
            "units": self.units,
            "quantity_label": self.quantity_label,
        }
