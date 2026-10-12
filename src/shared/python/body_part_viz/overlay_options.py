"""Strict display-only opacity options, without a native SDK dependency."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
import math


@dataclass(frozen=True)
class ShapeOverlayOptions:
    """Authored display opacity; it never changes model or fitted coordinates."""

    opacity: float = 0.35

    def __post_init__(self) -> None:
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, Real):
            raise TypeError("Shape opacity must be a real number")
        if not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValueError("Shape opacity must be finite and within [0, 1]")
        object.__setattr__(self, "opacity", float(self.opacity))

    def to_record(self) -> dict[str, float]:
        """Return the exact JSON options contract."""
        return {"opacity": self.opacity}

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ShapeOverlayOptions":
        """Reject unknown/missing keys and numeric coercion."""
        if not isinstance(record, Mapping) or set(record) != {"opacity"}:
            raise ValueError("Shape options require exactly opacity")
        return cls(record["opacity"])  # type: ignore[arg-type]
