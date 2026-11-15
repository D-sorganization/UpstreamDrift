"""Field-range validation helpers for grip authoring types."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.shared.python.contracts import require


def require_fields_in_range(
    obj: object, bounds: Sequence[tuple[str, float, float]]
) -> None:
    """Require each named attribute of ``obj`` to be a finite number in range."""
    for name, lo, hi in bounds:
        value = getattr(obj, name)
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"{name} must be a finite number, got {value!r}",
        )
        require(lo <= value <= hi, f"{name} must be in [{lo}, {hi}], got {value}")


__all__ = ["require_fields_in_range"]
