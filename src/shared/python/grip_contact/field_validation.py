"""Shared grip field validation helpers (engine-free, no grip_spec import)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.shared.python.contracts import require
from src.shared.python.grip_contact.contact_topology import GripStyle


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


def validate_grip_style_fields(style: object, trigger_finger: object) -> None:
    """Validate grip style enum and trigger-finger flag."""
    if not isinstance(style, GripStyle):
        raise TypeError(f"style must be a GripStyle, got {style!r}")
    if not isinstance(trigger_finger, bool):
        raise TypeError(f"trigger_finger must be a bool, got {trigger_finger!r}")


__all__ = ["require_fields_in_range", "validate_grip_style_fields"]
