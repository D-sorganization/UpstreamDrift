"""The per-hand grip force limit ``grip_force_limit_n`` (HRG-HX4, #13761).

``grip_force_limit_n`` is a force in newtons: the cap on one hand's total pad
normal force (or, on a weld/bushing attachment, on the magnitude of the
bushing force). It is not ``GripPose.grip_strength_deg``, which is a visual
roll in degrees. No torque limit is defined. No physiological value is claimed.

The default is the explicit string ``"unlimited"`` so that configuration and
capability reports show it; a finite limit must be a finite number above zero.
"""

from __future__ import annotations

import math
from typing import Final

from src.shared.python.contracts import require

UNLIMITED: Final = "unlimited"
GripForceLimit = float | str


def validate_grip_force_limit(
    value: object, name: str = "grip_force_limit_n"
) -> GripForceLimit:
    """Return ``value`` if it is ``"unlimited"`` or a finite number > 0 N.

    Raises ``TypeError`` for a non-number and ``ValueError`` for any other
    string, a non-finite number or a number that is not above zero.
    """
    if value == UNLIMITED and isinstance(value, str):
        return UNLIMITED
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be 'unlimited' or a number, got {value!r}")
    require(
        math.isfinite(value) and value > 0.0,
        f"{name} must be 'unlimited' or a finite number > 0 N, got {value!r}",
    )
    return float(value)


def finite_limit_n(value: GripForceLimit) -> float | None:
    """The limit in newtons, or ``None`` when it is ``"unlimited"``."""
    return None if value == UNLIMITED else float(value)
