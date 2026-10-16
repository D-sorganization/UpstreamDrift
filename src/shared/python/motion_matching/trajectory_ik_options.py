"""Parameter object for the trajectory IK solver (issue #12845).

``solve_full_body_ik_trajectory`` historically took five tuning keywords.
They are grouped here so the public function stays inside the architecture
parameter budget while remaining keyword-compatible with existing callers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from src.shared.python.motion_matching.marker_calibration import Array

ClosureFn = Callable[[Array], Array]
CoordinateBounds = tuple[Array, Array]

__all__ = [
    "ClosureFn",
    "CoordinateBounds",
    "TrajectoryIkOptions",
    "resolve_trajectory_options",
]


@dataclass(frozen=True)
class TrajectoryIkOptions:
    """Tuning options for per-frame trajectory IK.

    Omitting ``coordinate_bounds`` preserves the original Levenberg-Marquardt
    solver; explicit finite bounds opt into bounded TRF.
    """

    closure_fn: ClosureFn | None = None
    closure_weight: float = 10.0
    reg_weight: float = 1e-3
    max_nfev: int = 50
    coordinate_bounds: CoordinateBounds | None = None

    def __post_init__(self) -> None:
        if self.closure_weight < 0:
            raise ValueError("closure_weight must be non-negative")
        if self.reg_weight < 0:
            raise ValueError("reg_weight must be non-negative")


def resolve_trajectory_options(
    options: TrajectoryIkOptions | None,
    overrides: dict[str, Any],
) -> TrajectoryIkOptions:
    """Merge keyword overrides onto ``options`` (defaults when ``None``).

    Raises:
        TypeError: if ``options`` has the wrong type or an override name is
            not a ``TrajectoryIkOptions`` field.
        ValueError: if a merged weight is negative.
    """
    if options is not None and not isinstance(options, TrajectoryIkOptions):
        raise TypeError("options must be a TrajectoryIkOptions or None")
    base = TrajectoryIkOptions() if options is None else options
    if not overrides:
        return base
    return replace(base, **overrides)
