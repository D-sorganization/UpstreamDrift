"""Conservative structural support of the existing offline reference sampler.

This is an opt-in diagnostic, not a change to controller behavior. It covers
only raw position interpolation and the existing twice-applied numpy.gradient
rate tables; external filters, reference hooks, and controller dynamics are
outside its contract. A listed index can have zero weight through cancellation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceSupport:
    sample_times_s: tuple[float, ...]
    query_time_s: float
    position: tuple[int, ...]
    velocity: tuple[int, ...]
    acceleration: tuple[int, ...]

    def require_past_only(self) -> None:
        """Reject future reference rows in the modeled local sampler support.

        Passing does not establish causality of preprocessing or other inputs.
        """
        future = {
            name: tuple(
                i for i in indices if self.sample_times_s[i] > self.query_time_s
            )
            for name, indices in (
                ("position", self.position),
                ("velocity", self.velocity),
                ("acceleration", self.acceleration),
            )
        }
        if any(future.values()):
            raise ValueError(
                f"future reference samples affect modeled support: {future}"
            )


def _sample_rows(times: np.ndarray, t: float, lo: int, hi: int) -> set[int]:
    """Rows potentially used by np.interp, including endpoint clamping."""
    if t <= times[lo]:
        return {lo}
    if t >= times[hi - 1]:
        return {hi - 1}
    right = int(np.searchsorted(times, t, side="left"))
    return {right} if times[right] == t else {right - 1, right}


def _gradient_rows(index: int, lo: int, hi: int) -> set[int]:
    """Conservative edge-order-1 numpy.gradient stencil within one segment."""
    if index == lo:
        return {lo, lo + 1}
    if index == hi - 1:
        return {hi - 2, hi - 1}
    return {index - 1, index, index + 1}


def reference_support(
    times: Sequence[float], query_time_s: float, *, split_time_s: float | None = None
) -> ReferenceSupport:
    """Return possible reference-row dependencies for position, velocity and acceleration.

    The impact split partitions rate stencils and rate interpolation, matching
    ``reference_rates`` and ``sample_rate_table``. Position interpolation is
    unsplit in the existing GCV-20 controller.
    """
    clock = np.asarray(times, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        intervals = np.diff(clock) if clock.ndim == 1 else np.array([np.nan])
    if (
        clock.ndim != 1
        or clock.size < 2
        or not np.isfinite(clock).all()
        or not np.isfinite(intervals).all()
        or np.any(intervals <= 0)
        or not math.isfinite(query_time_s)
    ):
        raise ValueError("times must be finite, strictly increasing and have two rows")
    lo, hi = 0, len(clock)
    if split_time_s is not None:
        if not math.isfinite(split_time_s):
            raise ValueError("split time must be finite")
        last = int(np.searchsorted(clock, split_time_s, side="right")) - 1
        if not 1 <= last < len(clock) - 2:
            raise ValueError("split must leave two samples on each side")
        lo, hi = (
            (0, last + 1) if query_time_s <= split_time_s else (last + 1, len(clock))
        )
    positions = _sample_rows(clock, query_time_s, 0, len(clock))
    rate_rows = _sample_rows(clock, query_time_s, lo, hi)
    velocity = set().union(*(_gradient_rows(i, lo, hi) for i in rate_rows))
    acceleration = set().union(*(_gradient_rows(i, lo, hi) for i in velocity))
    return ReferenceSupport(
        tuple(float(t) for t in clock),
        float(query_time_s),
        tuple(sorted(positions)),
        tuple(sorted(velocity)),
        tuple(sorted(acceleration)),
    )
