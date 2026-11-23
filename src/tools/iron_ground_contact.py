"""Ground-first iron-strike diagnostics from signed sole-height samples.

Heights are metres above the local turf surface; negative means penetration.
Piecewise-linear interpolation resolves entry between samples. This observes
contact chronology only: it neither estimates turf forces nor applies losses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class IronGroundContact:
    """Observed penetrating contact, bounded by the supplied trace interval."""

    first_contact_time_s: float | None
    pre_ball_contact_duration_s: float
    max_depth_m: float

    @property
    def is_fat_shot(self) -> bool:
        """Whether penetrating ground contact precedes ball contact."""
        return self.pre_ball_contact_duration_s > 0.0


def analyze_iron_ground_contact(
    times_s: Sequence[float],
    sole_heights_m: Sequence[float],
    ball_contact_time_s: float,
) -> IronGroundContact:
    """Classify an observed iron strike using a piecewise-linear sole path.

    Preconditions: equal-length finite sequences with at least two samples,
    strictly increasing times, initial sole height >= 0, and finite ball time
    within the trace. Invalid records raise ValueError. The initial-height
    rule prevents inventing an entry time for a trace that starts in turf.

    Postconditions: nonnegative depth and pre-ball duration; first contact is
    None iff no segment penetrates. Tangency alone is not penetration. Entry
    exactly at ball contact is ball-first. Multiple contact intervals exclude
    airborne gaps; a trace ending in turf reports only its observed portion.
    No force, work, divot volume or calibrated physical acceptance is implied.
    """
    times = tuple(float(value) for value in times_s)
    heights = tuple(float(value) for value in sole_heights_m)
    if len(times) < 2 or len(times) != len(heights):
        raise ValueError("times and sole heights need equal lengths of at least two")
    if not all(math.isfinite(value) for value in (*times, *heights)):
        raise ValueError("times and sole heights must be finite")
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError("times must be strictly increasing")
    if not math.isfinite(times[-1] - times[0]):
        raise ValueError("trace duration must be finite")
    if heights[0] < 0:
        raise ValueError("trace must begin at or above turf to resolve ground entry")
    if not math.isfinite(ball_contact_time_s) or not (
        times[0] <= ball_contact_time_s <= times[-1]
    ):
        raise ValueError("ball contact time must be finite and within the trace")

    first_contact = None
    pre_ball_duration = 0.0
    for index in range(len(times) - 1):
        start, end = times[index : index + 2]
        z_start, z_end = heights[index : index + 2]
        if min(z_start, z_end) >= 0:
            continue
        # Convex weights avoid overflow in z_start - z_end at large heights.
        if max(z_start, z_end) >= 0:
            scale = max(abs(z_start), abs(z_end))
            left, right = abs(z_start) / scale, abs(z_end) / scale
            fraction = left / (left + right)
            crossing = (1 - fraction) * start + fraction * end
            if z_start >= 0:
                start = crossing
            else:
                end = crossing
        if first_contact is None:
            first_contact = start
        pre_ball_duration += max(0.0, min(end, ball_contact_time_s) - start)

    return IronGroundContact(
        first_contact_time_s=first_contact,
        pre_ball_contact_duration_s=pre_ball_duration,
        max_depth_m=max(0.0, -min(heights)),
    )
