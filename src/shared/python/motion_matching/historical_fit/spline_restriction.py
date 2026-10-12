"""Lossless Hermite interval restriction, without source or native admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from src.shared.python.estimation import CubicHermiteSplineTrajectory
from .contracts import ImageSplineStart

SPLINE_RESTRICTION_SCHEMA = "image-spline-interval-restriction/1"


def _interval(
    start: ImageSplineStart, lower: float, upper: float
) -> tuple[float, float]:
    if not isinstance(start, ImageSplineStart):
        raise ValueError("Restriction requires a typed immutable spline start")
    endpoints: list[float] = []
    for value in (lower, upper):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Restriction endpoints must be finite real numbers")
        try:
            endpoint = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "Restriction endpoints must be finite real numbers"
            ) from exc
        if not np.isfinite(endpoint):
            raise ValueError("Restriction endpoints must be finite real numbers")
        endpoints.append(endpoint)
    first, last = endpoints
    if first >= last:
        raise ValueError("Restriction interval must be nonempty and increasing")
    if first < start.knot_times[0] or last > start.knot_times[-1]:
        raise ValueError("Restriction cannot widen or extrapolate the parent interval")
    return first, last


def _restricted_start(
    start: ImageSplineStart, lower: float, upper: float
) -> ImageSplineStart:
    first, last = _interval(start, lower, upper)
    if (first, last) == (start.knot_times[0], start.knot_times[-1]):
        return start
    knots = (first, *(t for t in start.knot_times if first < t < last), last)
    parent = CubicHermiteSplineTrajectory(
        np.asarray(start.knot_times), len(start.free_coordinates)
    )
    values = parent.evaluate(np.asarray(start.spline_coefficients), np.asarray(knots))
    q, v = parent.unpack(np.asarray(start.spline_coefficients))
    # Preserve existing knot bytes; newly cut endpoints use the canonical evaluator.
    for index, time in enumerate(knots):
        if time in start.knot_times:
            old_index = start.knot_times.index(time)
            values.q[index], values.v[index] = q[old_index], v[old_index]
    child = CubicHermiteSplineTrajectory(np.asarray(knots), len(start.free_coordinates))
    return ImageSplineStart.from_coefficients(
        knots,
        child.pack(values.q, values.v),
        start.coordinate_order,
        start.free_coordinates,
        start.model_sha,
    )


@dataclass(frozen=True)
class SplineIntervalRestriction:
    """Immutable parent/new coefficient lineage, not authenticated source evidence.

    Both starts retain model, native order and free order. Construction verifies
    the declared restricted polynomial using the canonical evaluator and packer.
    This receipt neither carries nor changes locked native reference positions;
    callers must preserve those separately. Float times are numerical coordinates,
    not authenticated capture PTS or a qualified physical clock. Bounds and native
    binding remain independent caller obligations. Record decoding repeats these
    postconditions and never repairs a forged child.
    """

    original_start: ImageSplineStart
    restricted_start: ImageSplineStart

    def __post_init__(self) -> None:
        if not isinstance(self.original_start, ImageSplineStart) or not isinstance(
            self.restricted_start, ImageSplineStart
        ):
            raise ValueError(
                "Restriction receipt requires typed immutable spline starts"
            )
        expected = _restricted_start(
            self.original_start,
            self.restricted_start.knot_times[0],
            self.restricted_start.knot_times[-1],
        )
        if self.restricted_start != expected:
            raise ValueError(
                "Restricted spline differs from canonical parent restriction"
            )

    def to_record(self) -> dict[str, Any]:
        """Return detached complete parent/new coefficient and model identities."""
        return {
            "schema": SPLINE_RESTRICTION_SCHEMA,
            "original_start": self.original_start.to_record(),
            "restricted_start": self.restricted_start.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SplineIntervalRestriction:
        """Validate exact record fields and truthful canonical polynomial lineage."""
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "schema",
                "original_start",
                "restricted_start",
            }
            or record["schema"] != SPLINE_RESTRICTION_SCHEMA
        ):
            raise ValueError(
                "Restriction receipt requires its complete versioned schema"
            )
        return cls(
            ImageSplineStart.from_record(record["original_start"]),
            ImageSplineStart.from_record(record["restricted_start"]),
        )


def restrict_image_spline_interval(
    start: ImageSplineStart,
    lower: float,
    upper: float,
) -> SplineIntervalRestriction:
    """Retain the parent polynomial on a finite nonempty contained interval.

    Original interior knots and their q/v bytes remain; new endpoints evaluate
    the existing Hermite segment. Full-domain restriction returns the exact
    original start. q/v agree to floating roundoff throughout the new interval.
    Acceleration is segmentwise preserved: at an upper cut exactly on a C1-only
    interior knot, the new final endpoint uses the retained LEFT segment value,
    whereas the parent's evaluator chooses its right segment. No C2 continuity
    is promised. Elsewhere the canonical segment convention is unchanged.

    No positions are projected, no slopes zeroed and no clock qualified. Preserve
    locked reference values and authenticate capture/model scope separately. If
    authored bounds are available, validate the returned ordinary q/v packing
    through public HermiteBoundsDomain.encode; its term 'physical' distinguishes
    coefficients from normalized decisions, not calibrated historical timing.
    """
    return SplineIntervalRestriction(start, _restricted_start(start, lower, upper))
