"""Shared validation for time-aligned segment scalar/load series."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType

__all__ = [
    "coerce_finite_real_fields",
    "normalize_segment_samples",
    "validate_strictly_increasing_times",
]


def coerce_finite_real_fields(
    target: object,
    field_names: Sequence[str],
    *,
    nonnegative: Sequence[str] = (),
) -> None:
    """Coerce named real fields to float; raise on non-finite or wrong types."""
    for name in field_names:
        value = getattr(target, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        value_f = float(value)
        if name in nonnegative and value_f < 0:
            raise ValueError(f"{name} must be nonnegative")
        object.__setattr__(target, name, value_f)


def validate_strictly_increasing_times(time_s: Sequence[float]) -> tuple[float, ...]:
    """Return a finite, strictly increasing time axis."""
    times = tuple(time_s)
    if not times or any(
        isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t)
        for t in times
    ):
        raise ValueError("time_s must contain finite real sample times")
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError("time_s must be strictly increasing")
    return tuple(float(t) for t in times)


def normalize_segment_samples(
    times: tuple[float, ...],
    values: Mapping[str, Sequence[float | None]],
    *,
    segment_value_label: str,
) -> MappingProxyType[str, tuple[float | None, ...]]:
    """Validate per-segment samples and return an immutable mapping."""
    if not isinstance(values, Mapping):
        raise TypeError("values must be a segment-to-samples mapping")
    copied: dict[str, tuple[float | None, ...]] = {}
    for segment, samples in values.items():
        if not isinstance(segment, str) or not segment.strip():
            raise ValueError("segment identifiers must be nonempty strings")
        series = tuple(samples)
        if len(series) != len(times):
            raise ValueError(f"{segment}: {segment_value_label} must match time_s")
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, Real))
            for value in series
        ):
            raise TypeError(f"{segment_value_label} must be real numbers or None")
        copied[segment] = tuple(
            float(value) if value is not None and math.isfinite(value) else None
            for value in series
        )
    return MappingProxyType(copied)
