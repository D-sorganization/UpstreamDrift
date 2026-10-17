"""Linear interpolation of a sampled channel with a bias *estimate* (OF-05).

The interpolation bias is estimated pointwise as
``|(t - t_i)(t_{i+1} - t)| / 2 * M2`` where ``M2`` is the larger of the
neighbouring ``2 * |f[a, b, c]|`` (non-uniform second divided differences) and
an optional declared floor on ``|y''|``.  It is zero at a sample and at most
``h**2 / 8 * M2``.

Validity: this is an **estimate, not a bound**.  A second divided difference is
a mean of ``y''/2`` over three samples, so the estimate holds when the signal is
band-limited well below the sample rate, so that ``y''`` is bounded by its
neighbouring second divided differences.  It is NOT valid across an
under-sampled impact transient (a feature shorter than about two sample
intervals is invisible to every divided difference).  Callers must exclude or
separately bound such intervals.

Pure numpy; no engine or GUI imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.force_plates.contracts import Array

__all__ = ["Channel", "Estimate", "interpolate", "locate", "snap"]

#: Multiplier turning a second divided difference into an estimate of ``|y''|``.
_CURVATURE_FACTOR = 2.0


@dataclass(frozen=True, eq=False)
class Channel:
    """One sampled series ``values`` on ``times`` plus its curvature context.

    ``label`` and ``floor_option`` only name the series and the
    ``ReadingOptions`` field that would supply ``curvature_floor`` (a declared
    floor on ``|y''|`` in the series' units per s^2) in error messages.
    """

    times: Array
    values: Array
    curvature_floor: float | None
    label: str
    floor_option: str


@dataclass(frozen=True)
class Estimate:
    """Interpolated value, interpolation-bias estimate and skew rate ``|dy/dt|``."""

    value: float
    bias: float
    rate: float


def snap(times: Array, t: float, tolerance: float) -> float:
    """``t`` moved onto its nearest sample when it lies within ``tolerance``.

    Removes the floating-point residue of a clock offset (``1.015 - 1.0`` is
    not ``0.015``), so a time at a sample reads as that sample: it is neither
    refused as an extrapolation nor given a spurious interpolation bias.
    """
    nearest = float(times[int(np.argmin(np.abs(times - t)))])
    return nearest if abs(t - nearest) <= tolerance else t


def locate(times: Array, t: float) -> int:
    """Index ``i`` with ``times[i] <= t <= times[i + 1]``; refuses extrapolation."""
    if not times[0] <= t <= times[-1]:
        raise ValueError(
            f"trial time {t} s is outside the trial span "
            f"[{times[0]}, {times[-1]}] s; extrapolation is refused"
        )
    return min(int(np.searchsorted(times, t, side="right")) - 1, times.shape[0] - 2)


def _second_difference(times: Array, y: Array, start: int) -> float | None:
    """Second divided difference over samples ``start..start+2``, if finite."""
    if start < 0 or start + 2 >= times.shape[0]:
        return None
    t0, t1, t2 = times[start : start + 3]
    y0, y1, y2 = y[start : start + 3]
    value = ((y2 - y1) / (t2 - t1) - (y1 - y0) / (t1 - t0)) / (t2 - t0)
    return float(value) if math.isfinite(value) else None


def _curvature(channel: Channel, index: int, t: float) -> float:
    """``M2`` for interval ``index``: estimate floored by the declared floor.

    Raises ``ValueError`` when no neighbouring triple is usable and no floor is
    declared: a missing curvature must never read as zero bias.
    """
    triples = (
        _second_difference(channel.times, channel.values, index - 1),
        _second_difference(channel.times, channel.values, index),
    )
    found = [_CURVATURE_FACTOR * abs(dd) for dd in triples if dd is not None]
    floor = channel.curvature_floor
    if not found and floor is None:
        raise ValueError(
            f"curvature unavailable for {channel.label} at trial time {t} s; "
            f"declare ReadingOptions.{channel.floor_option}"
        )
    return max([*found, floor if floor is not None else 0.0])


def _skew_rate(channel: Channel, index: int, t: float, slope: float) -> float:
    """``|dy/dt|``; at an interior sample the steeper adjacent interval wins."""
    times, y = channel.times, channel.values
    rate = abs(slope)
    if index > 0 and t == times[index]:
        rate = max(
            rate, abs((y[index] - y[index - 1]) / (times[index] - times[index - 1]))
        )
    return rate


def interpolate(channel: Channel, index: int, t: float) -> Estimate:
    """Linear interpolation in interval ``index`` at ``t`` with bias estimate."""
    t0, t1 = float(channel.times[index]), float(channel.times[index + 1])
    y0, y1 = float(channel.values[index]), float(channel.values[index + 1])
    slope = (y1 - y0) / (t1 - t0)
    weight = abs((t - t0) * (t1 - t)) / 2.0
    bias = weight * _curvature(channel, index, t) if weight > 0.0 else 0.0
    return Estimate(
        value=y0 + slope * (t - t0),
        bias=bias,
        rate=_skew_rate(channel, index, t, slope),
    )
