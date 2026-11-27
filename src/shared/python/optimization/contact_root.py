"""Bracketed, approaching signed-distance contact root (#15493 Slice 1).

The impact event of a swing rollout is the first time the signed clearance
between the club face and the ball reaches zero *while closing*. This module
isolates that root-finding contract from any physics backend: the caller
supplies a scalar clearance function ``d(t)`` (positive = separated), and gets
back a bracketed root with an explicit convergence record, or a typed error.

Bisection is used deliberately: it is deterministic, never leaves the bracket,
and its iteration count is a closed-form function of the tolerance, so replay
bundles can record and re-verify it. A missed contact is never reported as
``NaN``; :func:`closest_approach_penalty` returns a smooth, finite cost instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DistanceFn = Callable[[float], float]

_DEFAULT_XTOL_S = 1e-12
_MISS_SAMPLES = 129
_MISS_EXPONENT = 32.0


class ContactBracketError(ValueError):
    """The interval does not bracket an approaching contact."""


@dataclass(frozen=True)
class ContactRoot:
    """Converged contact event.

    ``time_s`` is the penetrating end of the final bracket, so the clearance
    there is ``<= 0`` and the true root lies within ``bracket_width_s`` below.
    """

    time_s: float
    distance_m: float
    closing_speed_mps: float
    bracket_width_s: float
    iterations: int


def _clearance(distance_fn: DistanceFn, t: float) -> float:
    value = float(distance_fn(t))
    if not math.isfinite(value):
        raise ValueError(f"signed distance must be finite, got {value!r} at t={t!r}")
    return value


def _check_interval(t0_s: float, t1_s: float) -> None:
    if not (math.isfinite(t0_s) and math.isfinite(t1_s)):
        raise ValueError("interval ends must be finite")
    if not t1_s > t0_s:
        raise ValueError(f"t1_s must exceed t0_s, got {t0_s!r}..{t1_s!r}")


def find_contact_root(
    distance_fn: DistanceFn,
    t0_s: float,
    t1_s: float,
    *,
    xtol: float = _DEFAULT_XTOL_S,
) -> ContactRoot:
    """Bracket and converge the first-order approaching zero of ``distance_fn``.

    Preconditions: finite ``t0_s < t1_s``, ``xtol > 0``, ``d(t0_s) > 0`` (not yet
    touching) and ``d(t1_s) <= 0`` (in or past contact). Raises
    :class:`ContactBracketError` otherwise, or when the converged crossing is
    not closing (``d`` non-decreasing there).
    """
    _check_interval(t0_s, t1_s)
    if not (math.isfinite(xtol) and xtol > 0.0):
        raise ValueError(f"xtol must be finite and positive, got {xtol!r}")

    d_lo = _clearance(distance_fn, t0_s)
    d_hi = _clearance(distance_fn, t1_s)
    if d_lo <= 0.0:
        raise ContactBracketError(
            f"already in contact at t0_s (d={d_lo!r}); not an approach"
        )
    if d_hi > 0.0:
        raise ContactBracketError(
            f"no contact by t1_s (d={d_hi!r}); interval is not bracketed"
        )

    lo, hi = float(t0_s), float(t1_s)
    iterations = 0
    while hi - lo > xtol:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:  # float resolution reached before xtol
            break
        if _clearance(distance_fn, mid) <= 0.0:
            hi = mid
        else:
            lo = mid
        iterations += 1

    # Difference probes stay inside the caller's bracket (one-sided at its ends),
    # since the callback is only defined on [t0_s, t1_s].
    step = max(1e-6 * (t1_s - t0_s), 1e-12)
    a, b = max(float(t0_s), hi - step), min(float(t1_s), hi + step)
    closing = -(_clearance(distance_fn, b) - _clearance(distance_fn, a)) / (b - a)
    if not closing > 0.0:
        raise ContactBracketError(
            f"crossing at t={hi!r} is not approaching (closing speed {closing!r})"
        )

    return ContactRoot(
        time_s=hi,
        distance_m=_clearance(distance_fn, hi),
        closing_speed_mps=closing,
        bracket_width_s=hi - lo,
        iterations=iterations,
    )


def closest_approach_penalty(
    distance_fn: DistanceFn,
    t0_s: float,
    t1_s: float,
    *,
    samples: int = _MISS_SAMPLES,
) -> float:
    """Smooth, finite cost for a (possibly) missed contact; ``0.0`` once touching.

    The cost is ``m**2`` where ``m`` is a smooth minimum of the sampled
    clearances, the negative-exponent power mean
    ``m = d_min * mean((d_min / d_i) ** p) ** (-1 / p)`` with ``p = 32``. Unlike
    ``min`` it has no cusp where the closest sample changes, so the cost is C1
    in the trajectory parameters (and tends to ``0`` smoothly as any clearance
    reaches contact). It never under-estimates: ``d_min**2 <= cost <=
    n**(2/p) * d_min**2``. Raises ``ValueError`` if the cost would overflow.
    """
    _check_interval(t0_s, t1_s)
    if samples < 2:
        raise ValueError(f"samples must be at least 2, got {samples!r}")
    times = np.linspace(t0_s, t1_s, samples)
    clearances = np.array([_clearance(distance_fn, float(t)) for t in times])
    closest = float(clearances.min())
    if closest <= 0.0:
        return 0.0
    # Ratios are <= 1, so the power mean cannot overflow even for huge d_min.
    ratios = (closest / clearances) ** _MISS_EXPONENT
    smooth_min = closest * float(np.mean(ratios)) ** (-1.0 / _MISS_EXPONENT)
    cost = smooth_min * smooth_min
    if not math.isfinite(cost):
        raise ValueError(
            f"clearance {closest!r} overflows the squared miss cost; rescale it"
        )
    return cost
