"""Wrapped (circular) outcome dispersion for angle outcomes (NSX-82, #12711).

Implements ``docs/research/loop_nullspace/loop_nullspace.tex`` ``sec:var-face``
(epic rule V3).  Angles are compared through the wrapped error

    e = atan2(sin(phi - phi_tgt), cos(phi - phi_tgt))  in (-pi, pi]

and the report is bias (circular mean of ``e``), standard deviation,
interquartile range, tolerance exceedance ``P(|e| > eps)`` and miss
probability.  The coefficient of variation is banned for zero-centred
outcomes (it is NaN in :class:`MetricStatistics`, not 0).

Declared choices where the specification fixes no formula:

* **SD** is the population root-mean-square of the wrapped deviations
  ``d = wrap(e - bias)`` (``ddof=0``, as in :class:`MetricStatistics`).
* **IQR** is the interquartile range of ``d``.
* **Miss probability** is ``P(|e| > miss_half_width)``; the specification does
  not define a miss criterion, so the half-width must be declared by the
  caller, and the field is NaN with ``miss_reason`` when it is not.
* **Bootstrap CIs** are seeded percentile intervals over resampled trials
  (the bias interval is taken on wrapped offsets from the point estimate so
  it is valid near the +-180 degree wrap).
* ``mean_resultant_length`` is reported because the circular mean is
  ill-defined when it is near 0 (e.g. errors split between 0 and 180 degrees).

Face rows are ``UNAVAILABLE`` when axial club rotation is not observed; that
gating belongs to the caller, which must not construct this object then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["OutcomeDispersion", "compute_outcome_dispersion", "wrap_angle"]

_PERIODS = {"rad": 2.0 * math.pi, "deg": 360.0}
_MIN_BOOT = 20
_BOOT_CHUNK_ELEMENTS = 2_000_000  # bounds peak memory of the resample matrix
Interval = tuple[float, float]


def wrap_angle(values: ArrayLike, period: float = 2.0 * math.pi) -> np.ndarray:
    """Wrap to ``(-period/2, period/2]``.

    Pre: ``period > 0``.  Post: every element in the half-open interval.
    """
    if not (math.isfinite(period) and period > 0):
        raise ValueError(f"period must be positive and finite, got {period}")
    x = np.asarray(values, dtype=float)
    return x - period * np.ceil(x / period - 0.5)


@dataclass(frozen=True)
class OutcomeDispersion:
    """Wrapped dispersion of an angle outcome, in the caller's angle unit."""

    unit: str
    n_samples: int
    target: float
    tolerance: float
    miss_half_width: float | None
    bias: float
    sd: float
    iqr: float
    tolerance_exceedance: float
    miss_probability: float
    mean_resultant_length: float
    bias_ci: Interval
    sd_ci: Interval
    iqr_ci: Interval
    tolerance_exceedance_ci: Interval
    miss_probability_ci: Interval
    ci_level: float
    n_boot: int
    seed: int | None
    miss_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe dictionary (NaN becomes ``None``)."""

        def clean(v: object) -> object:
            if isinstance(v, tuple):
                return [clean(x) for x in v]
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        return {k: clean(v) for k, v in self.__dict__.items()}


def _circular_stats(e: np.ndarray, period: float) -> tuple[Any, Any, Any, Any]:
    """(bias, sd, iqr, resultant length) of wrapped errors over the last axis."""
    theta = e * (2.0 * math.pi / period)
    s, c = np.sin(theta).mean(axis=-1), np.cos(theta).mean(axis=-1)
    bias = np.arctan2(s, c) * period / (2.0 * math.pi)
    d = wrap_angle(e - bias[..., None], period)
    p25, p75 = np.percentile(d, [25, 75], axis=-1)
    return bias, np.sqrt(np.mean(d * d, axis=-1)), p75 - p25, np.hypot(s, c)


def _exceed(e: np.ndarray, half_width: float) -> Any:
    return np.mean(np.abs(e) > half_width, axis=-1)


def _validate(
    angles: np.ndarray,
    target: float,
    tolerance: float,
    miss_half_width: float | None,
    angle_unit: str,
    ci_level: float,
    n_boot: int,
) -> None:
    if angle_unit not in _PERIODS:
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got {angle_unit!r}")
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("angles must be a non-empty 1-D array")
    if not np.isfinite(angles).all():
        raise ValueError("angles must be finite")
    if not math.isfinite(target):
        raise ValueError(f"target must be finite, got {target}")
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be positive and finite, got {tolerance}")
    if miss_half_width is not None and not (
        math.isfinite(miss_half_width) and miss_half_width > 0
    ):
        raise ValueError(
            f"miss_half_width must be positive and finite, got {miss_half_width}"
        )
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if n_boot < _MIN_BOOT:
        raise ValueError(f"n_boot must be >= {_MIN_BOOT}, got {n_boot}")


def _bootstrap(
    e: np.ndarray,
    period: float,
    tolerance: float,
    miss: float | None,
    n_boot: int,
    seed: int | None,
) -> np.ndarray:
    """Rows of (bias offset, sd, iqr, exceedance, miss) per resample."""
    rng = np.random.default_rng(seed)
    point_bias = float(_circular_stats(e, period)[0])
    rows = np.full((n_boot, 5), np.nan)
    chunk = max(1, _BOOT_CHUNK_ELEMENTS // e.size)
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        eb = e[rng.integers(0, e.size, size=(stop - start, e.size))]
        bias, sd, iqr, _ = _circular_stats(eb, period)
        rows[start:stop, 0] = wrap_angle(bias - point_bias, period)
        rows[start:stop, 1] = sd
        rows[start:stop, 2] = iqr
        rows[start:stop, 3] = _exceed(eb, tolerance)
        if miss is not None:
            rows[start:stop, 4] = _exceed(eb, miss)
    return rows


def compute_outcome_dispersion(
    angles: ArrayLike,
    target: float,
    tolerance: float,
    *,
    miss_half_width: float | None = None,
    angle_unit: str = "rad",
    ci_level: float = 0.95,
    n_boot: int = 1000,
    seed: int | None = None,
) -> OutcomeDispersion:
    """Wrapped dispersion of ``angles`` about ``target`` with bootstrap CIs.

    Pre: ``angles`` finite, 1-D, non-empty; ``tolerance > 0``;
    ``miss_half_width`` positive if given; ``angle_unit`` in {'rad', 'deg'};
    ``0 < ci_level < 1``; ``n_boot >= 20``.
    Post: statistics are invariant to adding whole turns to ``angles`` or
    ``target``; every finite CI brackets its point estimate; results are
    identical for an identical ``seed``; ``sec:var-face``.
    """
    a = np.asarray(angles, dtype=float)
    _validate(a, target, tolerance, miss_half_width, angle_unit, ci_level, n_boot)
    period = _PERIODS[angle_unit]
    e = wrap_angle(a - target, period)
    bias_a, sd_a, iqr_a, resultant_a = _circular_stats(e, period)
    bias, sd, iqr, resultant = (float(v) for v in (bias_a, sd_a, iqr_a, resultant_a))
    exceed = float(_exceed(e, tolerance))
    boot = _bootstrap(e, period, tolerance, miss_half_width, n_boot, seed)

    lo_q, hi_q = 50.0 * (1.0 - ci_level), 50.0 * (1.0 + ci_level)

    def ci(col: int, estimate: float, offset: bool = False) -> Interval:
        if math.isnan(estimate):
            return (math.nan, math.nan)
        lo, hi = np.percentile(boot[:, col], [lo_q, hi_q])
        if offset:  # bias: wrapped offsets around the point estimate
            return (bias + float(lo), bias + float(hi))
        return (min(float(lo), estimate), max(float(hi), estimate))

    if miss_half_width is None:
        miss_p = math.nan
        reason: str | None = (
            "miss probability undefined: no miss_half_width declared "
            "(loop_nullspace.tex sec:var-face defines no miss criterion)"
        )
    else:
        miss_p, reason = float(_exceed(e, miss_half_width)), None
    return OutcomeDispersion(
        unit=angle_unit,
        n_samples=int(a.size),
        target=float(target),
        tolerance=float(tolerance),
        miss_half_width=miss_half_width,
        bias=bias,
        sd=sd,
        iqr=iqr,
        tolerance_exceedance=exceed,
        miss_probability=miss_p,
        mean_resultant_length=resultant,
        bias_ci=ci(0, bias, offset=True),
        sd_ci=ci(1, sd),
        iqr_ci=ci(2, iqr),
        tolerance_exceedance_ci=ci(3, exceed),
        miss_probability_ci=ci(4, miss_p),
        ci_level=ci_level,
        n_boot=n_boot,
        seed=seed,
        miss_reason=reason,
    )
