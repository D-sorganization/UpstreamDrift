"""Forgiveness sensitivities over the measured delivery ranges (issue #8614).

A wedge is forgiving when carry does *not* track the thing the player fails to
control. Wivou et al. (2016) measured how strongly it does track, on real
bunker shots:

* carry vs entry distance behind the ball: **r = -0.98**
* carry vs divot depth: **r = -0.91**

Those are the numbers to beat, and lower magnitude is better. They are
**measurements from published work, not outputs of this model** -- the
distinction matters, and this package has been burnt once already by presenting
a borrowed constant as a measurement (#7999).

Reported per factor:

============================ =============================================================
Quantity                     Definition
============================ =============================================================
``correlation_r``            Pearson correlation of carry against the factor. Directly
                             comparable with the published baselines.
``slope_m_per_unit``         Least-squares slope, carry [m] per factor unit.
``carry_change_over_span_m`` ``slope * (sampled span)`` -- how much carry moves across the
                             swept range.
``fractional_carry_change``  That change divided by the target carry; dimensionless, and
                             the number a designer compares across factors of different
                             units. Smaller is more forgiving.
============================ =============================================================

Pearson ``r`` is scale-free, which makes it comparable with the literature but
blind to *how much* carry actually moved: a factor can correlate at -0.99 and
shift carry by one metre. That is why the fractional change is reported
alongside, and why the ranking uses it. Variance-based sensitivity over a
multi-factor design lives in :mod:`bunkershot3d.study.sensitivity`; this module
is deliberately the one-factor linear form the baselines are stated in.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

__all__ = [
    "SWEEP_RANGES",
    "WIVOU_2016_CARRY_CORRELATION",
    "FactorSensitivity",
    "ForgivenessReport",
    "SweepRange",
    "forgiveness_report",
    "forgiveness_sensitivity",
]

#: Penetrometer readings are quoted in kg/cm^2 in the turf literature; this is
#: the conversion to the SI unit the model works in.
_KGF_PER_CM2_TO_KPA = 98.0665


@dataclass(frozen=True)
class SweepRange:
    """A factor's sweep range, in SI, with the source it came from.

    Attributes:
        name: Factor name, matching the keyword used across this package.
        low: Lower bound, SI.
        high: Upper bound, SI.
        unit: SI unit symbol.
        source: Where the range comes from. Recorded so a borrowed range is
            never mistaken for a measured one.
    """

    name: str
    low: float
    high: float
    unit: str
    source: str

    def __post_init__(self) -> None:
        """Validate the range.

        Raises:
            ValueError: If the bounds are not finite or not ordered.
        """
        if not (np.isfinite(self.low) and np.isfinite(self.high)):
            raise ValueError(f"range {self.name!r} must have finite bounds")
        if self.high <= self.low:
            raise ValueError(
                f"range {self.name!r} must have high > low, got {self.low} to {self.high}"
            )

    @property
    def span(self) -> float:
        """Width of the range in its own unit."""
        return self.high - self.low

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies inside the closed range."""
        return bool(self.low <= value <= self.high)


def _sweep_ranges() -> dict[str, SweepRange]:
    """Build the sweep-range registry.

    Returns:
        Mapping of factor name to its range. Angles are radians, lengths metres,
        pressures kilopascals.
    """
    entries = (
        SweepRange(
            "entry_distance_behind_ball_m",
            0.025,
            0.150,
            "m",
            "Wivou et al. 2016 delivery data (measured 0.080-0.280 m)",
        ),
        SweepRange(
            "divot_depth_m",
            0.020,
            0.060,
            "m",
            "Wivou et al. 2016 delivery data (measured 0.025-0.052 m)",
        ),
        SweepRange(
            "attack_angle_rad",
            float(np.radians(-12.0)),
            float(np.radians(-2.0)),
            "rad",
            "tour delivery; the largest single term in presentation to velocity",
        ),
        SweepRange(
            "face_open_angle_rad",
            0.0,
            float(np.radians(30.0)),
            "rad",
            "propagates as delta_loft = delta_bounce = Omega cos(lie)",
        ),
        SweepRange(
            "shaft_lean_rad",
            float(np.radians(4.0)),
            float(np.radians(14.0)),
            "rad",
            "tour delivery; subtracts from loft and bounce degree for degree",
        ),
        SweepRange(
            "strike_location_heel_toe_m",
            -0.015,
            0.015,
            "m",
            "changes the sand load distribution without face contact",
        ),
        SweepRange(
            "lie_deviation_rad",
            float(np.radians(-5.0)),
            float(np.radians(5.0)),
            "rad",
            "modulated by the sole rocker radius",
        ),
        SweepRange(
            "sand_firmness_kPa",
            1.6 * _KGF_PER_CM2_TO_KPA,
            2.8 * _KGF_PER_CM2_TO_KPA,
            "kPa",
            "golf-ball penetrometer 1.6/2.0/2.4/2.8 kg/cm^2 (USGA firmness bands)",
        ),
        SweepRange(
            "sand_depth_m",
            0.025,
            0.150,
            "m",
            "USGA 100-150 mm floors, 50-75 mm faces",
        ),
    )
    return {entry.name: entry for entry in entries}


#: Sweep ranges for the forgiveness study, SI, each carrying its source.
SWEEP_RANGES: MappingProxyType[str, SweepRange] = MappingProxyType(_sweep_ranges())

#: Published carry correlations to beat (Wivou et al. 2016). **Measured values
#: from the literature, not outputs of this model.** Lower magnitude = more
#: forgiving.
WIVOU_2016_CARRY_CORRELATION: MappingProxyType[str, float] = MappingProxyType(
    {"entry_distance_behind_ball_m": -0.98, "divot_depth_m": -0.91}
)


@dataclass(frozen=True)
class FactorSensitivity:
    """How strongly carry tracks one delivery factor.

    Attributes:
        factor: Factor name.
        unit: SI unit of the factor.
        n_samples: Number of paired samples.
        correlation_r: Pearson correlation of carry against the factor.
        slope_m_per_unit: Least-squares slope [m per factor unit].
        intercept_m: Least-squares intercept [m].
        sampled_low: Smallest factor value sampled.
        sampled_high: Largest factor value sampled.
        carry_change_over_span_m: ``slope * (sampled_high - sampled_low)``.
        fractional_carry_change: That change divided by the target carry.
        target_carry_m: Target carry the fraction is taken against.
        baseline_r: Published correlation to beat, or ``None``.
        covers_declared_range: Whether the samples span the registered sweep
            range, or ``None`` when the factor is not in :data:`SWEEP_RANGES`.
    """

    factor: str
    unit: str
    n_samples: int
    correlation_r: float
    slope_m_per_unit: float
    intercept_m: float
    sampled_low: float
    sampled_high: float
    carry_change_over_span_m: float
    fractional_carry_change: float
    target_carry_m: float
    baseline_r: float | None
    covers_declared_range: bool | None

    @property
    def more_forgiving_than_baseline(self) -> bool | None:
        """Whether ``|r|`` is below the published baseline, or ``None``.

        Returns:
            True when this design tracks the factor less strongly than the
            published measurement, i.e. is more forgiving; ``None`` when no
            baseline exists for the factor.
        """
        if self.baseline_r is None:
            return None
        return bool(abs(self.correlation_r) < abs(self.baseline_r))


def _linear_fit(factor: np.ndarray, carry_m: np.ndarray) -> tuple[float, float, float]:
    """Return ``(slope, intercept, pearson_r)`` for carry against a factor.

    Args:
        factor: ``(n,)`` factor samples.
        carry_m: ``(n,)`` carry samples [m].

    Returns:
        Slope [m per factor unit], intercept [m], and Pearson ``r``.

    Raises:
        ValueError: If either series has zero variance, which leaves both the
            slope and the correlation undefined.
    """
    factor_centred = factor - factor.mean()
    carry_centred = carry_m - carry_m.mean()
    factor_ss = float(factor_centred @ factor_centred)
    carry_ss = float(carry_centred @ carry_centred)
    if factor_ss <= 0.0:
        raise ValueError(
            "the factor samples are all equal, so no sensitivity can be measured"
        )
    if carry_ss <= 0.0:
        raise ValueError(
            "the carry samples are all equal, so the correlation is undefined; "
            "report a zero slope explicitly rather than inferring one"
        )
    covariance = float(factor_centred @ carry_centred)
    slope = covariance / factor_ss
    intercept = float(carry_m.mean() - slope * factor.mean())
    return slope, intercept, covariance / np.sqrt(factor_ss * carry_ss)


def _paired_samples(
    factor_values: np.ndarray, carry_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the paired sample arrays.

    Args:
        factor_values: Factor samples.
        carry_m: Carry samples [m].

    Returns:
        The two arrays as 1-D float64.

    Raises:
        ValueError: If the lengths disagree, fewer than three pairs are given,
            or a value is not finite. Three is the minimum at which a
            correlation says anything at all: two points always give |r| = 1.
    """
    factor = np.asarray(factor_values, dtype=float).reshape(-1)
    carry = np.asarray(carry_m, dtype=float).reshape(-1)
    if factor.shape != carry.shape:
        raise ValueError(
            f"factor and carry must be paired; got {factor.shape} and {carry.shape}"
        )
    if factor.size < 3:
        raise ValueError(
            "a forgiveness sensitivity needs at least 3 samples; two points are "
            f"always perfectly correlated, got {factor.size}"
        )
    if not (np.all(np.isfinite(factor)) and np.all(np.isfinite(carry))):
        raise ValueError("factor and carry samples must be finite; found NaN or inf")
    return factor, carry


def forgiveness_sensitivity(
    factor_values: np.ndarray,
    carry_m: np.ndarray,
    *,
    factor: str,
    target_carry_m: float,
    unit: str | None = None,
    baseline_r: float | None = None,
    coverage_rtol: float = 1e-9,
) -> FactorSensitivity:
    """Measure how strongly carry tracks one factor over its swept range.

    Args:
        factor_values: ``(n,)`` factor samples, SI.
        carry_m: ``(n,)`` carry distances [m], paired with ``factor_values``.
        factor: Factor name. When it is a key of :data:`SWEEP_RANGES`, the unit,
            the published baseline and the range-coverage check are filled in
            from the registry.
        target_carry_m: Target carry the fractional change is taken against [m].
        unit: SI unit, overriding the registry.
        baseline_r: Published correlation to beat, overriding the registry.
        coverage_rtol: Relative slack when checking that the samples span the
            registered range.

    Returns:
        The sensitivity.

    Raises:
        ValueError: If the samples are unusable, or the target carry is not
            positive.
    """
    values, carry = _paired_samples(factor_values, carry_m)
    if not np.isfinite(target_carry_m) or target_carry_m <= 0.0:
        raise ValueError(
            f"target_carry_m must be positive and finite, got {target_carry_m}"
        )
    declared = SWEEP_RANGES.get(factor)
    slope, intercept, correlation = _linear_fit(values, carry)
    low, high = float(values.min()), float(values.max())
    change_m = slope * (high - low)
    covers: bool | None = None
    if declared is not None:
        slack = coverage_rtol * declared.span
        covers = bool(low <= declared.low + slack and high >= declared.high - slack)
    return FactorSensitivity(
        factor=factor,
        unit=unit if unit is not None else (declared.unit if declared else ""),
        n_samples=int(values.size),
        correlation_r=float(correlation),
        slope_m_per_unit=float(slope),
        intercept_m=float(intercept),
        sampled_low=low,
        sampled_high=high,
        carry_change_over_span_m=float(change_m),
        fractional_carry_change=float(change_m / target_carry_m),
        target_carry_m=float(target_carry_m),
        baseline_r=(
            baseline_r
            if baseline_r is not None
            else WIVOU_2016_CARRY_CORRELATION.get(factor)
        ),
        covers_declared_range=covers,
    )


@dataclass(frozen=True)
class ForgivenessReport:
    """Sensitivities for several factors, ranked.

    Attributes:
        sensitivities: One entry per factor, in the order supplied.
    """

    sensitivities: tuple[FactorSensitivity, ...]

    def ranked(self) -> tuple[FactorSensitivity, ...]:
        """Return the sensitivities worst-first by fractional carry change.

        Returns:
            The entries sorted by ``|fractional_carry_change|``, descending. Ties
            keep their input order, so the ranking is deterministic.
        """
        return tuple(
            sorted(
                self.sensitivities,
                key=lambda entry: abs(entry.fractional_carry_change),
                reverse=True,
            )
        )

    def worse_than_baseline(self) -> tuple[FactorSensitivity, ...]:
        """Return the factors this design tracks at least as strongly as published.

        Returns:
            Entries with a baseline whose ``|r|`` is not below it.
        """
        return tuple(
            entry
            for entry in self.sensitivities
            if entry.more_forgiving_than_baseline is False
        )


def forgiveness_report(
    samples: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    target_carry_m: float,
) -> ForgivenessReport:
    """Measure several factors at once.

    Args:
        samples: Mapping of factor name to ``(factor_values, carry_m)``.
        target_carry_m: Target carry [m].

    Returns:
        The report, entries in the order the mapping supplies them.

    Raises:
        ValueError: If no factors were supplied, or any factor's samples are
            unusable.
    """
    if not samples:
        raise ValueError("a forgiveness report needs at least one factor")
    return ForgivenessReport(
        sensitivities=tuple(
            forgiveness_sensitivity(
                values,
                carry,
                factor=name,
                target_carry_m=target_carry_m,
            )
            for name, (values, carry) in samples.items()
        )
    )
