"""Particle size distribution for bunker sand (issue #8610).

A :class:`ParticleSizeDistribution` is a sieve analysis: a set of sieve
openings and the cumulative mass fraction passing each of them. Everything a
consumer needs -- percentile diameters, the uniformity coefficient, band
fractions for USGA compliance, and the number-weighted diameter used by the
grain-count feasibility guard -- is derived from that one representation.

All lengths are metres. Sieve openings are stored **ascending** so the
cumulative-passing curve is monotone non-decreasing in index order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .exceptions import ParticleSizeDistributionError

__all__ = ["ParticleSizeDistribution"]

_SUM_TOLERANCE = 1e-9
_MONOTONE_TOLERANCE = 1e-12


def _check_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ParticleSizeDistributionError(f"{name} must be finite, got {value!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class ParticleSizeDistribution:
    """A sieve analysis expressed as a cumulative-passing curve.

    Attributes:
        sieve_openings_m: Strictly ascending sieve openings, metres. The first
            entry is the lower size cutoff of the analysis; material finer
            than it is reported by ``fraction_passing[0]``.
        fraction_passing: Cumulative mass fraction passing each opening,
            monotone non-decreasing, ending at exactly 1.0.
        name: Optional label used in reports.
    """

    sieve_openings_m: tuple[float, ...]
    fraction_passing: tuple[float, ...]
    name: str = ""

    # ---------------------------------------------------------------- setup

    def __post_init__(self) -> None:
        openings = tuple(
            _check_finite(d, "sieve opening") for d in self.sieve_openings_m
        )
        passing = tuple(
            _check_finite(p, "fraction passing") for p in self.fraction_passing
        )
        if len(openings) < 2:
            raise ParticleSizeDistributionError(
                "a sieve analysis needs at least two openings (a lower and an "
                f"upper size cutoff), got {len(openings)}"
            )
        if len(openings) != len(passing):
            raise ParticleSizeDistributionError(
                "sieve_openings_m and fraction_passing must have equal length, "
                f"got {len(openings)} and {len(passing)}"
            )
        if any(d <= 0.0 for d in openings):
            raise ParticleSizeDistributionError(
                f"every sieve opening must be positive, got {openings}"
            )
        if any(b <= a for a, b in zip(openings, openings[1:], strict=False)):
            raise ParticleSizeDistributionError(
                f"sieve openings must be strictly ascending, got {openings}"
            )
        if any(
            p < -_MONOTONE_TOLERANCE or p > 1.0 + _MONOTONE_TOLERANCE for p in passing
        ):
            raise ParticleSizeDistributionError(
                f"every cumulative fraction passing must lie in [0, 1], got {passing}"
            )
        if any(
            b < a - _MONOTONE_TOLERANCE
            for a, b in zip(passing, passing[1:], strict=False)
        ):
            raise ParticleSizeDistributionError(
                "the cumulative-passing curve must be monotone non-decreasing, "
                f"got {passing}"
            )
        if abs(passing[-1] - 1.0) > _SUM_TOLERANCE:
            raise ParticleSizeDistributionError(
                "the coarsest opening must pass the whole sample, so "
                f"fraction_passing[-1] must be 1.0, got {passing[-1]!r}"
            )
        object.__setattr__(self, "sieve_openings_m", openings)
        object.__setattr__(self, "fraction_passing", passing)

    @classmethod
    def from_bins(
        cls,
        bin_edges_m: tuple[float, ...],
        bin_fractions: tuple[float, ...],
        name: str = "",
    ) -> ParticleSizeDistribution:
        """Build from size bins, the form published sieve tables use.

        Args:
            bin_edges_m: ``n + 1`` strictly ascending bin boundaries, metres.
            bin_fractions: ``n`` mass fractions, one per bin, summing to 1.
            name: Optional label.

        Returns:
            The equivalent cumulative-passing distribution.

        Raises:
            ParticleSizeDistributionError: on any malformed input.
        """
        edges = tuple(float(d) for d in bin_edges_m)
        fractions = tuple(float(f) for f in bin_fractions)
        if len(edges) < 2:
            raise ParticleSizeDistributionError(
                f"need at least two bin edges, got {len(edges)}"
            )
        if len(fractions) != len(edges) - 1:
            raise ParticleSizeDistributionError(
                f"expected {len(edges) - 1} bin fractions for {len(edges)} edges, "
                f"got {len(fractions)}"
            )
        if any(f < 0.0 for f in fractions):
            raise ParticleSizeDistributionError(
                f"bin fractions must not be negative, got {fractions}"
            )
        total = math.fsum(fractions)
        if abs(total - 1.0) > _SUM_TOLERANCE:
            raise ParticleSizeDistributionError(
                f"bin fractions must sum to 1.0, got {total!r}"
            )
        passing = [0.0]
        for fraction in fractions:
            passing.append(min(1.0, passing[-1] + fraction))
        passing[-1] = 1.0
        return cls(sieve_openings_m=edges, fraction_passing=tuple(passing), name=name)

    @classmethod
    def from_retained_fractions(
        cls,
        sieve_openings_m: tuple[float, ...],
        mass_fraction_retained: tuple[float, ...],
        largest_particle_m: float,
        name: str = "",
    ) -> ParticleSizeDistribution:
        """Build from a physical sieve stack.

        Args:
            sieve_openings_m: Sieve openings in stack order, coarsest first.
            mass_fraction_retained: One fraction per sieve plus a final pan
                fraction, so ``len(...) == len(sieve_openings_m) + 1``.
            largest_particle_m: Upper size cutoff, coarser than the top sieve.
            name: Optional label.

        Returns:
            The equivalent cumulative-passing distribution.

        Raises:
            ParticleSizeDistributionError: on any malformed input.
        """
        openings = tuple(float(d) for d in sieve_openings_m)
        retained = tuple(float(f) for f in mass_fraction_retained)
        if len(retained) != len(openings) + 1:
            raise ParticleSizeDistributionError(
                "mass_fraction_retained needs one entry per sieve plus a pan "
                f"entry: expected {len(openings) + 1}, got {len(retained)}"
            )
        if openings and largest_particle_m <= openings[0]:
            raise ParticleSizeDistributionError(
                f"largest_particle_m ({largest_particle_m}) must exceed the "
                f"coarsest sieve opening ({openings[0]})"
            )
        pan_fraction = retained[-1]
        if pan_fraction > _SUM_TOLERANCE:
            raise ParticleSizeDistributionError(
                f"{pan_fraction:.4f} of the sample is finer than the finest "
                "sieve; the analysis has no lower size cutoff, so percentile "
                "diameters and grain counts cannot be resolved. Add a finer "
                "sieve."
            )
        edges = (*reversed(openings), largest_particle_m)
        fractions = tuple(reversed(retained[:-1]))
        return cls.from_bins(bin_edges_m=edges, bin_fractions=fractions, name=name)

    # ----------------------------------------------------------- structure

    @property
    def bin_edges_m(self) -> tuple[float, ...]:
        """Alias of :attr:`sieve_openings_m`, read as bin boundaries."""
        return self.sieve_openings_m

    @property
    def bin_fractions(self) -> tuple[float, ...]:
        """Mass fraction per size class, summing to 1.

        The first entry is the fraction finer than the smallest sieve opening;
        entry ``i`` for ``i >= 1`` is the fraction between openings
        ``i - 1`` and ``i``.
        """
        passing = self.fraction_passing
        return (
            passing[0],
            *(b - a for a, b in zip(passing, passing[1:], strict=False)),
        )

    # ---------------------------------------------------------- percentiles

    def diameter_at_passing(self, fraction: float) -> float:
        """Return the diameter (m) at a given cumulative fraction passing.

        Interpolation is log-linear in diameter, the standard convention for
        a sieve curve plotted on a logarithmic size axis.

        Raises:
            ParticleSizeDistributionError: if ``fraction`` is outside the open
                interval (0, 1), or falls below the finest sieve.
        """
        target = _check_finite(fraction, "cumulative fraction")
        if not 0.0 < target < 1.0:
            raise ParticleSizeDistributionError(
                f"cumulative fraction must be between 0 and 1 exclusive, got {target!r}"
            )
        passing = self.fraction_passing
        openings = self.sieve_openings_m
        if target <= passing[0]:
            raise ParticleSizeDistributionError(
                f"d{target * 100:g} lies below the finest sieve "
                f"({openings[0] * 1e3:.4g} mm), which already passes "
                f"{passing[0] * 100:.3g}% of the sample. Add a finer sieve to "
                "resolve it."
            )
        index = next(i for i, p in enumerate(passing) if p >= target)
        if math.isclose(passing[index], target, rel_tol=0.0, abs_tol=1e-15):
            return openings[index]
        lower, upper = index - 1, index
        span = passing[upper] - passing[lower]
        weight = (target - passing[lower]) / span
        log_lower = math.log(openings[lower])
        log_upper = math.log(openings[upper])
        return math.exp(log_lower + weight * (log_upper - log_lower))

    @property
    def d10_m(self) -> float:
        """Effective size: the diameter at 10% passing."""
        return self.diameter_at_passing(0.10)

    @property
    def d30_m(self) -> float:
        """Diameter at 30% passing, used by the coefficient of curvature."""
        return self.diameter_at_passing(0.30)

    @property
    def d50_m(self) -> float:
        """Median diameter."""
        return self.diameter_at_passing(0.50)

    @property
    def d60_m(self) -> float:
        """Diameter at 60% passing."""
        return self.diameter_at_passing(0.60)

    @property
    def uniformity_coefficient(self) -> float:
        """Cu = d60 / d10. USGA bunker sand targets 2.0-5.0."""
        return self.d60_m / self.d10_m

    @property
    def coefficient_of_curvature(self) -> float:
        """Cc = d30^2 / (d10 * d60)."""
        return self.d30_m**2 / (self.d10_m * self.d60_m)

    # ------------------------------------------------------ band fractions

    def fraction_finer_than(self, diameter_m: float) -> float:
        """Return the mass fraction finer than ``diameter_m``."""
        target = _check_finite(diameter_m, "diameter")
        openings = self.sieve_openings_m
        passing = self.fraction_passing
        if target <= openings[0]:
            return 0.0 if target < openings[0] else passing[0]
        if target >= openings[-1]:
            return 1.0
        index = next(i for i, d in enumerate(openings) if d >= target)
        lower, upper = index - 1, index
        log_span = math.log(openings[upper]) - math.log(openings[lower])
        weight = (math.log(target) - math.log(openings[lower])) / log_span
        return passing[lower] + weight * (passing[upper] - passing[lower])

    def fraction_between(self, lower_m: float, upper_m: float) -> float:
        """Return the mass fraction between two diameters.

        Raises:
            ParticleSizeDistributionError: if ``lower_m >= upper_m``.
        """
        if lower_m >= upper_m:
            raise ParticleSizeDistributionError(
                f"lower bound {lower_m} must be smaller than upper bound {upper_m}"
            )
        return self.fraction_finer_than(upper_m) - self.fraction_finer_than(lower_m)

    # -------------------------------------------------- number-weighted size

    @property
    def volume_equivalent_diameter_m(self) -> float:
        """Diameter of the monodisperse population with the same grain count.

        For a solid volume ``V_s`` the true grain count is
        ``(6 V_s / pi) * sum_i f_i / d_i^3`` with ``f_i`` the mass fraction of
        bin ``i`` and ``d_i`` its geometric-mean diameter. This property
        returns ``(sum_i f_i / d_i^3) ** (-1/3)``, the single diameter that
        reproduces that count.

        It is **much smaller than d50** for a real bunker sand, because a few
        percent of silt contributes the overwhelming majority of the grains.
        That is the honest number: use it when reporting how many particles a
        resolved DEM run would truly need.
        """
        openings = self.sieve_openings_m
        fractions = self.bin_fractions
        total = 0.0
        for index, fraction in enumerate(fractions[1:], start=1):
            if fraction <= 0.0:
                continue
            representative = math.sqrt(openings[index - 1] * openings[index])
            total += fraction / representative**3
        if total <= 0.0:
            raise ParticleSizeDistributionError(
                "the distribution carries no mass above its lower size cutoff"
            )
        return total ** (-1.0 / 3.0)
