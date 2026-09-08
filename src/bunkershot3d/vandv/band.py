"""The interval a propagated quantity travels in (issue #9243).

Why this type exists
--------------------

Issue #8659 replaced the divot mass a ball launch divides by with an
**interval** -- :class:`~bunkershot3d.metrics.divot.AcceleratedSandMass`, whose
edges at the workbench's nominal greenside shot are 176 g and 413 g about a
central 270 g, a factor of 2.4. Everything downstream of that division inherits
the whole width: carry, the playability window, the energy split, and the
objective a design comparison is ranked on. Before this module the width was
computed at the metric and then thrown away, and a design was ranked on the
central value alone -- which is a claim of precision the model does not have.

A band, and deliberately not a distribution
-------------------------------------------

Nothing here is a probability. The edges of the accelerated-mass interval are
two *models* -- the smallest in-plane entrainment factor an F1 march produced
with no lateral spread at all, and the largest with the divot's walls laid back
at the bed's friction angle -- and the central value between them is a stated
convention, the geometric mean of two multiplicative factors. There is no
sampling distribution anywhere in that construction, no measurement to be
distributed about, and no coverage statement that could be attached to it: F1
is ``BEYOND_VALIDATION`` at a 1.44 m/s ceiling against a 25 m/s delivery, and
NASA-STD-7009B validation for this package is 0 of 4.

So this is a **consistency band**: the set of values the shipped models are
consistent with, and nothing more. It is not a confidence interval, it has no
coverage probability, and :data:`CONSISTENCY_BAND_NAMING_REASON` travels with
it so that no display can quietly promote it into one. Nothing on this class is
named ``confidence``, ``ci`` or ``sigma``, and a test enforces that.

What the arithmetic must get right
----------------------------------

Interval arithmetic, not three parallel point calculations:

* a **decreasing** map swaps the edges -- carry falls as the accelerated mass
  rises, so the lowest mass produces the *longest* carry;
* ``|x - target|``, the workbench's ranking objective, is **not monotone**, so
  a band straddling the target reaches zero and neither of its edges is the
  image of an edge (:meth:`ConsistencyBand.absolute_deviation_from`);
* averaging bands over a delivery sweep is **edge-wise** and never a standard
  error. One model-form choice moves every grid point in the same direction at
  once, so the width does not shrink as points are added. Treating it as noise
  that averages out is exactly the error this module exists to prevent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "CONSISTENCY_BAND_NAMING_REASON",
    "ConsistencyBand",
]

CONSISTENCY_BAND_NAMING_REASON = (
    "this is a CONSISTENCY band and NOT a confidence interval: its edges are "
    "two uncalibrated models rather than quantiles of a sampling "
    "distribution, so it carries no coverage probability and the value "
    "between them is a stated convention. Nothing in this package has been "
    "validated against a bunker (NASA-STD-7009B: 0 of 4), so a band this wide "
    "is a statement about what the models are consistent with, not about "
    "where a measurement would fall"
)
"""The sentence a band is never displayed without (issues #8616, #8659)."""


@dataclass(frozen=True, slots=True)
class ConsistencyBand:
    """A lower edge, a central convention and an upper edge, in one unit.

    Attributes:
        lower: Smallest value the models are consistent with.
        central: The conventional value between the edges. Reported, never
            quoted alone.
        upper: Largest value the models are consistent with.
    """

    lower: float
    central: float
    upper: float

    def __post_init__(self) -> None:
        """Refuse anything that is not an ordered, finite interval.

        A plain ``raise`` rather than a contract: ``python -O`` strips
        assertions and ``DBC_LEVEL=off`` disables contracts, and a band whose
        ordering evaporates under an optimisation flag would rank designs by
        accident.

        Raises:
            ValueError: If any edge is not finite, or the three values are not
                in non-decreasing order.
        """
        for name in ("lower", "central", "upper"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be a finite number, got {getattr(self, name)!r}"
                )
        if not self.lower <= self.central <= self.upper:
            raise ValueError(
                "a band must be ordered lower <= central <= upper, got "
                f"({self.lower!r}, {self.central!r}, {self.upper!r}); a central "
                "value outside its own edges is not a narrower claim, it is a "
                "different one"
            )

    @classmethod
    def from_point(cls, value: float) -> ConsistencyBand:
        """A zero-width band, for a quantity with no interval behind it.

        Args:
            value: The point estimate.

        Returns:
            The degenerate band, which reports :attr:`is_point`.
        """
        number = float(value)
        return cls(lower=number, central=number, upper=number)

    @classmethod
    def from_edges(
        cls, first: float, second: float, *, central: float | None = None
    ) -> ConsistencyBand:
        """Build a band from two edges in either order.

        Args:
            first: One edge.
            second: The other edge.
            central: The conventional value between them. When omitted the
                **geometric** mean is used, because the edges this package
                produces are multiplicative factors on a common base rather
                than additive offsets -- the same convention
                :class:`~bunkershot3d.metrics.divot.AcceleratedSandMass` uses.

        Returns:
            The band.

        Raises:
            ValueError: If the geometric mean is requested for edges that do
                not share a sign, where it is undefined.
        """
        low, high = sorted((float(first), float(second)))
        if central is not None:
            return cls(lower=low, central=float(central), upper=high)
        if low <= 0.0 <= high and not (low == 0.0 and high == 0.0):
            raise ValueError(
                "a geometric centre is undefined for edges spanning zero "
                f"([{low!r}, {high!r}]); pass an explicit central value"
            )
        sign = -1.0 if high < 0.0 else 1.0
        return cls(lower=low, central=sign * math.sqrt(low * high), upper=high)

    @property
    def width(self) -> float:
        """``upper - lower``, in the band's own unit."""
        return self.upper - self.lower

    @property
    def half_width(self) -> float:
        """Half the width. Never a standard uncertainty; see the module docs."""
        return 0.5 * self.width

    @property
    def is_point(self) -> bool:
        """True when the band has no width at all."""
        return self.upper == self.lower

    @property
    def relative_half_width(self) -> float:
        """Half width over ``|central|``; infinite for a band about zero."""
        if self.central == 0.0:
            return math.inf
        return self.half_width / abs(self.central)

    def contains(self, value: float) -> bool:
        """Whether a value lies within the band, edges included.

        Args:
            value: The value to test.

        Returns:
            True when ``lower <= value <= upper``.
        """
        return self.lower <= float(value) <= self.upper

    def overlaps(self, other: ConsistencyBand) -> bool:
        """Whether two bands share any value at all.

        Touching bands overlap: a shared endpoint is not a separation, and
        ranking two designs whose bands meet would be ordering them on an
        equality.

        Args:
            other: The other band.

        Returns:
            True when the two intervals intersect.
        """
        return self.lower <= other.upper and other.lower <= self.upper

    def gap_to(self, other: ConsistencyBand) -> float:
        """Signed separation between two bands, in the band's unit.

        Args:
            other: The other band.

        Returns:
            Positive and equal to the empty space between them when they are
            disjoint; zero when they touch; negative and equal to the depth of
            the interpenetration when they overlap.
        """
        return max(self.lower, other.lower) - min(self.upper, other.upper)

    def scaled(self, factor: float) -> ConsistencyBand:
        """Multiply every edge by a constant.

        Args:
            factor: The multiplier. A negative factor flips the band.

        Returns:
            The scaled band.
        """
        return self.map_monotone(lambda value: value * float(factor))

    def shifted(self, offset: float) -> ConsistencyBand:
        """Add a constant to every edge.

        Args:
            offset: The shift.

        Returns:
            The shifted band, of the same width.
        """
        return self.map_monotone(lambda value: value + float(offset))

    def map_monotone(self, transform: Callable[[float], float]) -> ConsistencyBand:
        """Push the band through a monotone model.

        The images of the two edges are re-sorted, because a **decreasing**
        map swaps them -- which is the case that matters here: carry falls as
        the accelerated sand mass rises, so the interval's lower mass edge
        produces the *upper* carry edge.

        Monotonicity cannot be checked in general, so the one consequence of
        it that is checkable is checked: the image of the centre must still lie
        between the images of the edges. That catches
        :func:`abs`-shaped maps, which have their own method
        (:meth:`absolute_deviation_from`) precisely because they are not
        monotone.

        Args:
            transform: The model, applied to each edge and to the centre.

        Returns:
            The transformed band.

        Raises:
            ValueError: If any image is not finite, or if the centre's image
                falls outside the edges' images.
        """
        images = [float(transform(value)) for value in (self.lower, self.upper)]
        centre = float(transform(self.central))
        if not all(math.isfinite(value) for value in (*images, centre)):
            raise ValueError(
                "the transform produced a non-finite edge; a band cannot "
                f"carry it (images {images!r}, centre {centre!r})"
            )
        low, high = sorted(images)
        if not low <= centre <= high:
            raise ValueError(
                f"the transform is not monotone over [{self.lower!r}, "
                f"{self.upper!r}]: the centre maps to {centre!r}, outside the "
                f"edges' images [{low!r}, {high!r}]. Use a method that knows "
                "the shape of the map, such as absolute_deviation_from"
            )
        return ConsistencyBand(lower=low, central=centre, upper=high)

    def absolute_deviation_from(self, target: float) -> ConsistencyBand:
        """The exact image of this band under ``|x - target|``.

        The workbench ranks designs on distance to a target carry, which is a
        V-shaped map: a band that straddles the target contains the target, so
        its image reaches **zero**, and neither image edge is the image of a
        band edge. Mapping the three values one at a time and sorting would
        report a floor of ``min(|lower - t|, |upper - t|)`` instead, which is
        a narrower objective band than the model supports.

        Args:
            target: The value distance is measured to.

        Returns:
            The image band, whose centre is ``|central - target|``.
        """
        aim = float(target)
        edges = (abs(self.lower - aim), abs(self.upper - aim))
        floor = 0.0 if self.contains(aim) else min(edges)
        return ConsistencyBand(
            lower=floor, central=abs(self.central - aim), upper=max(edges)
        )

    @classmethod
    def mean(cls, bands: Sequence[ConsistencyBand]) -> ConsistencyBand:
        """Average a set of bands **edge-wise**.

        Deliberately not a standard error. The width of every band in a
        workbench sweep comes from the *same* model-form choice -- one
        accelerated-mass interval, applied at every delivery condition -- so
        the edges move together and the width does not shrink as conditions
        are added. Combining them in quadrature would claim an independence
        that a common modelling assumption does not have, and would report a
        sweep of twenty conditions as four and a half times more certain than
        a sweep of one.

        Args:
            bands: The bands to average. Must be non-empty.

        Returns:
            The edge-wise mean.

        Raises:
            ValueError: If no bands are given.
        """
        collected = tuple(bands)
        if not collected:
            raise ValueError(
                "averaging needs at least one band; an empty sweep has no "
                "band and inventing one would be a claim"
            )
        count = float(len(collected))
        return cls(
            lower=math.fsum(band.lower for band in collected) / count,
            central=math.fsum(band.central for band in collected) / count,
            upper=math.fsum(band.upper for band in collected) / count,
        )

    def statement(self, unit: str = "", precision: int = 4) -> str:
        """A line fit for a report: the centre never without its edges.

        Args:
            unit: Unit suffix, appended to each number.
            precision: Significant figures.

        Returns:
            The formatted band.
        """
        suffix = f" {unit}" if unit else ""
        return (
            f"{self.central:.{precision}g}{suffix} "
            f"[{self.lower:.{precision}g}, {self.upper:.{precision}g}{suffix}]"
        )
