"""The interval value object every propagated quantity travels in (#9243).

These tests pin the two things that make a band useful rather than
decorative: the arithmetic is *interval* arithmetic and not three parallel
point calculations, and the name of the type never claims a coverage
probability it cannot support.
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.vandv.band import (
    CONSISTENCY_BAND_NAMING_REASON,
    ConsistencyBand,
)

pytestmark = pytest.mark.unit


class TestConstruction:
    """A band that is not a band is refused at the boundary."""

    def test_edges_must_be_ordered(self) -> None:
        """An inverted band is refused rather than silently sorted."""
        with pytest.raises(ValueError, match="ordered"):
            ConsistencyBand(lower=2.0, central=1.0, upper=3.0)

    def test_central_must_lie_inside(self) -> None:
        """A central value outside its own edges is a different claim."""
        with pytest.raises(ValueError, match="ordered"):
            ConsistencyBand(lower=1.0, central=5.0, upper=3.0)

    def test_non_finite_is_refused(self) -> None:
        """NaN edges would propagate silently through every later map."""
        with pytest.raises(ValueError, match="finite"):
            ConsistencyBand(lower=math.nan, central=1.0, upper=3.0)

    def test_from_point_is_a_zero_width_band(self) -> None:
        """A point estimate is representable, and says it is a point."""
        band = ConsistencyBand.from_point(2.5)
        assert band.is_point
        assert band.width == pytest.approx(0.0)

    def test_from_edges_sorts_and_takes_the_geometric_mean(self) -> None:
        """Multiplicative edges get a multiplicative centre by default."""
        band = ConsistencyBand.from_edges(4.0, 1.0)
        assert band.lower == pytest.approx(1.0)
        assert band.upper == pytest.approx(4.0)
        assert band.central == pytest.approx(2.0)

    def test_from_edges_refuses_a_geometric_centre_across_zero(self) -> None:
        """A geometric mean of edges spanning zero is not defined."""
        with pytest.raises(ValueError, match="geometric"):
            ConsistencyBand.from_edges(-1.0, 4.0)

    def test_from_edges_accepts_an_explicit_centre(self) -> None:
        """An additive quantity may state an arithmetic centre instead."""
        band = ConsistencyBand.from_edges(-1.0, 4.0, central=1.5)
        assert band.central == pytest.approx(1.5)


class TestNaming:
    """The word 'confidence' is the one this type must not borrow."""

    def test_the_reason_disclaims_a_confidence_interval(self) -> None:
        """A consistency band is not a statistical statement."""
        reason = CONSISTENCY_BAND_NAMING_REASON.lower()
        assert "not a confidence interval" in reason
        assert "consistency" in reason

    def test_no_public_member_is_named_confidence(self) -> None:
        """Nothing on the type invites a coverage reading."""
        assert not [
            name
            for name in dir(ConsistencyBand)
            if not name.startswith("_") and "confidence" in name.lower()
        ]


class TestOverlap:
    """Overlap is the whole ranking rule, so it is pinned exactly."""

    def test_touching_bands_overlap(self) -> None:
        """A shared endpoint does not separate two designs."""
        left = ConsistencyBand(0.0, 1.0, 2.0)
        right = ConsistencyBand(2.0, 3.0, 4.0)
        assert left.overlaps(right)
        assert left.gap_to(right) == pytest.approx(0.0)

    def test_disjoint_bands_report_a_positive_gap(self) -> None:
        """The gap is the separation the ranking is allowed to use."""
        left = ConsistencyBand(0.0, 1.0, 2.0)
        right = ConsistencyBand(3.0, 4.0, 5.0)
        assert not left.overlaps(right)
        assert left.gap_to(right) == pytest.approx(1.0)

    def test_gap_is_symmetric(self) -> None:
        """Order of the pair cannot change whether they separate."""
        left = ConsistencyBand(0.0, 1.0, 2.0)
        right = ConsistencyBand(3.0, 4.0, 5.0)
        assert left.gap_to(right) == pytest.approx(right.gap_to(left))

    def test_overlap_depth_is_negative(self) -> None:
        """A negative gap says how far the bands interpenetrate."""
        left = ConsistencyBand(0.0, 1.0, 3.0)
        right = ConsistencyBand(2.0, 4.0, 5.0)
        assert left.gap_to(right) == pytest.approx(-1.0)


class TestMonotoneMap:
    """Propagation through a monotone model moves the edges, not the point."""

    def test_increasing_map_keeps_the_order(self) -> None:
        """A rising map carries lower to lower."""
        band = ConsistencyBand(1.0, 2.0, 4.0).map_monotone(lambda x: x**2)
        assert (band.lower, band.central, band.upper) == pytest.approx((1.0, 4.0, 16.0))

    def test_decreasing_map_flips_the_edges(self) -> None:
        """Carry falls as the accelerated mass rises: the edges must swap."""
        band = ConsistencyBand(1.0, 2.0, 4.0).map_monotone(lambda x: 1.0 / x)
        assert band.lower == pytest.approx(0.25)
        assert band.central == pytest.approx(0.5)
        assert band.upper == pytest.approx(1.0)

    def test_non_monotone_map_is_refused(self) -> None:
        """A map whose centre escapes its own edges was not monotone."""
        with pytest.raises(ValueError, match="monotone"):
            ConsistencyBand(-1.0, 0.0, 1.0).map_monotone(lambda x: x**2)


class TestAbsoluteDeviation:
    """The workbench objective is ``|carry - target|``, which is not monotone."""

    def test_band_entirely_below_the_target(self) -> None:
        """Distance to target shrinks as carry rises, so the edges swap."""
        band = ConsistencyBand(0.76, 1.59, 3.17).absolute_deviation_from(12.0)
        assert band.lower == pytest.approx(12.0 - 3.17)
        assert band.central == pytest.approx(12.0 - 1.59)
        assert band.upper == pytest.approx(12.0 - 0.76)

    def test_band_straddling_the_target_reaches_zero(self) -> None:
        """A band containing the target can hit the target exactly."""
        band = ConsistencyBand(10.0, 11.0, 14.0).absolute_deviation_from(12.0)
        assert band.lower == pytest.approx(0.0)
        assert band.central == pytest.approx(1.0)
        assert band.upper == pytest.approx(2.0)

    def test_a_point_band_stays_a_point(self) -> None:
        """No width is invented where there was none."""
        band = ConsistencyBand.from_point(9.0).absolute_deviation_from(12.0)
        assert band.is_point
        assert band.central == pytest.approx(3.0)


class TestAggregation:
    """Averaging bands over a sweep must not pretend the edges are noise."""

    def test_mean_is_edge_wise(self) -> None:
        """One global model-form choice moves every grid point together."""
        bands = (ConsistencyBand(0.0, 1.0, 2.0), ConsistencyBand(2.0, 3.0, 6.0))
        mean = ConsistencyBand.mean(bands)
        assert (mean.lower, mean.central, mean.upper) == pytest.approx((1.0, 2.0, 4.0))

    def test_mean_of_nothing_is_refused(self) -> None:
        """An empty sweep has no band, and inventing one would be a claim."""
        with pytest.raises(ValueError, match="at least one"):
            ConsistencyBand.mean(())

    def test_mean_width_does_not_shrink_with_more_points(self) -> None:
        """Edge-wise averaging is not a standard error: it never averages out."""
        one = ConsistencyBand.mean((ConsistencyBand(0.0, 1.0, 2.0),))
        many = ConsistencyBand.mean((ConsistencyBand(0.0, 1.0, 2.0),) * 20)
        assert many.width == pytest.approx(one.width)


class TestReporting:
    """A band that reaches a display carries its own disclaimer."""

    def test_statement_shows_all_three_edges(self) -> None:
        """The central value never appears without its edges."""
        text = ConsistencyBand(0.76, 1.59, 3.17).statement(unit="m")
        assert "1.59" in text
        assert "0.76" in text
        assert "3.17" in text
        assert "m" in text

    def test_relative_half_width_of_a_zero_centre_is_infinite(self) -> None:
        """A band about zero has no meaningful relative width."""
        band = ConsistencyBand(-1.0, 0.0, 1.0)
        assert math.isinf(band.relative_half_width)
