"""Spanwise (heel-to-toe) sole load, against hand arithmetic (issue #8699).

The fixtures here are synthetic soles whose spanwise summaries can be worked
out on paper, because the point is to check the *arithmetic and the sign*, not
to check the code against itself.

**The reference sole.** Twelve spanwise stations evenly spaced across a 60 mm
span, each station carrying three chordwise elements. Stations therefore sit at
``-30 mm + i * (60/11) mm``, which puts exactly four stations in each third of
the span, and exactly two in each of six equal bins. With a uniform load:

* heel third fraction  ``4/12 = 1/3``
* toe third fraction   ``4/12 = 1/3``
* outer third fraction ``8/12 = 2/3``
* heel/toe balance     ``0`` exactly -- six stations either side of mid-span

**The sign that matters.** Relief sheds load from the end it is ground into, so
shedding load from the toe half must move the balance toward the heel. The
balance is signed along the body ``+y`` axis, which
:mod:`bunkershot3d.geometry.lofting` documents as running heel to toe, so
"toward the heel" means *more negative*. Getting that backwards is the defect
issue #9247 describes elsewhere in this model, so it is pinned here rather than
asserted in a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.exceptions import BunkerShot3DValueError
from bunkershot3d.metrics import SoleLoadTrace
from bunkershot3d.metrics.spanwise import (
    MIN_ELEMENTS_PER_SPANWISE_BIN,
    SPANWISE_AXIS_INDEX,
    SpanwiseLoad,
    spanwise_load,
)
from bunkershot3d.solvers import (
    EnvelopeStatus,
    FidelityTier,
    OutOfEnvelopeError,
)

from ._spanwise_fixtures import (
    N_SPAN_STATIONS,
    SPAN_HALF_M,
    build_span_load,
    refused_verdict,
    within_verdict,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def uniform_load() -> SoleLoadTrace:
    """The reference sole carrying the same load at every station."""
    return build_span_load(station_force_N=np.full(N_SPAN_STATIONS, 100.0))


@pytest.fixture
def uniform_result(uniform_load: SoleLoadTrace) -> SpanwiseLoad:
    """The reference sole resolved into six spanwise bins."""
    return spanwise_load(uniform_load, n_bins=6, fidelity_tier=FidelityTier.F0)


def _relieved(*, heel: float, toe: float) -> SpanwiseLoad:
    """Resolve the reference sole with one end's load scaled down.

    Args:
        heel: Multiplier on every station in the heel half.
        toe: Multiplier on every station in the toe half.

    Returns:
        The spanwise result at six bins, F0.
    """
    stations = np.linspace(-SPAN_HALF_M, SPAN_HALF_M, N_SPAN_STATIONS)
    scale = np.where(stations < 0.0, heel, toe)
    return spanwise_load(
        build_span_load(station_force_N=100.0 * scale),
        n_bins=6,
        fidelity_tier=FidelityTier.F0,
    )


class TestSpanwiseAxis:
    """The axis the distribution is resolved along."""

    def test_the_spanwise_axis_is_body_y(self) -> None:
        """``geometry.lofting`` fixes ``+x`` rearward, ``+y`` heel to toe."""
        assert SPANWISE_AXIS_INDEX == 1


class TestSymmetry:
    """A geometrically symmetric sole must read as symmetric."""

    def test_a_symmetric_sole_has_zero_balance(self, uniform_result) -> None:
        """Six stations either side of mid-span, equally loaded, cancel."""
        assert uniform_result.heel_toe_balance == pytest.approx(0.0, abs=1e-12)

    def test_a_symmetric_sole_centres_its_load_at_mid_span(
        self, uniform_result
    ) -> None:
        """The impulse-weighted station is mid-span, so the normalised one is 0."""
        assert uniform_result.mid_span_body_m == pytest.approx(0.0, abs=1e-15)
        assert uniform_result.centroid_body_m == pytest.approx(0.0, abs=1e-12)
        assert uniform_result.centroid_normalised == pytest.approx(0.0, abs=1e-12)

    def test_a_symmetric_sole_bins_symmetrically(self, uniform_result) -> None:
        """Six equal bins over twelve equally loaded stations are all equal."""
        distribution = uniform_result.distribution
        np.testing.assert_allclose(
            distribution.impulse_fraction, np.full(6, 1.0 / 6.0), rtol=1e-12
        )
        np.testing.assert_array_equal(distribution.element_count, np.full(6, 6))

    def test_a_mirrored_load_mirrors_the_balance(self) -> None:
        """Swapping heel and toe relief flips the sign and keeps the magnitude."""
        toe_ground = _relieved(heel=1.0, toe=0.6)
        heel_ground = _relieved(heel=0.6, toe=1.0)

        assert toe_ground.heel_toe_balance == pytest.approx(
            -heel_ground.heel_toe_balance, rel=1e-12
        )


class TestBalanceSign:
    """The physical claim: which way relief moves the load."""

    def test_toe_relief_shifts_the_balance_toward_the_heel(self) -> None:
        """Shedding toe load makes the balance negative -- heel is body ``-y``."""
        result = _relieved(heel=1.0, toe=0.6)

        assert result.heel_toe_balance < 0.0
        assert result.centroid_body_m < 0.0
        assert result.centroid_normalised < 0.0

    def test_heel_relief_shifts_the_balance_toward_the_toe(self) -> None:
        """The mirror image, so the sign is a convention and not an accident."""
        result = _relieved(heel=0.6, toe=1.0)

        assert result.heel_toe_balance > 0.0
        assert result.centroid_body_m > 0.0

    def test_more_toe_relief_shifts_the_balance_further(self) -> None:
        """The metric is monotone in how much load the toe sheds."""
        light = _relieved(heel=1.0, toe=0.8)
        heavy = _relieved(heel=1.0, toe=0.4)

        assert heavy.heel_toe_balance < light.heel_toe_balance < 0.0

    def test_the_balance_is_the_signed_impulse_difference(self) -> None:
        """Toe half at 0.6 of heel half: ``(0.6 - 1) / 1.6 = -0.25``."""
        result = _relieved(heel=1.0, toe=0.6)

        assert result.heel_toe_balance == pytest.approx(-0.25, rel=1e-12)

    def test_the_balance_is_bounded_by_one(self) -> None:
        """All the load on the heel half is the extreme: exactly ``-1``.

        The centroid does not reach ``-1`` with it: it is the mean loaded
        station, and the mean of the six heel-half stations of a twelve-station
        span is ``-6/11`` of the half-span, not its end. The two quantities
        answer different questions, which is why both are reported.
        """
        result = _relieved(heel=1.0, toe=0.0)

        assert result.heel_toe_balance == pytest.approx(-1.0, rel=1e-12)
        assert result.centroid_normalised == pytest.approx(-6 / 11, rel=1e-12)


class TestOuterThirds:
    """How much of the strike the ends of the blade carry."""

    def test_a_uniform_sole_puts_a_third_in_each_third(self, uniform_result) -> None:
        """Four of twelve stations sit in each third of the span."""
        assert uniform_result.heel_third_fraction == pytest.approx(1 / 3, rel=1e-12)
        assert uniform_result.toe_third_fraction == pytest.approx(1 / 3, rel=1e-12)
        assert uniform_result.outer_third_fraction == pytest.approx(2 / 3, rel=1e-12)

    def test_the_outer_thirds_are_the_two_ends_summed(self) -> None:
        """The outer fraction never double counts the middle third."""
        result = _relieved(heel=1.0, toe=0.5)

        assert result.outer_third_fraction == pytest.approx(
            result.heel_third_fraction + result.toe_third_fraction, rel=1e-12
        )
        assert 0.0 <= result.outer_third_fraction <= 1.0

    def test_load_confined_to_the_middle_leaves_the_ends_empty(self) -> None:
        """Only the middle third loaded: both outer fractions are zero."""
        stations = np.linspace(-SPAN_HALF_M, SPAN_HALF_M, N_SPAN_STATIONS)
        middle = np.abs(stations) < SPAN_HALF_M / 3.0
        result = spanwise_load(
            build_span_load(station_force_N=np.where(middle, 100.0, 0.0)),
            n_bins=6,
            fidelity_tier=FidelityTier.F0,
        )

        assert result.outer_third_fraction == pytest.approx(0.0, abs=1e-12)


class TestConservation:
    """Binning re-partitions the strike; it does not create or destroy it."""

    def test_the_bins_sum_to_the_total_impulse_and_area(
        self, uniform_load, uniform_result
    ) -> None:
        """A histogram of the elements is still all of the elements."""
        distribution = uniform_result.distribution

        assert distribution.impulse_Ns.sum() == pytest.approx(
            uniform_result.total_impulse_Ns, rel=1e-12
        )
        assert distribution.area_m2.sum() == pytest.approx(
            uniform_load.total_area_m2, rel=1e-12
        )
        assert int(distribution.element_count.sum()) == uniform_load.n_elements

    def test_the_summaries_do_not_depend_on_the_bin_count(self, uniform_load) -> None:
        """Balance, centroid and thirds are element quantities, not bin ones."""
        coarse = spanwise_load(uniform_load, n_bins=2, fidelity_tier=FidelityTier.F0)
        fine = spanwise_load(uniform_load, n_bins=6, fidelity_tier=FidelityTier.F0)

        assert coarse.heel_toe_balance == pytest.approx(fine.heel_toe_balance)
        assert coarse.centroid_body_m == pytest.approx(fine.centroid_body_m)
        assert coarse.outer_third_fraction == pytest.approx(fine.outer_third_fraction)

    def test_the_bin_peak_force_is_the_bin_total_not_an_element_total(
        self, uniform_result
    ) -> None:
        """Six elements at 100/3 N each peak together at 200 N in their bin."""
        np.testing.assert_allclose(
            uniform_result.distribution.peak_force_N, np.full(6, 200.0), rtol=1e-12
        )


class TestMigration:
    """Where the load sits at each instant, not only over the whole strike."""

    def test_the_centroid_tracks_the_load_from_toe_to_heel(self) -> None:
        """A strike that starts toe-side and finishes heel-side reads that way."""
        stations = np.linspace(-SPAN_HALF_M, SPAN_HALF_M, N_SPAN_STATIONS)
        toe_first = np.where(stations > 0.0, 100.0, 0.0)
        heel_last = np.where(stations < 0.0, 100.0, 0.0)
        result = spanwise_load(
            build_span_load(station_force_N=np.vstack([toe_first, heel_last])),
            n_bins=6,
            fidelity_tier=FidelityTier.F0,
        )
        migration = result.migration

        assert migration.centroid_body_m[0] > 0.0
        assert migration.centroid_body_m[-1] < 0.0
        assert migration.net_travel_m() < 0.0
        assert migration.range_m() == pytest.approx(
            float(migration.centroid_body_m[0] - migration.centroid_body_m[-1]),
            rel=1e-12,
        )

    def test_a_stationary_load_does_not_migrate(self, uniform_result) -> None:
        """A load that never moves has zero range and zero net travel."""
        migration = uniform_result.migration

        assert migration.range_m() == pytest.approx(0.0, abs=1e-12)
        assert migration.net_travel_m() == pytest.approx(0.0, abs=1e-12)

    def test_unloaded_samples_are_nan_rather_than_mid_span(self) -> None:
        """A sample carrying nothing has no centroid; zero would read as centred."""
        stations = np.linspace(-SPAN_HALF_M, SPAN_HALF_M, N_SPAN_STATIONS)
        loaded = np.full(N_SPAN_STATIONS, 100.0)
        result = spanwise_load(
            build_span_load(
                station_force_N=np.vstack([np.zeros_like(stations), loaded, loaded])
            ),
            n_bins=6,
            fidelity_tier=FidelityTier.F0,
        )
        migration = result.migration

        assert np.isnan(migration.centroid_body_m[0])
        np.testing.assert_array_equal(migration.loaded_sample_mask, [False, True, True])
        assert migration.loaded_sample_count == 2

    def test_a_single_loaded_sample_refuses_to_report_migration(self) -> None:
        """One instant is a position, not a migration."""
        stations = np.linspace(-SPAN_HALF_M, SPAN_HALF_M, N_SPAN_STATIONS)
        loaded = np.full(N_SPAN_STATIONS, 100.0)
        result = spanwise_load(
            build_span_load(
                station_force_N=np.vstack(
                    [np.zeros_like(stations), loaded, np.zeros_like(stations)]
                )
            ),
            n_bins=6,
            fidelity_tier=FidelityTier.F0,
        )

        with pytest.raises(BunkerShot3DValueError, match="one loaded sample"):
            result.migration.range_m()
        with pytest.raises(BunkerShot3DValueError, match="one loaded sample"):
            result.migration.net_travel_m()


class TestResolutionRefusals:
    """Where the elements cannot support the picture, refuse to draw it."""

    def test_more_bins_than_stations_is_refused(self, uniform_load) -> None:
        """Twelve stations at two per bin support six bins, never seven."""
        with pytest.raises(BunkerShot3DValueError, match="resolves 12 spanwise"):
            spanwise_load(uniform_load, n_bins=7, fidelity_tier=FidelityTier.F0)

    def test_the_supportable_bin_count_is_stated_in_the_refusal(
        self, uniform_load
    ) -> None:
        """The caller is told what it *can* ask for, not merely that it failed."""
        with pytest.raises(BunkerShot3DValueError, match="at most 6"):
            spanwise_load(uniform_load, n_bins=12, fidelity_tier=FidelityTier.F0)

        assert MIN_ELEMENTS_PER_SPANWISE_BIN == 2

    def test_an_empty_bin_is_refused_rather_than_smoothed(self) -> None:
        """Clustered stations leave a gap; a zero bin there is an artifact."""
        stations = np.concatenate(
            [
                np.linspace(-SPAN_HALF_M, -SPAN_HALF_M + 0.002, 6),
                np.linspace(SPAN_HALF_M - 0.002, SPAN_HALF_M, 6),
            ]
        )
        clustered = build_span_load(
            station_force_N=np.full(N_SPAN_STATIONS, 100.0), station_m=stations
        )

        with pytest.raises(BunkerShot3DValueError, match="no element at all"):
            spanwise_load(clustered, n_bins=6, fidelity_tier=FidelityTier.F0)

    def test_fewer_than_two_bins_is_refused(self, uniform_load) -> None:
        """A one-bin histogram carries no heel-to-toe information at all."""
        with pytest.raises(BunkerShot3DValueError, match="at least 2"):
            spanwise_load(uniform_load, n_bins=1, fidelity_tier=FidelityTier.F0)

    def test_a_single_element_sole_is_refused(self) -> None:
        """One element spans nothing, so it distributes nothing."""
        single = SoleLoadTrace(
            time_s=np.array([0.0, 0.01]),
            element_centroid_body_m=np.zeros((1, 3)),
            element_area_m2=np.array([1.0e-4]),
            element_normal_force_N=np.full((2, 1), 100.0),
        )

        with pytest.raises(BunkerShot3DValueError, match="one spanwise station"):
            spanwise_load(single, n_bins=2, fidelity_tier=FidelityTier.F0)

    def test_a_sole_collapsed_onto_one_station_is_refused(self) -> None:
        """Many elements at one station still resolve no span."""
        count = 8
        collapsed = SoleLoadTrace(
            time_s=np.array([0.0, 0.01]),
            element_centroid_body_m=np.column_stack(
                [np.linspace(-0.01, 0.01, count), np.zeros(count), np.zeros(count)]
            ),
            element_area_m2=np.full(count, 1.0e-5),
            element_normal_force_N=np.full((2, count), 10.0),
        )

        with pytest.raises(BunkerShot3DValueError, match="one spanwise station"):
            spanwise_load(collapsed, n_bins=2, fidelity_tier=FidelityTier.F0)

    def test_an_unloaded_sole_is_refused(self) -> None:
        """Nothing was carried, so there is no distribution of it."""
        with pytest.raises(BunkerShot3DValueError, match="carried no load"):
            spanwise_load(
                build_span_load(station_force_N=np.zeros(N_SPAN_STATIONS)),
                n_bins=6,
                fidelity_tier=FidelityTier.F0,
            )


class TestTierAndValidity:
    """The statement every number here has to be read under."""

    def test_a_plane_strain_tier_is_refused_outright(self, uniform_load) -> None:
        """F1 is 2-D by construction (ADR-0033/0044); it has no span."""
        with pytest.raises(BunkerShot3DValueError, match="plane-strain"):
            spanwise_load(uniform_load, n_bins=6, fidelity_tier=FidelityTier.F1)

    def test_the_tier_travels_with_the_result(self, uniform_result) -> None:
        """A distribution without its tier reads as though it were measured."""
        assert uniform_result.credibility.fidelity_tier is FidelityTier.F0

    def test_the_verdict_travels_with_the_result(self, uniform_load) -> None:
        """The envelope status is carried, not left in report prose."""
        result = spanwise_load(
            uniform_load,
            n_bins=6,
            fidelity_tier=FidelityTier.F0,
            verdict=within_verdict(),
        )

        assert result.credibility.status is EnvelopeStatus.WITHIN

    def test_no_verdict_leaves_the_status_unstated(self, uniform_result) -> None:
        """Absent a verdict the status is ``None``, never an optimistic default."""
        assert uniform_result.credibility.status is None

    def test_a_refused_verdict_refuses_the_distribution(self, uniform_load) -> None:
        """No number may be reported when the solver refused the query."""
        with pytest.raises(OutOfEnvelopeError):
            spanwise_load(
                uniform_load,
                n_bins=6,
                fidelity_tier=FidelityTier.F0,
                verdict=refused_verdict(),
            )

    def test_nothing_here_is_measured(self, uniform_result) -> None:
        """No spanwise sole-pressure corpus exists for any tier."""
        assert uniform_result.credibility.measured_constants() == ()

    def test_the_sole_load_may_not_be_quoted_as_the_sand_response(
        self, uniform_result
    ) -> None:
        """F0 resolves no grain: this is what the sole carried, not sand flow."""
        with pytest.raises(BunkerShot3DValueError, match="not the sand"):
            uniform_result.credibility.require_sand_response()

    def test_the_summary_names_the_tier_and_the_reasons(self, uniform_result) -> None:
        """A report shows this beside the chart, so it cannot be cropped off."""
        summary = uniform_result.credibility.summary()

        assert "F0" in summary
        assert "sand" in summary

    def test_the_result_summarises_the_numbers_a_designer_reads(
        self, uniform_result
    ) -> None:
        """One line each for balance, centroid and the outer thirds."""
        summary = uniform_result.summary()

        assert "balance" in summary
        assert "centroid" in summary
        assert "outer third" in summary
