"""Particle size distribution tests (issue #8610).

The invariants exercised with Hypothesis are the ones a sieve analysis must
satisfy by construction: bin fractions sum to one, the cumulative-passing
curve is monotone, ``d10 <= d50 <= d60`` and the uniformity coefficient
``Cu = d60 / d10`` is at least one.
"""

from __future__ import annotations

import pytest
from bunkershot3d.sand.exceptions import ParticleSizeDistributionError
from bunkershot3d.sand.psd import ParticleSizeDistribution
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit

# Turf & Soil Diagnostics bunker-sand mid-band, expressed as USDA size classes.
_MIDBAND_EDGES_M = (2e-6, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
_MIDBAND_FRACTIONS = (0.015, 0.025, 0.080, 0.435, 0.380, 0.055, 0.010)


def _midband() -> ParticleSizeDistribution:
    return ParticleSizeDistribution.from_bins(
        bin_edges_m=_MIDBAND_EDGES_M,
        bin_fractions=_MIDBAND_FRACTIONS,
        name="test-midband",
    )


class TestConstruction:
    def test_from_bins_builds_a_cumulative_curve(self) -> None:
        psd = _midband()
        assert psd.sieve_openings_m == _MIDBAND_EDGES_M
        assert psd.fraction_passing[0] == pytest.approx(0.0)
        assert psd.fraction_passing[-1] == pytest.approx(1.0)
        assert psd.fraction_passing[1] == pytest.approx(0.015)
        assert psd.fraction_passing[4] == pytest.approx(0.555)

    def test_bin_fractions_round_trip(self) -> None:
        psd = _midband()
        assert psd.bin_fractions[1:] == pytest.approx(_MIDBAND_FRACTIONS)
        assert sum(psd.bin_fractions) == pytest.approx(1.0)

    def test_from_retained_fractions_matches_from_bins(self) -> None:
        # A real sieve stack: coarsest sieve first, pan fraction last.
        psd = ParticleSizeDistribution.from_retained_fractions(
            sieve_openings_m=(2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2e-6),
            mass_fraction_retained=(
                0.010,  # retained on 2.00 mm (gravel)
                0.055,  # 1.00-2.00 mm, very coarse
                0.380,  # 0.50-1.00 mm, coarse
                0.435,  # 0.25-0.50 mm, medium
                0.080,  # 0.10-0.25 mm, fine
                0.025,  # 0.05-0.10 mm, very fine
                0.015,  # 0.002-0.05 mm, silt
                0.000,  # pan
            ),
            largest_particle_m=4e-3,
        )
        assert psd.fraction_passing == pytest.approx(_midband().fraction_passing)

    @pytest.mark.parametrize(
        ("edges", "fractions", "match"),
        [
            ((1e-4,), (), "at least two"),
            ((1e-4, 5e-5), (1.0,), "ascending"),
            ((-1e-4, 5e-5), (1.0,), "positive"),
            ((1e-4, 5e-4), (0.5,), "sum to 1"),
            ((1e-4, 5e-4), (-0.2,), "negative"),
        ],
    )
    def test_invalid_specifications_raise(
        self, edges: tuple[float, ...], fractions: tuple[float, ...], match: str
    ) -> None:
        with pytest.raises(ParticleSizeDistributionError, match=match):
            ParticleSizeDistribution.from_bins(
                bin_edges_m=edges, bin_fractions=fractions
            )

    def test_non_monotone_passing_curve_raises(self) -> None:
        with pytest.raises(ParticleSizeDistributionError, match="monotone"):
            ParticleSizeDistribution(
                sieve_openings_m=(1e-4, 5e-4, 1e-3),
                fraction_passing=(0.0, 0.9, 0.5),
            )

    def test_passing_curve_must_reach_one(self) -> None:
        with pytest.raises(ParticleSizeDistributionError, match="1.0"):
            ParticleSizeDistribution(
                sieve_openings_m=(1e-4, 5e-4, 1e-3),
                fraction_passing=(0.0, 0.5, 0.9),
            )


class TestPercentileDiameters:
    def test_midband_percentiles(self) -> None:
        psd = _midband()
        assert psd.d10_m == pytest.approx(1.988e-4, rel=1e-3)
        assert psd.d50_m == pytest.approx(4.581e-4, rel=1e-3)
        assert psd.d60_m == pytest.approx(5.427e-4, rel=1e-3)

    def test_midband_uniformity_coefficient_is_in_the_usga_band(self) -> None:
        psd = _midband()
        assert psd.uniformity_coefficient == pytest.approx(2.73, rel=1e-2)
        assert 2.0 <= psd.uniformity_coefficient <= 5.0

    def test_coefficient_of_curvature(self) -> None:
        psd = _midband()
        expected = psd.d30_m**2 / (psd.d10_m * psd.d60_m)
        assert psd.coefficient_of_curvature == pytest.approx(expected)

    def test_exact_sieve_hit_returns_that_opening(self) -> None:
        psd = _midband()
        assert psd.diameter_at_passing(0.015) == pytest.approx(5e-5)

    def test_percentile_below_finest_sieve_is_refused(self) -> None:
        psd = ParticleSizeDistribution(
            sieve_openings_m=(1e-4, 1e-3),
            fraction_passing=(0.2, 1.0),
        )
        with pytest.raises(ParticleSizeDistributionError, match="finest sieve"):
            psd.diameter_at_passing(0.10)

    @pytest.mark.parametrize("passing", [0.0, 1.0, -0.1, 1.5])
    def test_percentile_outside_the_open_unit_interval_is_refused(
        self, passing: float
    ) -> None:
        with pytest.raises(ParticleSizeDistributionError, match="between 0 and 1"):
            _midband().diameter_at_passing(passing)


class TestFractionQueries:
    def test_coarse_plus_medium_band(self) -> None:
        psd = _midband()
        assert psd.fraction_between(2.5e-4, 1e-3) == pytest.approx(0.815)

    def test_fraction_finer_than_a_sieve_opening(self) -> None:
        assert _midband().fraction_finer_than(5e-5) == pytest.approx(0.015)

    def test_fraction_between_is_order_checked(self) -> None:
        with pytest.raises(ParticleSizeDistributionError, match="lower"):
            _midband().fraction_between(1e-3, 2.5e-4)


class TestVolumeEquivalentDiameter:
    def test_fines_dominate_the_number_count(self) -> None:
        """Number-weighted mean is far below d50 because silt dominates counts."""
        psd = _midband()
        assert psd.volume_equivalent_diameter_m < psd.d50_m
        assert psd.volume_equivalent_diameter_m == pytest.approx(4.05e-5, rel=5e-2)

    def test_monodisperse_distribution_recovers_its_own_diameter(self) -> None:
        psd = ParticleSizeDistribution.from_bins(
            bin_edges_m=(3.0e-4 / 1.001, 3.0e-4 * 1.001),
            bin_fractions=(1.0,),
        )
        assert psd.volume_equivalent_diameter_m == pytest.approx(3.0e-4, rel=1e-3)


# --------------------------------------------------------------------------
# Hypothesis invariants
# --------------------------------------------------------------------------


@st.composite
def _psd_strategy(draw: st.DrawFn) -> ParticleSizeDistribution:
    n_bins = draw(st.integers(min_value=1, max_value=8))
    exponents = draw(
        st.lists(
            st.floats(min_value=-6.0, max_value=-2.0, allow_subnormal=False),
            min_size=n_bins + 1,
            max_size=n_bins + 1,
            unique=True,
        )
    )
    edges = tuple(10.0**e for e in sorted(exponents))
    weights = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=n_bins,
            max_size=n_bins,
        )
    )
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0] * n_bins
        total = float(n_bins)
    fractions = tuple(w / total for w in weights)
    return ParticleSizeDistribution.from_bins(
        bin_edges_m=edges, bin_fractions=fractions
    )


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_psd_strategy())
def test_bin_fractions_sum_to_one(psd: ParticleSizeDistribution) -> None:
    assert sum(psd.bin_fractions) == pytest.approx(1.0, abs=1e-9)


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_psd_strategy())
def test_cumulative_curve_is_monotone(psd: ParticleSizeDistribution) -> None:
    passing = psd.fraction_passing
    assert all(b >= a for a, b in zip(passing, passing[1:], strict=False))
    assert passing[-1] == pytest.approx(1.0, abs=1e-9)


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_psd_strategy())
def test_percentile_diameters_are_ordered(psd: ParticleSizeDistribution) -> None:
    assert psd.d10_m <= psd.d50_m <= psd.d60_m


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_psd_strategy())
def test_uniformity_coefficient_is_at_least_one(
    psd: ParticleSizeDistribution,
) -> None:
    assert psd.uniformity_coefficient >= 1.0


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_psd_strategy())
def test_volume_equivalent_diameter_is_inside_the_size_range(
    psd: ParticleSizeDistribution,
) -> None:
    assert psd.sieve_openings_m[0] <= psd.volume_equivalent_diameter_m
    assert psd.volume_equivalent_diameter_m <= psd.sieve_openings_m[-1]
