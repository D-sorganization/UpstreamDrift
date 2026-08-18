"""Penetrometer firmness scale tests (issue #8610).

USGA / Turf & Soil Diagnostics rate a bunker sand by pushing a penetrometer
with a golf ball on the tip into oven-dried, loosened sand until the ball is
buried to its hemisphere. Force in kg/cm^2:

===========  ==========  ===========
Bury a ball  kg/cm^2     Rating
===========  ==========  ===========
high         < 1.8       undesirable
moderate     1.8 - 2.2   acceptable
slight       2.2 - 2.4   acceptable
very low     > 2.4       desirable
===========  ==========  ===========
"""

from __future__ import annotations

import pytest
from bunkershot3d.sand.exceptions import SandModelError
from bunkershot3d.sand.firmness import (
    FIRMNESS_SWEEP_KG_PER_CM2,
    KG_PER_CM2_IN_PASCAL,
    FirmnessRating,
    firmness_kg_per_cm2_from_pa,
    firmness_pa_from_kg_per_cm2,
    firmness_rating,
    relative_density_from_firmness,
)

pytestmark = pytest.mark.unit


class TestUnits:
    def test_one_kg_per_cm2_is_standard_gravity_over_a_square_centimetre(self) -> None:
        assert pytest.approx(9.80665e4) == KG_PER_CM2_IN_PASCAL
        assert firmness_pa_from_kg_per_cm2(1.0) == pytest.approx(98066.5)

    def test_round_trip(self) -> None:
        for value in FIRMNESS_SWEEP_KG_PER_CM2:
            pa = firmness_pa_from_kg_per_cm2(value)
            assert firmness_kg_per_cm2_from_pa(pa) == pytest.approx(value)

    def test_non_positive_firmness_is_refused(self) -> None:
        with pytest.raises(SandModelError, match="firmness"):
            firmness_pa_from_kg_per_cm2(0.0)


class TestRating:
    @pytest.mark.parametrize(
        ("value", "rating"),
        [
            (1.6, FirmnessRating.UNDESIRABLE),
            (1.79, FirmnessRating.UNDESIRABLE),
            (1.8, FirmnessRating.ACCEPTABLE),
            (2.0, FirmnessRating.ACCEPTABLE),
            (2.2, FirmnessRating.ACCEPTABLE),
            (2.4, FirmnessRating.ACCEPTABLE),
            (2.41, FirmnessRating.DESIRABLE),
            (2.8, FirmnessRating.DESIRABLE),
        ],
    )
    def test_published_thresholds(self, value: float, rating: FirmnessRating) -> None:
        assert firmness_rating(value) is rating

    def test_the_sweep_spans_undesirable_to_desirable(self) -> None:
        assert FIRMNESS_SWEEP_KG_PER_CM2 == (1.6, 2.0, 2.4, 2.8)
        ratings = [firmness_rating(v) for v in FIRMNESS_SWEEP_KG_PER_CM2]
        assert ratings[0] is FirmnessRating.UNDESIRABLE
        assert ratings[-1] is FirmnessRating.DESIRABLE


class TestRelativeDensityMapping:
    def test_mapping_is_monotone_over_the_sweep(self) -> None:
        values = [relative_density_from_firmness(v) for v in FIRMNESS_SWEEP_KG_PER_CM2]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_mapping_is_clipped_at_both_ends(self) -> None:
        assert relative_density_from_firmness(0.5) == pytest.approx(0.0)
        assert relative_density_from_firmness(9.0) == pytest.approx(1.0)

    def test_sweep_endpoints_are_loose_and_dense(self) -> None:
        assert relative_density_from_firmness(1.6) < 0.2
        assert relative_density_from_firmness(2.8) > 0.8
