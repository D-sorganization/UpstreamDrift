"""Bed feasibility precondition tests (issue #8610, defects B26/B29).

The canonical BunkerShot3D configuration asks for 50,000 grains of d = 0.4 mm
in a 0.4 x 0.3 x 0.1 m domain. That is a solid volume fraction of 1.4e-4 -- a
settled bed 0.023 mm deep, about one seventeenth of a single grain diameter.
Every "bunker shot" run from that configuration swung a club through an
essentially empty box. These tests reproduce the arithmetic and pin the guard
that makes it impossible to configure again.
"""

from __future__ import annotations

import math

import pytest
from bunkershot3d.sand.bed import BunkerBedGeometry
from bunkershot3d.sand.exceptions import InfeasibleBedError
from bunkershot3d.sand.feasibility import (
    MAX_PHYSICAL_SOLID_FRACTION,
    achieved_solid_fraction,
    evaluate_bed_feasibility,
    grain_volume_m3,
    require_feasible_bed,
    required_grain_count,
    settled_bed_depth_m,
)

pytestmark = pytest.mark.unit

_B29_BED = BunkerBedGeometry(depth_m=0.1, plan_length_m=0.4, plan_width_m=0.3)
_B29_COUNT = 50_000
_B29_DIAMETER_M = 4.0e-4
_TARGET_SOLID_FRACTION = 0.60


class TestB29Arithmetic:
    def test_grain_volume(self) -> None:
        assert grain_volume_m3(_B29_DIAMETER_M) == pytest.approx(
            (math.pi / 6.0) * _B29_DIAMETER_M**3
        )

    def test_configured_solid_fraction_is_one_point_four_e_minus_four(self) -> None:
        phi = achieved_solid_fraction(
            grain_count=_B29_COUNT,
            grain_diameter_m=_B29_DIAMETER_M,
            bulk_volume_m3=_B29_BED.bulk_volume_m3,
        )
        assert phi == pytest.approx(1.4e-4, rel=1e-2)

    def test_settled_bed_is_twenty_three_micrometres_deep(self) -> None:
        depth_m = settled_bed_depth_m(
            grain_count=_B29_COUNT,
            grain_diameter_m=_B29_DIAMETER_M,
            plan_area_m2=_B29_BED.plan_area_m2,
            solid_fraction=_TARGET_SOLID_FRACTION,
        )
        assert depth_m == pytest.approx(2.33e-5, rel=1e-2)
        assert depth_m < _B29_DIAMETER_M  # thinner than one grain

    def test_a_hundred_millimetre_usga_base_needs_two_hundred_million_grains(
        self,
    ) -> None:
        count = required_grain_count(
            bulk_volume_m3=_B29_BED.bulk_volume_m3,
            solid_fraction=_TARGET_SOLID_FRACTION,
            grain_diameter_m=_B29_DIAMETER_M,
        )
        assert count == 214_859_174
        assert round(count, -7) == 2.1e8  # the 2.1e8 quoted in ADR-0032
        assert count / _B29_COUNT > 4000.0

    def test_even_a_ten_millimetre_token_bed_needs_twenty_million_grains(self) -> None:
        token_bed = BunkerBedGeometry(
            depth_m=0.010, plan_length_m=0.4, plan_width_m=0.3
        )
        count = required_grain_count(
            bulk_volume_m3=token_bed.bulk_volume_m3,
            solid_fraction=_TARGET_SOLID_FRACTION,
            grain_diameter_m=_B29_DIAMETER_M,
        )
        assert round(count, -6) == 2.1e7


class TestRefusal:
    def test_the_canonical_configuration_is_refused(self) -> None:
        report = evaluate_bed_feasibility(
            bed=_B29_BED,
            grain_count=_B29_COUNT,
            grain_diameter_m=_B29_DIAMETER_M,
            target_solid_fraction=_TARGET_SOLID_FRACTION,
        )
        assert not report.is_feasible
        assert round(report.required_grain_count, -7) == 2.1e8

    def test_refusal_message_is_actionable(self) -> None:
        with pytest.raises(InfeasibleBedError) as excinfo:
            require_feasible_bed(
                bed=_B29_BED,
                grain_count=_B29_COUNT,
                grain_diameter_m=_B29_DIAMETER_M,
                target_solid_fraction=_TARGET_SOLID_FRACTION,
            )
        message = str(excinfo.value)
        # names the defect, the numbers, and at least one concrete remedy
        assert "50000" in message.replace(",", "").replace("_", "")
        assert "214" in message.replace(",", "").replace("_", "")
        for token in ("solid fraction", "depth", "grain diameter"):
            assert token in message.lower()
        assert "coarse" in message.lower()

    def test_an_over_filled_domain_is_refused(self) -> None:
        """More grains than random close packing can hold."""
        overfull = int(
            _B29_BED.bulk_volume_m3
            / grain_volume_m3(_B29_DIAMETER_M)
            * MAX_PHYSICAL_SOLID_FRACTION
            * 1.5
        )
        with pytest.raises(InfeasibleBedError, match="exceeds"):
            require_feasible_bed(
                bed=_B29_BED,
                grain_count=overfull,
                grain_diameter_m=_B29_DIAMETER_M,
                target_solid_fraction=_TARGET_SOLID_FRACTION,
            )

    def test_a_consistent_configuration_is_accepted(self) -> None:
        count = required_grain_count(
            bulk_volume_m3=_B29_BED.bulk_volume_m3,
            solid_fraction=_TARGET_SOLID_FRACTION,
            grain_diameter_m=_B29_DIAMETER_M,
        )
        report = evaluate_bed_feasibility(
            bed=_B29_BED,
            grain_count=count,
            grain_diameter_m=_B29_DIAMETER_M,
            target_solid_fraction=_TARGET_SOLID_FRACTION,
        )
        assert report.is_feasible
        assert report.reasons == ()
        assert report.depth_ratio == pytest.approx(1.0, rel=1e-6)
        require_feasible_bed(
            bed=_B29_BED,
            grain_count=count,
            grain_diameter_m=_B29_DIAMETER_M,
            target_solid_fraction=_TARGET_SOLID_FRACTION,
        )

    def test_coarse_graining_can_make_a_bed_feasible(self) -> None:
        """The documented remedy has to actually work."""
        coarse_diameter_m = _B29_DIAMETER_M * 10.0
        count = required_grain_count(
            bulk_volume_m3=_B29_BED.bulk_volume_m3,
            solid_fraction=_TARGET_SOLID_FRACTION,
            grain_diameter_m=coarse_diameter_m,
        )
        assert count < 3.0e5
        require_feasible_bed(
            bed=_B29_BED,
            grain_count=count,
            grain_diameter_m=coarse_diameter_m,
            target_solid_fraction=_TARGET_SOLID_FRACTION,
        )

    def test_a_resource_ceiling_is_enforced_when_requested(self) -> None:
        count = required_grain_count(
            bulk_volume_m3=_B29_BED.bulk_volume_m3,
            solid_fraction=_TARGET_SOLID_FRACTION,
            grain_diameter_m=_B29_DIAMETER_M,
        )
        with pytest.raises(InfeasibleBedError, match="tractab"):
            require_feasible_bed(
                bed=_B29_BED,
                grain_count=count,
                grain_diameter_m=_B29_DIAMETER_M,
                target_solid_fraction=_TARGET_SOLID_FRACTION,
                max_grain_count=5_000_000,
            )

    def test_the_guard_raises_rather_than_asserting(self) -> None:
        """``python -O`` strips ``assert``; the feasibility guard must not."""
        import inspect

        from bunkershot3d.sand import feasibility

        assert "assert " not in inspect.getsource(feasibility)


class TestArgumentValidation:
    @pytest.mark.parametrize("diameter_m", [0.0, -1e-4, float("nan")])
    def test_bad_diameter_raises(self, diameter_m: float) -> None:
        with pytest.raises(InfeasibleBedError, match="diameter"):
            grain_volume_m3(diameter_m)

    def test_grain_count_must_be_positive(self) -> None:
        with pytest.raises(InfeasibleBedError, match="grain count"):
            achieved_solid_fraction(
                grain_count=0, grain_diameter_m=4e-4, bulk_volume_m3=0.012
            )

    def test_target_solid_fraction_must_be_physical(self) -> None:
        with pytest.raises(InfeasibleBedError, match="solid fraction"):
            required_grain_count(
                bulk_volume_m3=0.012,
                solid_fraction=MAX_PHYSICAL_SOLID_FRACTION + 0.05,
                grain_diameter_m=4e-4,
            )

    def test_a_grain_larger_than_the_bed_is_refused(self) -> None:
        with pytest.raises(InfeasibleBedError, match="larger than"):
            evaluate_bed_feasibility(
                bed=_B29_BED,
                grain_count=10,
                grain_diameter_m=0.5,
                target_solid_fraction=_TARGET_SOLID_FRACTION,
            )
