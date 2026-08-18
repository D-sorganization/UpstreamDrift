"""The reference-data registry, and the register of what does not exist (#8616).

The most important assertions here are the *negative* ones. An exhaustive
enumeration of ISEA / Procedia Engineering / *Engineering of Sport* volumes 2,
13, 32, 34, 60, 72, 112 and 147, of *Sports Engineering* and of the *Journal of
Sports Sciences* found **no** paper on bunkers, sand, wedges, club-turf
interaction or divot mechanics. That is a real gap in the field, not a search
failure, and the code has to make it impossible to quietly validate against
data that was never collected.
"""

from __future__ import annotations

import pytest

from bunkershot3d.vandv import (
    GRANULAR_INTRUSION_BENCHMARK,
    UNMEASURED_QUANTITIES,
    WIVOU_2016,
    NoReferenceDataError,
    domain_overlap,
    reference_dataset,
    require_measurable,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]


class TestWivou2016:
    """The only primary greenside-bunker dataset located."""

    def test_measured_ranges_match_the_publication(self) -> None:
        entry = WIVOU_2016.ranges["entry_distance_behind_ball_m"]
        divot = WIVOU_2016.ranges["divot_depth_m"]
        carry = WIVOU_2016.ranges["carry_m"]
        assert (entry.low, entry.high) == (0.080, 0.280)
        assert (divot.low, divot.high) == (0.025, 0.052)
        assert (carry.low, carry.high) == (1.0, 12.0)

    def test_carry_correlations_match_the_publication(self) -> None:
        assert WIVOU_2016.correlations["entry_distance_behind_ball_m"] == -0.98
        assert WIVOU_2016.correlations["divot_depth_m"] == -0.91

    def test_the_dataset_records_what_it_does_not_contain(self) -> None:
        """The paper has no clubhead speed, launch angle, ball speed or spin."""
        absent = set(WIVOU_2016.absent_quantities)
        assert {
            "clubhead_speed_m_s",
            "ball_launch_angle_rad",
            "ball_speed_m_s",
            "ball_spin_rad_s",
        } <= absent

    def test_citing_the_dataset_for_a_quantity_it_lacks_is_refused(self) -> None:
        with pytest.raises(NoReferenceDataError, match="does not contain"):
            WIVOU_2016.value_range("ball_spin_rad_s")

    def test_the_dataset_is_reachable_by_name(self) -> None:
        assert reference_dataset("wivou_2016") is WIVOU_2016


class TestGranularIntrusionBenchmark:
    """The wheel-in-sand benchmark behind the DRFT constants."""

    def test_sinkage_errors_rank_rft_ahead_of_bekker(self) -> None:
        errors = GRANULAR_INTRUSION_BENCHMARK.sinkage_mae_m
        assert errors["rft"] == pytest.approx(2.7e-3)
        assert errors["mpm"] == pytest.approx(3.2e-3)
        assert errors["bekker_wong_reece"] == pytest.approx(26.1e-3)
        assert errors["rft"] < errors["mpm"] < errors["bekker_wong_reece"]

    def test_the_corpus_speed_ceiling_is_recorded(self) -> None:
        assert GRANULAR_INTRUSION_BENCHMARK.max_speed_m_s == pytest.approx(1.44)

    def test_the_natural_sand_bias_is_recorded(self) -> None:
        assert GRANULAR_INTRUSION_BENCHMARK.natural_sand_bias == pytest.approx(0.35)


class TestUnmeasuredQuantities:
    """The register of quantities no published measurement exists for."""

    @pytest.mark.parametrize(
        "quantity",
        [
            "ball_launch_angle_rad",
            "ball_speed_m_s",
            "ball_spin_rad_s",
            "clubhead_deceleration_m_s2",
            "energy_split_fraction",
            "ejecta_mass_kg",
            "coefficient_of_restitution_through_sand",
        ],
    )
    def test_every_ball_outcome_quantity_is_registered_as_unmeasured(
        self, quantity: str
    ) -> None:
        assert quantity in UNMEASURED_QUANTITIES

    def test_requiring_an_unmeasured_quantity_raises(self) -> None:
        with pytest.raises(NoReferenceDataError, match="no published measurement"):
            require_measurable("ball_spin_rad_s")

    def test_the_refusal_explains_that_the_gap_is_real(self) -> None:
        with pytest.raises(NoReferenceDataError) as excinfo:
            require_measurable("ejecta_mass_kg")
        assert "not a search failure" in str(excinfo.value)

    def test_a_measurable_quantity_passes(self) -> None:
        require_measurable("divot_depth_m")

    def test_every_register_entry_carries_a_reason(self) -> None:
        for quantity, reason in UNMEASURED_QUANTITIES.items():
            assert reason.strip(), f"{quantity} has no stated reason"


class TestDomainOverlap:
    """How much of a swept range the published measurement actually covers."""

    def test_full_containment_is_one(self) -> None:
        assert domain_overlap((0.1, 0.2), (0.0, 1.0)).covered_fraction == 1.0

    def test_disjoint_ranges_are_zero(self) -> None:
        overlap = domain_overlap((0.0, 0.1), (0.5, 1.0))
        assert overlap.covered_fraction == 0.0
        assert overlap.is_extrapolation

    def test_partial_overlap_is_the_measured_share(self) -> None:
        overlap = domain_overlap((0.0, 0.4), (0.2, 1.0))
        assert overlap.covered_fraction == pytest.approx(0.5)
        assert overlap.is_extrapolation

    def test_the_declared_entry_sweep_runs_below_any_measurement(self) -> None:
        """The package sweeps entry distance from 25 mm; Wivou starts at 80 mm."""
        overlap = domain_overlap(
            (0.025, 0.150), WIVOU_2016.value_range("entry_distance_behind_ball_m")
        )
        assert overlap.covered_fraction == pytest.approx(0.56)
        assert overlap.is_extrapolation

    def test_an_inverted_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            domain_overlap((1.0, 0.0), (0.0, 1.0))
