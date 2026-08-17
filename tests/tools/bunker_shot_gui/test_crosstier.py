"""The cross-tier comparison, headless (issue #8713, epic #8699).

Two things are being tested here and they are not the same thing.

The **arithmetic** -- ratios, agreement classes, divergence spans, the
inertial-share crossover, the licence statement -- is tested on hand-built
value objects, because it has to be right for numbers nobody has run yet
and because an MPM march inside a unit test would put a 30-second solve
behind an assertion about a division.

The **wiring** -- that F0's recorded pose really does reach F1 unchanged,
and that the pair produces the divergence ADR-0033 measured -- is tested
once, on a deliberately coarse bed, and is checked for *sign and shape*
rather than pinned to a number the discretisation would move.

What is deliberately not tested is agreement. ADR-0033 is explicit that
consistency between two uncalibrated models is not validation, so a test
asserting the two tiers agree would be asserting the one thing this view
exists to refuse to claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.solvers import EnvelopeStatus, FidelityTier
from bunkershot3d.solvers.envelope import MAX_VALIDATED_SPEED_M_S
from bunkershot3d.solvers.mpm.state import SurfaceDepression
from bunkershot3d.solvers.mpm.verification import F0CrossCheck
from src.tools.bunker_shot_gui.crosstier import (
    DECLARED_AGREEMENT_BAND,
    AgreementClass,
    ComparedQuantity,
    CrossTierComparison,
    CrossTierProbe,
    QuantityAgreement,
    inertial_share_crossover,
    licence_statement,
)
from src.tools.bunker_shot_gui.traces import ValidityBand

pytestmark = pytest.mark.unit


# --------------------------------------------------------------- fixtures


def cross_check(
    *,
    speed_m_s: float,
    f0_force_n: float,
    f1_force_n: float,
    f0_inertial_share: float,
    f1_flux_share: float,
    depth_m: float = 0.010,
    f1_section_area_m2: float = 4.0e-4,
    n_empty_bins: int = 0,
) -> F0CrossCheck:
    """One paired query, built from the shares rather than solved for them."""
    direction = np.array([1.0, 0.0, -0.4])
    direction = direction / np.linalg.norm(direction)
    return F0CrossCheck(
        speed_m_s=speed_m_s,
        f0_force_n=f0_force_n * direction,
        f1_force_n=f1_force_n * direction,
        f0_depth_force_n=f0_force_n * (1.0 - f0_inertial_share) * direction,
        f0_inertial_force_n=f0_force_n * f0_inertial_share * direction,
        f1_stress_force_n=f1_force_n * (1.0 - f1_flux_share) * direction,
        f1_flux_force_n=f1_force_n * f1_flux_share * direction,
        submerged_depth_m=depth_m,
        f0_max_depth_m=depth_m,
        f1_max_depth_m=depth_m,
        f1_divot=SurfaceDepression(
            section_area_m2=f1_section_area_m2,
            max_depth_m=depth_m,
            n_bins=64,
            n_empty_bins=n_empty_bins,
            bed_width_m=0.30,
        ),
        effective_width_m=0.030,
    )


def probe(
    frame: int,
    time_s: float,
    *,
    speed_m_s: float = 12.0,
    f0_force_n: float = 40.0,
    f1_force_n: float = 60.0,
    f0_inertial_share: float = 0.93,
    f1_flux_share: float = 0.69,
    f0_section_area_m2: float = 5.0e-4,
    f1_section_area_m2: float = 4.0e-4,
    depth_m: float = 0.010,
    n_empty_bins: int = 0,
) -> CrossTierProbe:
    return CrossTierProbe(
        frame=frame,
        time_s=time_s,
        check=cross_check(
            speed_m_s=speed_m_s,
            f0_force_n=f0_force_n,
            f1_force_n=f1_force_n,
            f0_inertial_share=f0_inertial_share,
            f1_flux_share=f1_flux_share,
            depth_m=depth_m,
            f1_section_area_m2=f1_section_area_m2,
            n_empty_bins=n_empty_bins,
        ),
        f0_divot_section_area_m2=f0_section_area_m2,
        f0_sole_depth_m=depth_m,
        declared_width_m=0.030,
        bulk_density_kg_m3=1550.0,
    )


def comparison(probes: tuple[CrossTierProbe, ...]) -> CrossTierComparison:
    """A comparison over a synthetic 9-sample F0 record."""
    time_s = np.linspace(0.0, 8.0e-3, 9)
    speed = np.linspace(25.0, 21.0, 9)
    force = np.zeros((9, 3))
    force[:, 0] = -np.linspace(10.0, 50.0, 9)
    force[:, 2] = np.linspace(10.0, 50.0, 9)
    return CrossTierComparison(
        shot_probes=probes,
        time_s=time_s,
        f0_force_n=force,
        f0_sole_depth_m=np.linspace(0.0, 0.012, 9),
        f0_speed_m_s=speed,
        f0_divot_section_area_m2=np.linspace(0.0, 6.0e-4, 9),
        band=ValidityBand(
            time_s=time_s,
            statuses=tuple([EnvelopeStatus.BEYOND_VALIDATION] * 9),
        ),
        head_mass_kg=0.300,
        declared_width_m=0.030,
        bulk_density_kg_m3=1550.0,
    )


# ------------------------------------------------------------- agreement


class TestQuantityAgreementIsARatioAndAVerdictOnIt:
    """The number, and the declared band it is judged against."""

    def test_the_ratio_is_f1_over_f0(self) -> None:
        agreement = QuantityAgreement(
            quantity=ComparedQuantity.WRENCH, f0_value=40.0, f1_value=60.0
        )
        assert agreement.ratio == pytest.approx(1.5)
        assert agreement.relative_difference == pytest.approx(0.5)

    def test_a_ratio_inside_the_band_is_consistent(self) -> None:
        agreement = QuantityAgreement(
            quantity=ComparedQuantity.SOLE_DEPTH, f0_value=0.0100, f1_value=0.0105
        )
        assert agreement.agreement is AgreementClass.CONSISTENT
        assert not agreement.diverged

    def test_a_ratio_outside_the_band_is_divergent(self) -> None:
        agreement = QuantityAgreement(
            quantity=ComparedQuantity.WRENCH, f0_value=166.0, f1_value=446.0
        )
        assert agreement.agreement is AgreementClass.DIVERGENT
        assert agreement.diverged

    def test_the_band_is_symmetric_in_the_ratio(self) -> None:
        """2x and 0.5x are the same disagreement, so the band is on log ratio."""
        up = QuantityAgreement(
            quantity=ComparedQuantity.WRENCH, f0_value=10.0, f1_value=20.0
        )
        down = QuantityAgreement(
            quantity=ComparedQuantity.WRENCH, f0_value=20.0, f1_value=10.0
        )
        assert up.log_ratio == pytest.approx(down.log_ratio)
        assert up.agreement is down.agreement

    def test_a_tier_that_produced_nothing_is_incomparable_not_agreeing(self) -> None:
        """Two zeroes are not a match; they are an unanswered question."""
        nothing = QuantityAgreement(
            quantity=ComparedQuantity.DIVOT_SECTION, f0_value=0.0, f1_value=0.0
        )
        assert nothing.agreement is AgreementClass.INCOMPARABLE
        one_sided = QuantityAgreement(
            quantity=ComparedQuantity.DIVOT_SECTION, f0_value=0.0, f1_value=4.0e-4
        )
        assert one_sided.agreement is AgreementClass.INCOMPARABLE

    def test_a_negative_magnitude_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            QuantityAgreement(
                quantity=ComparedQuantity.WRENCH, f0_value=-1.0, f1_value=1.0
            )

    def test_a_band_of_zero_is_refused(self) -> None:
        """A zero band would class every finite discretisation as divergent."""
        with pytest.raises(ValueError, match="band"):
            QuantityAgreement(
                quantity=ComparedQuantity.WRENCH,
                f0_value=1.0,
                f1_value=1.0,
                band=0.0,
            )

    def test_the_summary_names_the_quantity_its_unit_and_the_band(self) -> None:
        agreement = QuantityAgreement(
            quantity=ComparedQuantity.WRENCH, f0_value=40.0, f1_value=60.0
        )
        text = agreement.summary()
        assert "N" in text
        assert "1.5" in text
        assert f"{DECLARED_AGREEMENT_BAND:g}" in text


class TestEveryComparedQuantitySaysWhatEachTierMeansByIt:
    """The two tiers use the same word for different measurements."""

    def test_each_quantity_carries_a_unit_and_a_note(self) -> None:
        for quantity in ComparedQuantity:
            assert quantity.unit
            assert quantity.label
            assert len(quantity.note) > 30, quantity

    def test_the_divot_note_says_the_two_divots_are_different_things(self) -> None:
        note = ComparedQuantity.DIVOT_SECTION.note
        assert "envelope" in note
        assert "sand" in note

    def test_the_speed_lost_note_says_f1_is_one_way_coupled(self) -> None:
        assert "one-way" in ComparedQuantity.SPEED_LOST.note


# ----------------------------------------------------------------- probes


class TestAProbeIsOneInstantGivenToBothTiersUnchanged:
    def test_it_reports_both_magnitudes_and_the_direction_cosine(self) -> None:
        one = probe(4, 4.0e-3)
        assert one.f0_force_magnitude_n == pytest.approx(40.0)
        assert one.f1_force_magnitude_n == pytest.approx(60.0)
        assert one.direction_agreement == pytest.approx(1.0)

    def test_the_two_divot_masses_use_one_declared_width(self) -> None:
        """Otherwise the mass comparison is confounded by two assumptions."""
        one = probe(4, 4.0e-3, f0_section_area_m2=5.0e-4, f1_section_area_m2=4.0e-4)
        assert one.f0_divot_mass_kg == pytest.approx(5.0e-4 * 0.030 * 1550.0)
        assert one.f1_divot_mass_kg == pytest.approx(4.0e-4 * 0.030 * 1550.0)

    def test_the_inertial_share_gap_is_f0_minus_f1(self) -> None:
        one = probe(4, 4.0e-3, f0_inertial_share=0.93, f1_flux_share=0.69)
        assert one.inertial_share_gap == pytest.approx(0.24, abs=1e-6)

    def test_it_yields_an_agreement_per_instantaneous_quantity(self) -> None:
        one = probe(4, 4.0e-3)
        for quantity in (
            ComparedQuantity.WRENCH,
            ComparedQuantity.SOLE_DEPTH,
            ComparedQuantity.DIVOT_SECTION,
            ComparedQuantity.DIVOT_MASS,
        ):
            assert one.agreement(quantity).quantity is quantity

    def test_speed_lost_is_refused_at_a_single_instant(self) -> None:
        """It is a window integral; an instant cannot answer it."""
        with pytest.raises(ValueError, match="window"):
            probe(4, 4.0e-3).agreement(ComparedQuantity.SPEED_LOST)

    def test_an_unresolved_divot_is_flagged_rather_than_quoted_quietly(self) -> None:
        one = probe(4, 4.0e-3, n_empty_bins=3)
        assert not one.divot_fully_resolved
        assert "lower bound" in one.divot_caveat()


# ------------------------------------------------------------ comparison


class TestTheComparisonOverlaysThePairOnOneCursor:
    def test_the_probes_index_into_the_f0_record(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        assert model.n_frames == 9
        assert model.probe_frames == (2, 6)
        assert model.probe_times_s == pytest.approx((2.0e-3, 6.0e-3))

    def test_a_probe_outside_the_record_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            comparison(
                (probe(2, 2.0e-3), probe(99, 9.9e-3)),
            )

    def test_a_probe_whose_time_disagrees_with_its_frame_is_refused(self) -> None:
        """A probe drawn at the wrong moment is worse than a missing one."""
        with pytest.raises(ValueError, match="frame"):
            comparison((probe(2, 7.0e-3),))

    def test_a_comparison_with_no_probes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            comparison(())

    def test_the_shot_level_agreement_is_taken_at_f0s_peak_probe(self) -> None:
        weak = probe(2, 2.0e-3, f0_force_n=10.0, f1_force_n=11.0)
        strong = probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=120.0)
        model = comparison((weak, strong))
        assert model.peak_probe is strong
        assert model.agreement(ComparedQuantity.WRENCH).ratio == pytest.approx(3.0)

    def test_every_quantity_gets_an_agreement(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        covered = {item.quantity for item in model.agreements()}
        assert covered == set(ComparedQuantity)


class TestSpeedLostIsDerivedAndSaysSo:
    """F1 is driven kinematically, so it loses no speed of its own."""

    def test_f0s_speed_lost_is_read_off_its_own_record(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        # The synthetic record runs 25 -> 21 m/s over 9 samples, so the
        # window between samples 2 and 6 is half of the 4 m/s total.
        assert model.f0_speed_lost_m_s == pytest.approx(2.0)

    def test_f1s_speed_lost_borrows_the_magnitude_and_keeps_f0s_direction(
        self,
    ) -> None:
        """F0's own decrements, each weighted by the ``|F1|/|F0|`` ratio.

        Integrating F1's force directly would assume the two resultants
        point the same way, and the measured direction cosines say they do
        not.
        """
        model = comparison(
            (
                probe(2, 2.0e-3, f0_force_n=40.0, f1_force_n=60.0),
                probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=60.0),
            )
        )
        assert model.f1_speed_lost_m_s == pytest.approx(1.5 * 2.0, rel=1e-9)

    def test_a_ratio_that_varies_weights_where_the_deceleration_happened(
        self,
    ) -> None:
        """Not simply the peak ratio times F0's loss."""
        model = comparison(
            (
                probe(2, 2.0e-3, f0_force_n=40.0, f1_force_n=40.0),
                probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=120.0),
            )
        )
        # The record decelerates uniformly, so the ratio's mean over the
        # four intervals -- 1.25, 1.75, 2.25, 2.75 at the interval starts --
        # is what F0's 2.0 m/s loss is scaled by.
        assert model.f1_speed_lost_m_s == pytest.approx(2.0 * 2.0, rel=1e-9)
        assert model.f1_speed_lost_m_s < 3.0 * 2.0

    def test_it_needs_two_probes_to_form_a_window(self) -> None:
        model = comparison((probe(4, 4.0e-3),))
        with pytest.raises(ValueError, match="window"):
            _ = model.f1_speed_lost_m_s

    def test_the_agreement_carries_the_one_way_coupling_note(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        agreement = model.agreement(ComparedQuantity.SPEED_LOST)
        assert "one-way" in agreement.quantity.note


class TestDivergenceIsMarkedRatherThanLeftForTheEye:
    def test_a_divergent_stretch_becomes_a_span(self) -> None:
        model = comparison(
            (
                probe(2, 2.0e-3, f0_force_n=40.0, f1_force_n=42.0),
                probe(4, 4.0e-3, f0_force_n=40.0, f1_force_n=120.0),
                probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=130.0),
            )
        )
        spans = model.divergence_spans(ComparedQuantity.WRENCH)
        assert len(spans) == 1
        assert spans[0].start_s == pytest.approx(4.0e-3)
        assert spans[0].end_s == pytest.approx(6.0e-3)
        assert spans[0].worst_ratio == pytest.approx(130.0 / 40.0)

    def test_a_consistent_run_produces_no_spans(self) -> None:
        model = comparison(
            (
                probe(2, 2.0e-3, f0_force_n=40.0, f1_force_n=41.0),
                probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=42.0),
            )
        )
        assert model.divergence_spans(ComparedQuantity.WRENCH) == ()

    def test_a_lone_divergent_probe_still_marks_a_span(self) -> None:
        """A single instant of disagreement must not vanish for want of a pair."""
        model = comparison(
            (
                probe(2, 2.0e-3, f0_force_n=40.0, f1_force_n=41.0),
                probe(4, 4.0e-3, f0_force_n=40.0, f1_force_n=200.0),
                probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=42.0),
            )
        )
        spans = model.divergence_spans(ComparedQuantity.WRENCH)
        assert len(spans) == 1
        assert spans[0].start_s < 4.0e-3 < spans[0].end_s

    def test_the_span_label_names_the_quantity_and_the_ratio(self) -> None:
        model = comparison(
            (
                probe(2, 2.0e-3, f0_force_n=40.0, f1_force_n=120.0),
                probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=130.0),
            )
        )
        label = model.divergence_spans(ComparedQuantity.WRENCH)[0].label
        assert ComparedQuantity.WRENCH.label in label
        assert "3.25x" in label


# ------------------------------------------------------- the crossover


class TestTheInertialShareCrossover:
    """The sharpest single result: where F0's shortcut stops describing it."""

    def test_a_bracketed_crossing_is_located_by_interpolation(self) -> None:
        """ADR-0033's own numbers: F0 0.52 -> 0.93, F1 flat at 0.68 -> 0.69."""
        probes = (
            probe(0, 0.0, speed_m_s=5.0, f0_inertial_share=0.52, f1_flux_share=0.68),
            probe(
                4, 4.0e-3, speed_m_s=12.0, f0_inertial_share=0.93, f1_flux_share=0.69
            ),
        )
        crossover = inertial_share_crossover(probes)
        assert crossover is not None
        # gap goes -0.16 -> +0.24, so the crossing sits at 0.16/0.40 of the way.
        assert crossover.speed_m_s == pytest.approx(5.0 + 0.4 * 7.0, rel=1e-6)
        assert 0.68 <= crossover.shared_share <= 0.69

    def test_an_unbracketed_range_returns_nothing_rather_than_extrapolating(
        self,
    ) -> None:
        probes = (
            probe(0, 0.0, speed_m_s=20.0, f0_inertial_share=0.98, f1_flux_share=0.66),
            probe(
                4, 4.0e-3, speed_m_s=25.0, f0_inertial_share=0.99, f1_flux_share=0.65
            ),
        )
        assert inertial_share_crossover(probes) is None

    def test_one_probe_cannot_bracket_anything(self) -> None:
        assert inertial_share_crossover((probe(0, 0.0),)) is None

    def test_the_comparison_states_the_crossover_either_way(self) -> None:
        above = comparison(
            (
                probe(
                    2,
                    2.0e-3,
                    speed_m_s=25.0,
                    f0_inertial_share=0.99,
                    f1_flux_share=0.65,
                ),
                probe(
                    6,
                    6.0e-3,
                    speed_m_s=21.0,
                    f0_inertial_share=0.98,
                    f1_flux_share=0.66,
                ),
            )
        )
        text = above.crossover_summary()
        assert "above" in text.lower()
        assert "0.9" in text

    def test_the_summary_names_the_mechanism_not_only_the_number(self) -> None:
        probes = (
            probe(0, 0.0, speed_m_s=5.0, f0_inertial_share=0.52, f1_flux_share=0.68),
            probe(
                4, 4.0e-3, speed_m_s=12.0, f0_inertial_share=0.93, f1_flux_share=0.69
            ),
        )
        crossover = inertial_share_crossover(probes)
        assert crossover is not None
        assert "yield" in crossover.summary()

    def test_a_declared_speed_sweep_drives_the_crossover_when_present(self) -> None:
        """The shot need not span the crossing; a sweep is a separate probe set."""
        model = comparison((probe(2, 2.0e-3, speed_m_s=25.0), probe(6, 6.0e-3)))
        swept = CrossTierComparison(
            shot_probes=model.shot_probes,
            time_s=model.time_s,
            f0_force_n=model.f0_force_n,
            f0_sole_depth_m=model.f0_sole_depth_m,
            f0_speed_m_s=model.f0_speed_m_s,
            f0_divot_section_area_m2=model.f0_divot_section_area_m2,
            band=model.band,
            head_mass_kg=model.head_mass_kg,
            declared_width_m=model.declared_width_m,
            bulk_density_kg_m3=model.bulk_density_kg_m3,
            sweep_probes=(
                probe(
                    0, 0.0, speed_m_s=5.0, f0_inertial_share=0.52, f1_flux_share=0.68
                ),
                probe(
                    0, 0.0, speed_m_s=12.0, f0_inertial_share=0.93, f1_flux_share=0.69
                ),
            ),
        )
        crossover = swept.crossover()
        assert crossover is not None
        assert 5.0 < crossover.speed_m_s < 12.0


# ------------------------------------------------------------- the licence


class TestTheViewStatesWhatAgreementDoesAndDoesNotLicense:
    """The requirement of the issue, tested rather than assumed present."""

    def test_it_says_consistency_is_not_validation(self) -> None:
        text = licence_statement(speed_m_s=25.0)
        assert "not validation" in text.lower()
        assert "uncalibrated" in text.lower()

    def test_it_quotes_the_nasa_std_7009b_validation_level_from_the_assessment(
        self,
    ) -> None:
        text = licence_statement(speed_m_s=25.0)
        assert "0 of 4" in text
        assert "7009" in text

    def test_it_quotes_the_published_speed_ceiling_from_the_envelope(self) -> None:
        text = licence_statement(speed_m_s=25.0)
        assert f"{MAX_VALIDATED_SPEED_M_S:g}" in text
        assert "first sample" in text

    def test_it_says_the_magnitude_rests_on_a_declared_width(self) -> None:
        text = licence_statement(speed_m_s=25.0, effective_width_m=0.030)
        assert "30" in text
        assert "declared" in text.lower()

    def test_it_says_what_disagreement_does_license(self) -> None:
        """Falsification is the one thing the comparison can actually do."""
        text = licence_statement(speed_m_s=25.0)
        assert "falsif" in text.lower()

    def test_the_comparison_carries_it_and_a_short_form_for_a_stamp(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        assert "not validation" in model.licence().lower()
        stamp = model.licence_stamp()
        assert len(stamp) < len(model.licence())
        assert "not validation" in stamp.lower()

    def test_the_summary_carries_the_licence_and_every_agreement(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        text = model.summary()
        assert "not validation" in text.lower()
        for quantity in ComparedQuantity:
            assert quantity.label in text


class TestTheComparisonInheritsRatherThanImprovesTheVerdict:
    def test_the_worst_status_comes_from_the_f0_band(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        assert model.worst_status is EnvelopeStatus.BEYOND_VALIDATION

    def test_the_pair_of_tiers_is_named(self) -> None:
        model = comparison((probe(2, 2.0e-3), probe(6, 6.0e-3)))
        assert model.tiers == (FidelityTier.F0, FidelityTier.F1)
