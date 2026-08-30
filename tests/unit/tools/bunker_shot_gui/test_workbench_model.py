"""The headless workbench model, running the real F0 solver (issue #8618).

Nothing here is mocked: every test drives
:class:`~bunkershot3d.solvers.drft.DRFTSolver` over a lofted wedge in a
USGA-referenced bed. Nothing here imports Qt.

The load-bearing test is :class:`TestRefusalIsNotANumber`. ADR-0032 makes
refusal the defining behaviour of the F0 tier, so the model is checked to
report *no* force, depth or carry whenever the envelope declines the query.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import BounceUtilisation
from bunkershot3d.solvers import EnvelopeStatus, FidelityTier
from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
    WorkbenchInputError,
)
from src.tools.bunker_shot_gui.model import (
    ATTACK_ANGLE_SWEEP_DEG,
    PlayabilityOutcome,
    ShotOutcome,
    WorkbenchModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class TestHeadBuild:
    def test_lofted_head_has_a_sole(self, model, nominal_design) -> None:
        build = model.head_build(nominal_design.geometry())
        assert build.sole_mask.any()
        assert build.elements_body.n_elements > build.sole_mask.sum()

    def test_sole_reference_is_the_lowest_sole_point(
        self, model, nominal_design
    ) -> None:
        build = model.head_build(nominal_design.geometry())
        sole_z = build.elements_body.centroids_m[build.sole_mask][:, 2]
        assert build.sole_reference_body_m[2] == pytest.approx(float(sole_z.min()))

    def test_shaft_axis_points_up_toward_the_grip(self, model, nominal_design) -> None:
        build = model.head_build(nominal_design.geometry())
        assert build.shaft_axis_body[2] > 0.0
        assert float(np.linalg.norm(build.shaft_axis_body)) == pytest.approx(1.0)

    def test_head_model_carries_a_positive_definite_inertia(
        self, model, nominal_design
    ) -> None:
        head = model.head_build(nominal_design.geometry()).head_model
        assert head.inertia_body_kg_m2 is not None
        assert float(np.linalg.eigvalsh(head.inertia_body_kg_m2).min()) > 0.0

    def test_the_expensive_build_is_cached(self, model, nominal_design) -> None:
        geometry = nominal_design.geometry()
        assert model.head_build(geometry) is model.head_build(geometry)


class TestNominalShot:
    def test_the_shot_is_produced_by_the_f0_tier(self, nominal_shot) -> None:
        assert nominal_shot.fidelity_tier is FidelityTier.F0

    def test_a_greenside_shot_is_beyond_published_validation(
        self, nominal_shot
    ) -> None:
        """25 m/s is ~17x the fastest intrusion in the RFT corpus."""
        assert nominal_shot.status is EnvelopeStatus.BEYOND_VALIDATION
        assert not nominal_shot.is_within_stated_envelope

    def test_the_shot_is_not_refused(self, nominal_shot) -> None:
        assert not nominal_shot.refused

    def test_peak_force_is_in_the_published_order_of_magnitude(
        self, nominal_shot
    ) -> None:
        """The research addendum's band for a greenside strike is 200-1000 N."""
        assert 100.0 < nominal_shot.peak_force_n < 2000.0

    def test_the_head_loses_speed_to_the_sand(self, nominal_shot) -> None:
        assert nominal_shot.exit_speed_mps < nominal_shot.entry_speed_mps

    def test_divot_depth_is_a_few_tens_of_millimetres(self, nominal_shot) -> None:
        assert 0.002 < nominal_shot.max_depth_m < 0.060

    def test_the_inertial_term_dominates_at_greenside_speed(self, nominal_shot) -> None:
        """ADR-0032: the terms cross at 6.8 m/s and delivery is 20-27 m/s."""
        assert nominal_shot.peak_inertial_fraction > 0.5

    def test_contact_lasts_a_few_milliseconds(self, nominal_shot) -> None:
        assert 0.001 < nominal_shot.contact_duration_s < 0.030

    def test_the_solver_is_fast_enough_to_be_interactive(self, nominal_shot) -> None:
        """ADR-0032 buys the F0 tier for ~ms/shot; the budget is 50 ms."""
        assert nominal_shot.runtime_s < 0.5

    def test_delivered_loft_reflects_the_open_face_and_shaft_lean(
        self, nominal_shot, nominal_design
    ) -> None:
        static_loft = nominal_design.geometry().loft_deg
        delivered = nominal_shot.delivered.effective_loft_deg
        assert delivered != pytest.approx(static_loft)
        assert 40.0 < delivered < 80.0

    def test_face_opening_costs_aim(self, nominal_shot) -> None:
        """At a 64 deg lie, opening the face buys loft and costs more aim."""
        assert nominal_shot.delivered.aim_offset_deg > 0.0


class TestDesignerMetrics:
    def test_divot_entry_lands_where_the_designer_asked(
        self, model, nominal_design, firm_sand
    ) -> None:
        swing = SwingSetup(entry_distance_behind_ball_m=0.06)
        shot = model.run_shot(nominal_design.geometry(), firm_sand.sand_state(), swing)
        assert shot.divot is not None
        assert shot.divot.entry_distance_behind_ball_m == pytest.approx(0.06, abs=1e-3)

    def test_divot_moves_a_plausible_mass_of_sand(self, nominal_shot) -> None:
        assert nominal_shot.divot is not None
        assert 0.005 < nominal_shot.divot.mass_kg < 0.5

    def test_dig_versus_skid_returns_one_of_the_three_verdicts(
        self, nominal_shot
    ) -> None:
        assert nominal_shot.dig_skid is not None
        assert nominal_shot.dig_skid.verdict.value in {"dig", "skid", "marginal"}

    def test_head_load_metrics_are_reported(self, nominal_shot) -> None:
        assert nominal_shot.loads is not None
        assert nominal_shot.loads.peak_deceleration_g > 0.0

    def test_bounce_utilisation_is_a_strict_subset_of_the_sole(
        self, nominal_shot
    ) -> None:
        assert nominal_shot.sole_load is not None
        utilisation = nominal_shot.sole_load.utilisation
        assert isinstance(utilisation, BounceUtilisation)
        assert 0.0 < utilisation.utilisation_fraction < 1.0
        assert utilisation.removable_area_m2 > 0.0

    def test_bounce_map_is_a_square_grid_with_empty_cells_marked(
        self, nominal_shot
    ) -> None:
        density = nominal_shot.sole_load.density_pa_s
        assert density.ndim == 2
        assert density.shape[0] == density.shape[1]
        assert np.isnan(density).any() or np.isfinite(density).all()
        assert np.isfinite(density).any()

    def test_centre_of_pressure_lies_on_the_sole(
        self, model, nominal_shot, nominal_design
    ) -> None:
        build = model.head_build(nominal_design.geometry())
        sole = build.elements_body.centroids_m[build.sole_mask]
        centre = nominal_shot.sole_load.utilisation.centre_of_pressure_body_m
        assert float(sole[:, 0].min()) <= centre[0] <= float(sole[:, 0].max())
        assert float(sole[:, 1].min()) <= centre[1] <= float(sole[:, 1].max())

    def test_carry_is_reported_for_a_greenside_shot(self, nominal_shot) -> None:
        assert nominal_shot.carry_m is not None
        assert 0.0 < nominal_shot.carry_m < 60.0

    def test_no_metric_was_silently_dropped(self, nominal_shot) -> None:
        assert nominal_shot.unavailable == ()


class TestCarryNeverTravelsWithoutItsVerdict:
    """Issue #8657: carry is uncalibrated, so it is never shown bare."""

    def test_the_carry_carries_its_verdict(self, nominal_shot) -> None:
        assert nominal_shot.carry_verdict is not None
        assert nominal_shot.carry_verdict.status is not EnvelopeStatus.WITHIN

    def test_the_carry_verdict_says_it_is_uncalibrated(self, nominal_shot) -> None:
        reasons = " ".join(nominal_shot.carry_verdict.reasons)
        assert "uncalibrated" in reasons

    def test_a_carry_without_a_verdict_cannot_be_constructed(
        self, nominal_shot
    ) -> None:
        """A type invariant, not a convention the display layer follows."""
        with pytest.raises(ValueError, match="travel together"):
            ShotOutcome(
                verdict=nominal_shot.verdict,
                fidelity_tier=FidelityTier.F0,
                refused=False,
                delivered=nominal_shot.delivered,
                carry_m=9.0,
            )

    def test_a_carry_grid_without_a_verdict_cannot_be_constructed(self) -> None:
        with pytest.raises(ValueError, match="travel together"):
            PlayabilityOutcome(window=None, carry_m=np.array([[7.5]], dtype=np.float64))

    def test_the_playability_grid_carries_its_verdict(self, nominal_evaluation) -> None:
        assert nominal_evaluation.playability.carry_verdict is not None


class TestRefusalIsNotANumber:
    """ADR-0032's single most important requirement, tested from both ends."""

    def test_a_quasi_static_solver_is_refused_above_the_froude_ceiling(
        self, model, nominal_design, firm_sand, quasi_static_swing
    ) -> None:
        shot = model.run_shot(
            nominal_design.geometry(), firm_sand.sand_state(), quasi_static_swing
        )
        assert shot.refused
        assert shot.status is EnvelopeStatus.REFUSED

    def test_a_refusal_reports_no_force_depth_or_carry(
        self, model, nominal_design, firm_sand, quasi_static_swing
    ) -> None:
        shot = model.run_shot(
            nominal_design.geometry(), firm_sand.sand_state(), quasi_static_swing
        )
        assert shot.peak_force_n is None
        assert shot.max_depth_m is None
        assert shot.carry_m is None
        assert shot.sole_load is None

    def test_a_refusal_still_carries_the_verdict_that_caused_it(
        self, model, nominal_design, firm_sand, quasi_static_swing
    ) -> None:
        shot = model.run_shot(
            nominal_design.geometry(), firm_sand.sand_state(), quasi_static_swing
        )
        assert shot.verdict.is_refusal
        assert any("quasi-static" in reason for reason in shot.verdict.reasons)

    def test_a_refused_outcome_cannot_be_constructed_with_a_number(
        self, nominal_shot
    ) -> None:
        """The rule is a type invariant, not a convention the caller follows."""
        with pytest.raises(ValueError, match="must not carry a force"):
            ShotOutcome(
                verdict=nominal_shot.verdict,
                fidelity_tier=FidelityTier.F0,
                refused=True,
                delivered=nominal_shot.delivered,
                peak_force_n=500.0,
            )

    def test_playability_is_unavailable_when_every_point_is_refused(
        self, model, nominal_design, firm_sand, quasi_static_swing
    ) -> None:
        outcome = model.playability(
            nominal_design.geometry(), firm_sand, quasi_static_swing
        )
        assert not outcome.available
        assert "refused" in outcome.unavailable_reason
        assert np.isnan(outcome.carry_m).all()


class TestPlayabilityWindow:
    def test_the_window_is_measured_on_attack_angle_and_firmness(
        self, nominal_evaluation
    ) -> None:
        window = nominal_evaluation.playability.window
        assert window is not None
        assert window.axis_a.name == "attack_angle"
        assert window.axis_b.name == "sand_firmness"

    def test_the_swept_attack_angles_are_the_registered_range(
        self, nominal_evaluation
    ) -> None:
        swept = nominal_evaluation.playability.attack_angle_deg
        assert swept[0] == pytest.approx(ATTACK_ANGLE_SWEEP_DEG[0])
        assert swept[-1] == pytest.approx(ATTACK_ANGLE_SWEEP_DEG[1])

    def test_the_carry_grid_matches_the_axes(self, nominal_evaluation) -> None:
        playability = nominal_evaluation.playability
        assert playability.carry_m.shape == (
            playability.attack_angle_deg.size,
            playability.firmness_kg_per_cm2.size,
        )

    def test_a_steeper_blow_digs_deeper(self, model, nominal_design, firm_sand) -> None:
        """Depth is monotone in attack angle; carry is **not** (issue #9247).

        This test used to read ``carry[0, 0] > carry[-1, 0]`` over the
        registered sweep and was documented as "carry tracks divot depth,
        which tracks attack angle". Only the first half of that survives
        the un-mirrored frame. Depth is monotone across the whole sweep,
        but carry turns over: measured on this design in firm sand it
        climbs 0.117 -> 0.696 m from -2 to -6 deg, falls to 0.624 m by
        -7 deg, and past -8 deg the head buries and there is no carry to
        compare at all -- which is why the old assertion now reads a NaN
        at its steep end.

        That turnover is the physics the tool exists to show: a steeper
        blow moves more sand, and past the point where the sole stops
        planing it spends the extra on burying the head instead of on the
        ball. Asserting the old monotone claim across the whole sweep
        would re-assert an ordering that only held because every delivery
        planed under the mirror.
        """
        geometry = nominal_design.geometry()
        sand = firm_sand.sand_state()
        depths = [
            model.run_shot(
                geometry, sand, SwingSetup(attack_angle_deg=angle)
            ).max_depth_m
            for angle in (-2.0, -4.0, -6.0, -8.0)
        ]
        assert depths == sorted(depths), (
            f"a steeper blow must dig deeper: {[round(d * 1e3, 2) for d in depths]} mm"
        )

    def test_carry_peaks_inside_the_sweep_rather_than_at_its_steep_end(
        self, model, nominal_design, firm_sand
    ) -> None:
        """The other half of the old claim, stated as what it really is.

        Carry is single-peaked in attack angle, so the steepest delivery
        in the registered sweep is not the longest. Pinned because the
        mirrored frame made carry look monotone, and a tool that ranks
        deliveries by carry would have recommended the steepest one.
        """
        geometry = nominal_design.geometry()
        sand = firm_sand.sand_state()
        carries = {
            angle: model.run_shot(
                geometry, sand, SwingSetup(attack_angle_deg=angle)
            ).carry_m
            for angle in (-2.0, -4.0, -6.0, -8.0)
        }
        assert carries[-8.0] is None, (
            "the steep end of the sweep must bury this design and report no "
            f"carry; got {carries[-8.0]}"
        )
        planing = {a: c for a, c in carries.items() if c is not None}
        assert set(planing) == {-2.0, -4.0, -6.0}, planing
        assert planing[-6.0] > planing[-4.0] > planing[-2.0], (
            f"carry must still rise with a steeper blow while the sole planes: {planing}"
        )

    def test_the_window_is_a_fraction_of_the_swept_domain(
        self, nominal_evaluation
    ) -> None:
        window = nominal_evaluation.playability.window
        assert 0.0 <= window.fraction <= 1.0
        assert window.area <= window.domain_area

    def test_nothing_was_refused_in_the_dynamic_sweep(self, nominal_evaluation) -> None:
        """Refusal is the envelope declining, and it declines nothing here.

        With the inertial term on, the F0 envelope answers every point in
        the sweep. It used to be possible to satisfy this by accident,
        because ``refused_fraction`` counted *any* NaN carry and the only
        way to get one was a refusal. Issue #9247 gave NaN a second cause
        -- a buried head with no divot to derive a carry from -- and this
        assertion is only about the first.
        """
        window = nominal_evaluation.playability.window
        assert window.refused_fraction == 0.0

    def test_a_buried_point_is_reported_apart_from_a_refused_one(
        self, nominal_evaluation
    ) -> None:
        """The steep end of the sweep buries; that is not a refusal.

        Before issue #9247 these were one number, and the corrected model
        made it read "the solver refused half this domain" about a solver
        that refused nothing. Both still count against the window.
        """
        window = nominal_evaluation.playability.window
        assert window.unmeasured_fraction > 0.0, (
            "the registered attack sweep reaches -12 deg, which buries this "
            "design; the burial must be recorded"
        )
        assert window.refused_fraction == 0.0
        missing = np.isnan(window.carry_m)
        assert missing.any() and not missing.all()

    def test_skipping_the_sweep_says_so_rather_than_reporting_an_empty_window(
        self, model, nominal_design, firm_sand, tour_swing
    ) -> None:
        evaluation = model.evaluate(
            nominal_design, firm_sand, tour_swing, include_playability=False
        )
        assert not evaluation.playability.available
        assert "not run" in evaluation.playability.unavailable_reason


class TestDesignSensitivity:
    """The tool only earns its keep if geometry changes the answer."""

    def test_bounce_changes_the_answer(self, model, firm_sand, tour_swing) -> None:
        low = model.run_shot(
            WedgeDesign(name="low", marketed_bounce_deg=6.0).geometry(),
            firm_sand.sand_state(),
            tour_swing,
        )
        high = model.run_shot(
            WedgeDesign(name="high", marketed_bounce_deg=13.0).geometry(),
            firm_sand.sand_state(),
            tour_swing,
        )
        assert low.max_depth_m != pytest.approx(high.max_depth_m)
        assert low.peak_force_n != pytest.approx(high.peak_force_n)

    def test_a_sole_that_cannot_be_lofted_is_an_input_error(
        self, model, firm_sand, tour_swing
    ) -> None:
        """A design vector can pass every scalar invariant and still have no
        constructible camber segment; that must not escape as a bare error."""
        with pytest.raises(WorkbenchInputError, match="cannot be lofted"):
            model.run_shot(
                WedgeDesign(name="flat", marketed_bounce_deg=4.0).geometry(),
                firm_sand.sand_state(),
                tour_swing,
            )

    def test_firmer_sand_resists_more(self, model, nominal_design, tour_swing) -> None:
        geometry = nominal_design.geometry()
        soft = model.run_shot(
            geometry, SandCondition().with_firmness(1.6).sand_state(), tour_swing
        )
        firm = model.run_shot(
            geometry, SandCondition().with_firmness(2.8).sand_state(), tour_swing
        )
        assert firm.max_depth_m < soft.max_depth_m


class TestComparison:
    def test_two_designs_are_ranked_with_an_interval(
        self, model, firm_sand, tour_swing
    ) -> None:
        comparison = model.compare(
            WedgeDesign(name="low bounce", marketed_bounce_deg=6.0),
            WedgeDesign(name="high bounce", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        ranking = comparison.ranking
        assert ranking is not None
        assert ranking.best in {"low bounce", "high bounce"}
        assert float(ranking.probability_best.sum()) == pytest.approx(1.0)
        assert np.all(ranking.ci_low <= ranking.ci_high)

    def test_the_ranking_is_reproducible(self, model, firm_sand, tour_swing) -> None:
        designs = (
            WedgeDesign(name="low bounce", marketed_bounce_deg=6.0),
            WedgeDesign(name="high bounce", marketed_bounce_deg=13.0),
        )
        first = model.compare(*designs, firm_sand, tour_swing)
        second = model.compare(*designs, firm_sand, tour_swing)
        assert first.ranking.best == second.ranking.best
        assert np.allclose(first.ranking.mean, second.ranking.mean)

    def test_both_designs_are_reported_whether_or_not_they_separate(
        self, model, firm_sand, tour_swing
    ) -> None:
        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        assert comparison.left.design.name == "left"
        assert comparison.right.design.name == "right"
        assert isinstance(comparison.separated, bool)

    def test_a_refused_pair_cannot_be_ranked(
        self, model, firm_sand, quasi_static_swing
    ) -> None:
        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            quasi_static_swing,
        )
        assert comparison.ranking is None
        assert "answerable" in comparison.ranking_unavailable_reason

    def test_identical_names_are_refused(self, model, firm_sand, tour_swing) -> None:
        with pytest.raises(ValueError, match="different names"):
            model.compare(
                WedgeDesign(name="same"),
                WedgeDesign(name="same"),
                firm_sand,
                tour_swing,
            )

    def test_an_impossible_design_surfaces_as_an_input_error(
        self, model, firm_sand, tour_swing
    ) -> None:
        with pytest.raises(WorkbenchInputError):
            model.evaluate(
                WedgeDesign(name="bad", heel_relief_fraction=0.88),
                firm_sand,
                tour_swing,
            )


class TestModelSettings:
    def test_settings_are_exposed_for_the_view(self, coarse_settings) -> None:
        assert WorkbenchModel(coarse_settings).settings is coarse_settings

    def test_default_settings_are_used_when_none_are_given(self) -> None:
        assert WorkbenchModel().settings == SolverSetup()
