"""What the designer actually reads (issue #8618).

The report layer is where honesty is either delivered or lost, so it is
tested as carefully as the arithmetic: the verdict must lead, a refusal must
never sit next to a number, and an empty cell of a map must not look like an
unloaded one.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.solvers import EnvelopeStatus
from src.tools.bunker_shot_gui.model import ShotOutcome
from src.tools.bunker_shot_gui.report import (
    SHADE_RAMP,
    comparison_report,
    evaluation_report,
    playability_text,
    shade_grid,
    shot_report,
    sole_map_text,
    status_colour,
    status_headline,
    verdict_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class TestStatusPresentation:
    @pytest.mark.parametrize("status", list(EnvelopeStatus))
    def test_every_status_has_a_headline_and_a_colour(
        self, status: EnvelopeStatus
    ) -> None:
        assert status_headline(status)
        assert status_colour(status).startswith("#")

    def test_refusal_is_the_only_red(self) -> None:
        reds = {
            status
            for status in EnvelopeStatus
            if status_colour(status) == status_colour(EnvelopeStatus.REFUSED)
        }
        assert reds == {EnvelopeStatus.REFUSED}

    def test_the_beyond_validation_headline_says_it_is_an_extrapolation(self) -> None:
        headline = status_headline(EnvelopeStatus.BEYOND_VALIDATION)
        assert "extrapolation" in headline
        assert "not a measurement" in headline

    def test_the_refusal_headline_says_no_number_is_reported(self) -> None:
        assert "No force" in status_headline(EnvelopeStatus.REFUSED)


class TestVerdictReport:
    def test_the_status_leads(self, nominal_shot) -> None:
        first = verdict_report(nominal_shot.verdict).splitlines()[0]
        assert first == status_headline(nominal_shot.status)

    def test_the_governing_scale_is_named(self, nominal_shot) -> None:
        assert "governing scale" in verdict_report(nominal_shot.verdict)

    def test_reasons_are_truncated_but_counted(self, nominal_shot) -> None:
        text = verdict_report(nominal_shot.verdict, max_reasons=2)
        assert text.count("reason:") == 2
        assert "further finding" in text

    def test_caveats_are_never_truncated(self, nominal_shot) -> None:
        text = verdict_report(nominal_shot.verdict, max_reasons=0)
        assert text.count("caveat:") == len(nominal_shot.verdict.caveats)

    def test_the_borrowed_coefficient_caveat_is_always_stated(
        self, nominal_shot
    ) -> None:
        assert "borrowed from a published analogue" in verdict_report(
            nominal_shot.verdict
        )

    def test_negative_truncation_is_refused(self, nominal_shot) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            verdict_report(nominal_shot.verdict, max_reasons=-1)


class TestShotReport:
    def test_the_verdict_precedes_every_number(self, nominal_shot) -> None:
        text = shot_report(nominal_shot)
        assert text.index(status_headline(nominal_shot.status)) < text.index(
            "Peak resultant force"
        )

    def test_the_numbers_are_present_for_an_answerable_shot(self, nominal_shot) -> None:
        text = shot_report(nominal_shot)
        for label in ("Peak resultant force", "Maximum depth", "Carry"):
            assert label in text

    def test_a_refusal_reports_no_numbers_at_all(self, nominal_shot) -> None:
        refused = ShotOutcome(
            verdict=nominal_shot.verdict,
            fidelity_tier=nominal_shot.fidelity_tier,
            refused=True,
            delivered=nominal_shot.delivered,
        )
        text = shot_report(refused)
        assert "No numbers are reported" in text
        assert "Peak resultant force" not in text
        assert "Carry" not in text

    def test_delivered_geometry_names_its_bounce_convention(self, nominal_shot) -> None:
        assert "Effective bounce (marketed)" in shot_report(nominal_shot)


class TestShadeGrid:
    def test_an_empty_cell_is_not_an_unloaded_cell(self) -> None:
        rows = shade_grid(np.array([[np.nan, 0.0]]))
        assert rows[0][0] == " "
        assert rows[0][1] != " "

    def test_the_peak_takes_the_darkest_shade(self) -> None:
        rows = shade_grid(np.array([[0.0, 1.0]]))
        assert rows[0][1] == SHADE_RAMP[-1]

    def test_an_all_nan_grid_renders_blank(self) -> None:
        rows = shade_grid(np.full((2, 3), np.nan))
        assert rows == ("   ", "   ")

    def test_an_all_zero_grid_does_not_divide_by_zero(self) -> None:
        rows = shade_grid(np.zeros((2, 2)))
        assert all(len(row) == 2 for row in rows)

    def test_shape_is_preserved(self) -> None:
        rows = shade_grid(np.arange(12.0).reshape(3, 4))
        assert len(rows) == 3
        assert all(len(row) == 4 for row in rows)

    def test_a_one_dimensional_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2-D array"):
            shade_grid(np.zeros(4))


class TestMapAndWindowText:
    def test_the_sole_map_says_where_to_grind(self, nominal_shot) -> None:
        text = sole_map_text(nominal_shot.sole_load)
        assert "Removable for free" in text
        assert "Centre of pressure" in text

    def test_the_sole_map_labels_its_axes(self, nominal_shot) -> None:
        assert "heel -> toe" in sole_map_text(nominal_shot.sole_load)

    def test_the_playability_text_states_the_acceptance_band(
        self, nominal_evaluation
    ) -> None:
        text = playability_text(nominal_evaluation.playability)
        assert "Acceptance band" in text
        assert "Nominal delivery inside" in text

    def test_an_unmeasured_window_says_why(
        self, model, nominal_design, firm_sand, tour_swing
    ) -> None:
        evaluation = model.evaluate(
            nominal_design, firm_sand, tour_swing, include_playability=False
        )
        text = playability_text(evaluation.playability)
        assert "not measured" in text
        assert "Window area" not in text


class TestEvaluationAndComparisonReports:
    def test_the_evaluation_report_states_both_bounce_conventions(
        self, nominal_evaluation
    ) -> None:
        text = evaluation_report(nominal_evaluation)
        assert "Marketed bounce" in text
        assert "Geometric bounce (patent)" in text

    def test_the_evaluation_report_names_the_sand(self, nominal_evaluation) -> None:
        assert "kg/cm^2" in evaluation_report(nominal_evaluation)

    def test_the_comparison_names_both_designs(
        self, model, firm_sand, tour_swing
    ) -> None:
        from src.tools.bunker_shot_gui.design import WedgeDesign

        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        text = comparison_report(comparison)
        assert "left" in text
        assert "right" in text
        assert "Verdict" in text

    def test_the_comparison_no_longer_headlines_a_leader(
        self, model, firm_sand, tour_swing
    ) -> None:
        """Issue #9243: a leader was named whether or not the bands separated.

        The headline is now the verdict, which is allowed to be a tie, and the
        word "Leader" is gone from this report so that a reader skimming for
        one cannot find a design name that the uncertainty does not support.
        """
        from src.tools.bunker_shot_gui.design import WedgeDesign

        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        text = comparison_report(comparison)
        assert "Leader" not in text
        assert ("INDISTINGUISHABLE" in text) == (comparison.winner is None)

    def test_the_comparison_reports_the_uncertainty_budget(
        self, model, firm_sand, tour_swing
    ) -> None:
        """The split, the dominant term and the unsized terms all appear."""
        from src.tools.bunker_shot_gui.design import WedgeDesign

        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        text = comparison_report(comparison)
        assert "Uncertainty budget" in text
        assert "model-form" in text
        assert "sampling" in text
        assert "Dominant term" in text
        assert "UNQUANTIFIED" in text
        assert "NOT a confidence interval" in text

    def test_the_bootstrap_interval_is_qualified(
        self, model, firm_sand, tour_swing
    ) -> None:
        """It covers the delivery sweep only, and the report has to say so."""
        from src.tools.bunker_shot_gui.design import WedgeDesign

        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        text = comparison_report(comparison)
        assert "NOT the whole uncertainty" in text
        assert "#8659" in text

    def test_an_overlapping_comparison_says_it_does_not_separate(
        self, model, firm_sand, tour_swing
    ) -> None:
        from src.tools.bunker_shot_gui.design import WedgeDesign

        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        text = comparison_report(comparison)
        expected = "is better" if comparison.separated else "not ordered"
        assert expected in text

    def test_an_unrankable_comparison_says_why(
        self, model, firm_sand, quasi_static_swing
    ) -> None:
        from src.tools.bunker_shot_gui.design import WedgeDesign

        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            quasi_static_swing,
        )
        text = comparison_report(comparison)
        assert "not available" in text
        assert "Uncertainty budget" not in text
        assert "is better" not in text
