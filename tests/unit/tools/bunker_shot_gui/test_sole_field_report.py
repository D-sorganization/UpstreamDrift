"""Reporting the load field and the contact patch in words (#8705, #8707).

The animated view answers "which part of this grind is doing the work, and
when" by eye. The same numbers have to be quotable, because a figure cannot be
pasted into an issue and a screenshot cannot be diffed. This module pins the
text.

It also pins the caveat #8707 asks to be carried: the bounce trend the patch
view illustrates is confounded by the camber substitution of #8698. Where a
spanwise station could not carry the camber its relieved sole width implied,
a comparison labelled "bounce" is really a bounce-and-camber comparison, and
the report has to say so next to the patch numbers rather than only next to
the geometry.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.tools.bunker_shot_gui.design import SandCondition, SwingSetup, WedgeDesign
from src.tools.bunker_shot_gui.model import WorkbenchModel
from src.tools.bunker_shot_gui.report import (
    PATCH_CONFOUND_CAVEAT,
    evaluation_report,
    shot_report,
    sole_field_text,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="module")
def text(nominal_shot) -> str:  # type: ignore[no-untyped-def]
    """The field section of the nominal shot."""
    assert nominal_shot.sole_field is not None
    assert nominal_shot.contact_patch is not None
    return sole_field_text(nominal_shot.sole_field, nominal_shot.contact_patch)


class TestTheSectionLeadsWithItsValidity:
    def test_the_status_is_stated_in_the_section_itself(self, text: str) -> None:
        assert "BEYOND VALIDATION" in text

    def test_the_section_names_what_it_is(self, text: str) -> None:
        assert "Per-element sole load" in text


class TestTheTwoTermsAreReportedApart:
    def test_each_term_reports_a_peak_in_newtons(self, text: str) -> None:
        assert text.count(" N at ") >= 2

    def test_each_term_reports_when_it_peaked_in_milliseconds(self, text: str) -> None:
        assert "ms" in text

    def test_the_inertial_share_is_stated(self, text: str) -> None:
        assert "inertial share" in text.lower()

    def test_each_term_says_what_it_physically_is(self, text: str) -> None:
        """A designer reading "depth-dependent" should not have to guess."""
        from src.tools.bunker_shot_gui.field import LoadComponent

        for component in (LoadComponent.DEPTH, LoadComponent.INERTIAL):
            assert component.description in text

    def test_the_resolution_is_stated_so_the_bin_count_is_not_confused(
        self, text: str, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        assert str(nominal_shot.sole_field.n_elements) in text
        assert str(nominal_shot.sole_field.n_frames) in text


class TestTheContactPatchIsReported:
    def test_the_initial_patch_is_quoted_in_square_centimetres(self, text: str) -> None:
        assert "cm^2" in text
        assert "first contact" in text

    def test_the_peak_patch_is_quoted(self, text: str) -> None:
        assert "argest" in text  # "Largest contact patch"

    def test_the_leading_edge_gap_is_quoted_in_millimetres(self, text: str) -> None:
        assert "leading edge" in text
        assert "mm" in text


class TestTheConfoundIsCarried:
    """#8707: a bounce trend drawn over a substituted camber is not a bounce trend."""

    def test_a_clamped_camber_earns_the_caveat(self, nominal_evaluation) -> None:  # type: ignore[no-untyped-def]
        report = evaluation_report(nominal_evaluation)
        if nominal_evaluation.clamped_camber_stations:
            assert PATCH_CONFOUND_CAVEAT in report
        else:
            assert PATCH_CONFOUND_CAVEAT not in report

    def test_the_caveat_is_gated_on_stations_not_on_the_aggregate(
        self, nominal_evaluation
    ) -> None:  # type: ignore[no-untyped-def]
        """The point of the rebase, isolated from the lofting resolution.

        The shipped ``sm9_58_m`` preset declares an area its own sole width
        admits while still refitting stations along the sole, so a caveat
        gated on the declared-versus-effective aggregate is silent on the
        default design.  (The session fixture lofts at *coarse* test
        resolution, where the aggregate happens to clamp too, so the
        aggregate is neutralised here rather than relied upon; the shipped
        resolution is pinned in the geometry suite.)
        """
        assert nominal_evaluation.clamped_camber_stations, (
            "this test needs a design that refits stations"
        )

        # Force the aggregate flag False while leaving the stations refitted:
        # exactly the state the old gate misread.
        in_band = dataclasses.replace(
            nominal_evaluation,
            effective_camber_area_m2=nominal_evaluation.geometry.sole_camber_area_m2,
        )
        assert in_band.aggregate_camber_was_clamped is False
        assert in_band.clamped_camber_stations
        assert PATCH_CONFOUND_CAVEAT in evaluation_report(in_band)

    def test_an_unrefitted_design_does_not_carry_the_caveat(
        self, nominal_evaluation
    ) -> None:  # type: ignore[no-untyped-def]
        """The gate must not be trivially true, or the caveat means nothing."""
        clean = dataclasses.replace(
            nominal_evaluation,
            effective_camber_area_m2=nominal_evaluation.geometry.sole_camber_area_m2,
            camber_stations=(),
        )
        assert clean.any_camber_was_clamped is False
        assert PATCH_CONFOUND_CAVEAT not in evaluation_report(clean)

    def test_the_caveat_counts_the_stations_it_fired_on(
        self, nominal_evaluation
    ) -> None:  # type: ignore[no-untyped-def]
        report = evaluation_report(nominal_evaluation)
        clamped = nominal_evaluation.clamped_camber_stations
        if not clamped:
            pytest.skip("no station was refitted on this design")
        total = len(nominal_evaluation.camber_stations)
        assert f"({len(clamped)} of {total} spanwise stations refitted)" in report

    def test_a_design_whose_camber_is_substituted_says_so_by_the_patch(
        self, model: WorkbenchModel, firm_sand: SandCondition, tour_swing: SwingSetup
    ) -> None:
        wide = WedgeDesign(name="clamped", camber_area_mm2=70.0)
        evaluation = model.evaluate(
            wide, firm_sand, tour_swing, include_playability=False
        )
        if not evaluation.clamped_camber_stations:
            pytest.skip("this geometry realised its declared camber everywhere")
        assert PATCH_CONFOUND_CAVEAT in evaluation_report(evaluation)

    def test_the_caveat_names_the_issue_it_comes_from(self) -> None:
        assert "8698" in PATCH_CONFOUND_CAVEAT

    def test_the_caveat_is_stated_in_terms_of_stations(self) -> None:
        """It must describe what actually triggers it, not the aggregate.

        The previous wording claimed the *declared* area was inconstructible,
        which is false on the design the caveat fires for.
        """
        assert "station" in PATCH_CONFOUND_CAVEAT
        assert "declared camber area was not constructible" not in (
            PATCH_CONFOUND_CAVEAT
        )


class TestTheReportIsWiredIn:
    def test_the_shot_report_carries_the_field_section(self, nominal_shot) -> None:  # type: ignore[no-untyped-def]
        assert "Per-element sole load" in shot_report(nominal_shot)

    def test_the_evaluation_report_carries_it_too(self, nominal_evaluation) -> None:  # type: ignore[no-untyped-def]
        assert "Per-element sole load" in evaluation_report(nominal_evaluation)

    def test_a_refused_shot_reports_no_field_section(
        self,
        model: WorkbenchModel,
        nominal_design: WedgeDesign,
        firm_sand: SandCondition,
        quasi_static_swing: SwingSetup,
    ) -> None:
        refused = model.run_shot(
            nominal_design.geometry(), firm_sand.sand_state(), quasi_static_swing
        )
        assert "Per-element sole load" not in shot_report(refused)

    def test_the_summed_map_section_survives_alongside_it(
        self, nominal_evaluation
    ) -> None:  # type: ignore[no-untyped-def]
        report = evaluation_report(nominal_evaluation)
        assert "Bounce utilisation" in report
        assert "Per-element sole load" in report
