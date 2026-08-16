"""Reporting the load field and the contact patch in words (#8705, #8707).

The animated view answers "which part of this grind is doing the work, and
when" by eye. The same numbers have to be quotable, because a figure cannot be
pasted into an issue and a screenshot cannot be diffed. This module pins the
text.

It also pins the caveat #8707 asks to be carried: the bounce trend the patch
view illustrates is confounded by the camber substitution of #8698. Where the
declared camber was not constructible, a comparison labelled "bounce" is
really a bounce-and-camber comparison, and the report has to say so next to
the patch numbers rather than only next to the geometry.
"""

from __future__ import annotations

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
        if nominal_evaluation.camber_was_clamped:
            assert PATCH_CONFOUND_CAVEAT in report
        else:
            assert PATCH_CONFOUND_CAVEAT not in report

    def test_a_design_whose_camber_is_substituted_says_so_by_the_patch(
        self, model: WorkbenchModel, firm_sand: SandCondition, tour_swing: SwingSetup
    ) -> None:
        wide = WedgeDesign(name="clamped", camber_area_mm2=70.0)
        evaluation = model.evaluate(
            wide, firm_sand, tour_swing, include_playability=False
        )
        if not evaluation.camber_was_clamped:
            pytest.skip("this geometry realised its declared camber")
        assert PATCH_CONFOUND_CAVEAT in evaluation_report(evaluation)

    def test_the_caveat_names_the_issue_it_comes_from(self) -> None:
        assert "8698" in PATCH_CONFOUND_CAVEAT


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
