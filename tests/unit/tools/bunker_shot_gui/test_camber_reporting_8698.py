"""The workbench must report the sole it built, not the one it was asked for.

Issue #8698: a wedge sole can only realise camber areas inside a band set by
its width and bounce, so the lofter fits a declared area that falls outside
it.  The workbench lofts with
:attr:`~bunkershot3d.geometry.CamberFit.NEAREST` on purpose -- a designer
dragging a bounce slider has to keep getting a head to look at -- which makes
*reporting* the substitution the workbench's obligation.
"""

from __future__ import annotations

import pytest

from bunkershot3d.geometry import constructible_camber_range_m2
from src.tools.bunker_shot_gui.report import evaluation_report

pytestmark = pytest.mark.unit


class TestTheEvaluationCarriesTheRealisedCamber:
    def test_the_effective_area_is_inside_the_constructible_band(
        self, nominal_evaluation
    ) -> None:
        low, high = constructible_camber_range_m2(
            nominal_evaluation.geometry, n_points=12
        )
        assert low <= nominal_evaluation.effective_camber_area_m2 <= high

    def test_clamping_is_reported_consistently(self, nominal_evaluation) -> None:
        declared = nominal_evaluation.geometry.sole_camber_area_m2
        effective = nominal_evaluation.effective_camber_area_m2
        assert nominal_evaluation.aggregate_camber_was_clamped == (
            effective != declared
        )

    def test_the_broad_flag_covers_the_stations_too(self, nominal_evaluation) -> None:
        """The workbench's flag carries the same invariant as the lofter's."""
        assert not (
            nominal_evaluation.clamped_camber_stations
            and not nominal_evaluation.any_camber_was_clamped
        )
        assert (
            nominal_evaluation.any_camber_was_clamped
            >= nominal_evaluation.aggregate_camber_was_clamped
        )


class TestTheReportStatesBothNumbers:
    def test_the_camber_line_is_present(self, nominal_evaluation) -> None:
        text = evaluation_report(nominal_evaluation)
        assert "Camber area" in text

    def test_a_substituted_camber_names_the_declared_value_too(
        self, nominal_evaluation
    ) -> None:
        text = evaluation_report(nominal_evaluation)
        camber_line = next(
            line for line in text.splitlines() if line.startswith("Camber area")
        )
        effective_mm2 = nominal_evaluation.effective_camber_area_m2 * 1e6
        assert f"{effective_mm2:.1f} mm^2" in camber_line
        if nominal_evaluation.aggregate_camber_was_clamped:
            declared_mm2 = nominal_evaluation.geometry.sole_camber_area_m2 * 1e6
            assert f"declared {declared_mm2:.1f} mm^2" in camber_line
            assert "not constructible" in camber_line
        else:
            assert "declared" not in camber_line

    def test_refitted_stations_are_named_on_the_camber_line(
        self, nominal_evaluation
    ) -> None:
        """An in-band declaration over refitted stations must not read clean.

        Without this the line renders a bare "42.0 mm^2" for a sole whose
        heel and toe carry something else -- the same silence at finer grain.
        """
        camber_line = next(
            line
            for line in evaluation_report(nominal_evaluation).splitlines()
            if line.startswith("Camber area")
        )
        clamped = nominal_evaluation.clamped_camber_stations
        if clamped:
            assert f"{len(clamped)} of " in camber_line
            assert "stations refitted" in camber_line
        else:
            assert "refitted" not in camber_line
