"""USGA specification compliance tests (issue #8610)."""

from __future__ import annotations

import pytest
from bunkershot3d.sand.presets import (
    USGA_GSR_2020_WINDY_PSD,
    USGA_LAB_MIDBAND_PSD,
)
from bunkershot3d.sand.psd import ParticleSizeDistribution
from bunkershot3d.sand.specification import (
    USGA_GSR_2020_SPECIFICATION,
    USGA_LAB_SPECIFICATION,
    ComplianceReport,
    evaluate_compliance,
)

pytestmark = pytest.mark.unit


class TestSpecificationTables:
    def test_lab_specification_carries_its_citation(self) -> None:
        spec = USGA_LAB_SPECIFICATION
        assert "F1632" in spec.citation or "Turf" in spec.citation
        assert spec.uniformity_coefficient_range == (2.0, 5.0)
        names = [band.name for band in spec.bands]
        assert "coarse + medium" in names
        assert "silt + clay" in names

    def test_gsr_specification_is_the_tighter_by_volume_table(self) -> None:
        spec = USGA_GSR_2020_SPECIFICATION
        assert "58" in spec.citation and "2020" in spec.citation
        coarse_medium = next(
            band for band in spec.bands if band.name == "coarse + medium"
        )
        assert coarse_medium.min_fraction == pytest.approx(0.65)


class TestMidbandCompliance:
    def test_midband_passes_both_specifications(self) -> None:
        for spec in (USGA_LAB_SPECIFICATION, USGA_GSR_2020_SPECIFICATION):
            report = evaluate_compliance(USGA_LAB_MIDBAND_PSD, spec)
            assert isinstance(report, ComplianceReport)
            assert report.passed, report.violations
            assert report.violations == ()

    def test_report_records_every_measured_band(self) -> None:
        report = evaluate_compliance(USGA_LAB_MIDBAND_PSD, USGA_LAB_SPECIFICATION)
        measured = dict(report.measurements)
        assert measured["coarse + medium"] == pytest.approx(0.815)
        assert measured["silt + clay"] == pytest.approx(0.015)
        assert report.uniformity_coefficient == pytest.approx(2.73, rel=1e-2)


class TestWindySiteException:
    def test_windy_sand_passes_the_lab_specification(self) -> None:
        report = evaluate_compliance(USGA_GSR_2020_WINDY_PSD, USGA_LAB_SPECIFICATION)
        assert report.passed, report.violations

    def test_windy_sand_deliberately_breaches_the_gsr_very_coarse_cap(self) -> None:
        """GSR 58(11) allows >80% in 0.25-1 mm with 10-20% at 1-2 mm on windy
        sites; that conflicts with its own <=7% very-coarse cap, so compliance
        is reported rather than enforced."""
        report = evaluate_compliance(
            USGA_GSR_2020_WINDY_PSD, USGA_GSR_2020_SPECIFICATION
        )
        assert not report.passed
        assert any("very coarse" in violation for violation in report.violations)


class TestViolationReporting:
    def test_out_of_band_sand_is_reported_not_raised(self) -> None:
        too_fine = ParticleSizeDistribution.from_bins(
            bin_edges_m=(2e-6, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3),
            bin_fractions=(0.20, 0.30, 0.30, 0.15, 0.04, 0.01, 0.0),
            name="too-fine",
        )
        report = evaluate_compliance(too_fine, USGA_LAB_SPECIFICATION)
        assert not report.passed
        assert len(report.violations) >= 2
        assert all(isinstance(v, str) for v in report.violations)

    def test_uniformity_coefficient_out_of_range_is_a_violation(self) -> None:
        very_uniform = ParticleSizeDistribution.from_bins(
            bin_edges_m=(2.5e-4, 5e-4, 1e-3),
            bin_fractions=(0.9, 0.1),
            name="single-size",
        )
        report = evaluate_compliance(very_uniform, USGA_LAB_SPECIFICATION)
        assert any("uniformity" in violation.lower() for violation in report.violations)
