"""Tests for method citation metadata and cross-engine validation (Issue #777)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.dataclasses import (
    CITATION_CRUNCH_FACTOR,
    CITATION_KINEMATIC_SEQUENCE,
    CITATION_SPINAL_LOAD,
    CITATION_X_FACTOR,
    MethodCitation,
    validate_angle_cross_engine,
    validate_timing_cross_engine,
)

pytestmark = [pytest.mark.unit]


class TestMethodCitation:
    """MethodCitation dataclass tests."""

    def test_method_citations_frozen(self) -> None:
        """Citations should be immutable."""
        c = MethodCitation(name="test", authors="A", year=2000, title="T")
        try:
            c.name = "modified"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass  # Expected — frozen dataclass

    def test_predefined_citations_exist(self) -> None:
        """All four predefined citations should be populated."""
        for c in [
            CITATION_KINEMATIC_SEQUENCE,
            CITATION_X_FACTOR,
            CITATION_CRUNCH_FACTOR,
            CITATION_SPINAL_LOAD,
        ]:
            assert c.name
            assert c.authors
            assert c.year > 0
            assert c.title

    def test_optional_fields(self) -> None:
        """DOI and notes should be optional."""
        c = MethodCitation(name="N", authors="A", year=2000, title="T")
        assert c.doi is None
        assert c.notes is None

    def test_format_citation_with_doi(self) -> None:
        """Citation should format authors, year, title, and DOI."""
        formatted = CITATION_KINEMATIC_SEQUENCE.format_citation()
        assert "Putnam (1993)" in formatted
        assert "Sequential motions of body segments" in formatted
        assert "DOI: 10.1016/0021-9290(93)90084-R" in formatted

    def test_format_citation_without_doi(self) -> None:
        """Citation without DOI should omit DOI field."""
        formatted = CITATION_X_FACTOR.format_citation()
        assert "Cheetham et al. (2001)" in formatted
        assert "DOI" not in formatted

    def test_to_dict(self) -> None:
        """to_dict should include all dataclass fields."""
        data = CITATION_CRUNCH_FACTOR.to_dict()
        assert data["name"] == "Crunch Factor"
        assert data["authors"] == "McHardy & Pollard"
        assert data["year"] == 2005
        assert data["doi"] == "10.1136/bjsm.2004.014514"
        assert "notes" in data


class TestKinematicSequenceCitation:
    """Kinematic sequence result carries methodology citation."""

    def test_result_has_methodology(self) -> None:
        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )

        times = np.linspace(0, 1.0, 100)
        data = {
            "A": np.exp(-((times - 0.2) ** 2) / 0.01) * 10,
            "B": np.exp(-((times - 0.4) ** 2) / 0.01) * 20,
        }
        analyzer = SegmentTimingAnalyzer(expected_order=["A", "B"])
        result = analyzer.analyze(data, times)

        assert result.methodology is not None
        assert result.methodology.name == "Proximal-to-Distal Sequencing"
        assert result.methodology.authors == "Putnam"

    def test_plot_kinematic_sequence_renders_citation(self) -> None:
        """Plotting kinematic sequence should render citation when result carries methodology."""
        from matplotlib.figure import Figure

        from src.shared.python.biomechanics.kinematic_sequence import (
            SegmentTimingAnalyzer,
        )
        from src.shared.python.plotting.renderers._coordination_sequence import (
            CoordinationSequenceMixin,
        )

        class _MockData:
            def __init__(self, times: np.ndarray, velocities: np.ndarray) -> None:
                self._times = times
                self._velocities = velocities

            def get_series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
                return self._times, self._velocities

        class _MockCoord(CoordinationSequenceMixin):
            def __init__(self, times: np.ndarray, velocities: np.ndarray) -> None:
                self.data = _MockData(times, velocities)  # type: ignore[assignment]
                self.colors = {
                    "primary": "blue",
                    "secondary": "red",
                    "tertiary": "green",
                    "quaternary": "orange",
                    "quinary": "purple",
                }

        times = np.linspace(0, 1.0, 50)
        vel = np.zeros((50, 2))
        vel[:, 0] = np.sin(times * 3)
        vel[:, 1] = np.cos(times * 3)
        coord = _MockCoord(times, vel)

        analyzer = SegmentTimingAnalyzer(expected_order=["A", "B"])
        data = {
            "A": np.exp(-((times - 0.2) ** 2) / 0.01) * 10,
            "B": np.exp(-((times - 0.4) ** 2) / 0.01) * 20,
        }
        analyzer_result = analyzer.analyze(data, times)

        fig = Figure()
        coord.plot_kinematic_sequence(
            fig, {"A": 0, "B": 1}, analyzer_result=analyzer_result
        )
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        assert "Methodology: Putnam (1993)" in xlabel
        assert "10.1016/0021-9290(93)90084-R" in xlabel


class TestSpinalLoadCitation:
    """Spinal load dataclasses carry methodology citations."""

    def test_spinal_load_result_default_citation(self) -> None:
        from src.shared.python.injury.spinal_load_analysis import SpinalLoadResult

        result = SpinalLoadResult(time=np.array([0.0]))
        assert result.methodology is not None
        assert result.methodology.authors == "Hosea et al."

    def test_x_factor_metrics_default_citation(self) -> None:
        from src.shared.python.injury.spinal_load_analysis import XFactorMetrics

        m = XFactorMetrics(
            x_factor_angle=np.array([0.0]),
            x_factor_stretch=30.0,
            x_factor_stretch_time=0.5,
            separation_rate=100.0,
            transition_duration=0.3,
        )
        assert m.methodology.name == "X-Factor"

    def test_crunch_factor_metrics_default_citation(self) -> None:
        from src.shared.python.injury.spinal_load_analysis import CrunchFactorMetrics

        m = CrunchFactorMetrics(
            lateral_bend_angle=np.array([0.0]),
            rotation_angle=np.array([0.0]),
            crunch_factor=np.array([0.0]),
            peak_crunch=10.0,
            peak_crunch_time=0.4,
            asymmetry_ratio=1.0,
        )
        assert m.methodology.name == "Crunch Factor"

    def test_spinal_load_result_get_and_format_citations(self) -> None:
        """SpinalLoadResult extracts and formats all component citations."""
        from src.shared.python.injury.spinal_load_analysis import (
            CrunchFactorMetrics,
            SpinalLoadResult,
            XFactorMetrics,
        )

        x_m = XFactorMetrics(
            x_factor_angle=np.array([0.0]),
            x_factor_stretch=30.0,
            x_factor_stretch_time=0.5,
            separation_rate=100.0,
            transition_duration=0.3,
        )
        c_m = CrunchFactorMetrics(
            lateral_bend_angle=np.array([0.0]),
            rotation_angle=np.array([0.0]),
            crunch_factor=np.array([0.0]),
            peak_crunch=10.0,
            peak_crunch_time=0.4,
            asymmetry_ratio=1.0,
        )
        result = SpinalLoadResult(time=np.array([0.0]), x_factor=x_m, crunch_factor=c_m)
        citations = result.get_citations()
        assert len(citations) == 3
        names = [c.name for c in citations]
        assert "Spinal Load Analysis" in names
        assert "X-Factor" in names
        assert "Crunch Factor" in names

        formatted = result.format_citations()
        assert len(formatted) == 3
        assert any("Hosea et al. (1990)" in f for f in formatted)
        assert any("Cheetham et al. (2001)" in f for f in formatted)
        assert any("McHardy & Pollard (2005)" in f for f in formatted)
        assert any("DOI: 10.1136/bjsm.2004.014514" in f for f in formatted)


class TestAnalysisServiceCitation:
    """Analysis service surfaces methodology in API outputs."""

    def test_populate_kinematic_sequence_includes_methodology(self) -> None:
        """API response must include methodology dict and citation string."""
        from unittest.mock import MagicMock
        from src.api.services.analysis_service import AnalysisService

        engine_manager = MagicMock()
        service = AnalysisService(engine_manager)
        result: dict[str, object] = {"metadata": {}}
        mock_request = MagicMock()
        mock_request.trajectories = {}

        times = np.linspace(0, 1.0, 50)
        segment_velocities = {
            "pelvis": np.exp(-((times - 0.2) ** 2) / 0.01) * 10,
            "torso": np.exp(-((times - 0.4) ** 2) / 0.01) * 20,
            "arm": np.exp(-((times - 0.6) ** 2) / 0.01) * 30,
            "club": np.exp(-((times - 0.8) ** 2) / 0.01) * 40,
        }
        service._populate_kinematic_sequence(
            result,
            mock_request,
            segment_velocities,
            request_data={"times": times.tolist()},
        )
        assert "kinematic_sequence" in result
        seq = result["kinematic_sequence"]
        assert isinstance(seq, dict)
        assert "methodology" in seq
        assert seq["methodology"]["name"] == "Proximal-to-Distal Sequencing"
        assert seq["methodology"]["authors"] == "Putnam"
        assert seq["methodology"]["doi"] == "10.1016/0021-9290(93)90084-R"
        assert "citation" in seq
        assert "Putnam (1993)" in seq["citation"]
        assert "DOI: 10.1016/0021-9290(93)90084-R" in seq["citation"]


class TestTimingValidation:
    """Cross-engine timing validation tests."""

    def test_identical_timings_pass(self) -> None:
        t = np.array([0.1, 0.2, 0.3, 0.4])
        result = validate_timing_cross_engine(t, t)
        assert result["passed"] is True
        assert result["max_diff_s"] == 0.0

    def test_within_tolerance_passes(self) -> None:
        a = np.array([0.100, 0.200, 0.300])
        b = np.array([0.102, 0.198, 0.304])
        result = validate_timing_cross_engine(a, b, tolerance_s=0.005)
        assert result["passed"] is True

    def test_exceeds_tolerance_fails(self) -> None:
        a = np.array([0.1, 0.2])
        b = np.array([0.1, 0.3])  # 100 ms off
        result = validate_timing_cross_engine(a, b, tolerance_s=0.005)
        assert result["passed"] is False

    def test_mismatched_lengths_fails(self) -> None:
        result = validate_timing_cross_engine(np.array([0.1]), np.array([0.1, 0.2]))
        assert result["passed"] is False


class TestAngleValidation:
    """Cross-engine angle validation tests."""

    def test_identical_angles_pass(self) -> None:
        a = np.array([30.0, 45.0, 50.0])
        result = validate_angle_cross_engine(a, a)
        assert result["passed"] is True
        assert result["max_diff_deg"] == 0.0

    def test_within_tolerance_passes(self) -> None:
        a = np.array([30.0, 45.0])
        b = np.array([31.5, 43.5])
        result = validate_angle_cross_engine(a, b, tolerance_deg=2.0)
        assert result["passed"] is True

    def test_exceeds_tolerance_fails(self) -> None:
        a = np.array([30.0])
        b = np.array([35.0])
        result = validate_angle_cross_engine(a, b, tolerance_deg=2.0)
        assert result["passed"] is False


class TestWorkbenchReportTraceability:
    """Workbench scientific reports surface formula and methodology traceability."""

    def test_launch_monitor_report_text_includes_formula_and_citation(self) -> None:
        """Report text must include SG formula and peer-reviewed citation."""
        from unittest.mock import MagicMock
        from src.tools.launch_monitor_analytics.gui import MainWidget

        widget = MagicMock(spec=MainWidget)
        widget.project = MagicMock(name="Test Project", sessions=[], audit_log=[])
        widget.project.name = "Test Project"
        widget.analysis_frame = MagicMock(columns=[])
        widget.report_text = MagicMock()

        MainWidget._refresh_report(widget)
        call_args = widget.report_text.setPlainText.call_args[0][0]
        assert "Methodology & Formula Traceability:" in call_args
        assert (
            "SG = verified E(start state) - 1 - verified E(finish state)" in call_args
        )
        assert "Broadie 2011/2014, DOI: 10.1287/inte.1110.0594" in call_args
        assert "Hotelling T^2" in call_args
        assert "Variance Inflation Factor" in call_args
