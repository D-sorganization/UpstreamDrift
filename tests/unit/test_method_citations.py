"""Tests for method citation metadata and cross-engine validation (Issue #777)."""

from __future__ import annotations

import numpy as np

from src.shared.python.analysis.dataclasses import (
    CITATION_CRUNCH_FACTOR,
    CITATION_KINEMATIC_SEQUENCE,
    CITATION_SPINAL_LOAD,
    CITATION_X_FACTOR,
    MethodCitation,
    validate_angle_cross_engine,
    validate_timing_cross_engine,
)


class TestMethodCitation:
    """MethodCitation dataclass tests."""

    def test_frozen(self) -> None:
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
