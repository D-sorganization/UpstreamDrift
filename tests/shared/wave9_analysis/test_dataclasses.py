"""Tests for src/shared/python/analysis/dataclasses.py validators and citations."""

from __future__ import annotations

import numpy as np

from src.shared.python.analysis.dataclasses import (
    ANGLE_TOLERANCE_DEG,
    CITATION_CRUNCH_FACTOR,
    CITATION_KINEMATIC_SEQUENCE,
    CITATION_SEGMENT_TIMING,
    CITATION_SPINAL_LOAD,
    CITATION_X_FACTOR,
    TIMING_TOLERANCE_S,
    KinematicSequenceInfo,
    MethodCitation,
    PeakInfo,
    SegmentTimingInfo,
    SummaryStatistics,
    validate_angle_cross_engine,
    validate_timing_cross_engine,
)


class TestCitations:
    def test_x_factor_metadata(self) -> None:
        assert CITATION_X_FACTOR.name == "X-Factor"
        assert CITATION_X_FACTOR.year == 2001
        assert "Cheetham" in CITATION_X_FACTOR.authors

    def test_segment_timing_is_kinematic_alias(self) -> None:
        assert CITATION_KINEMATIC_SEQUENCE is CITATION_SEGMENT_TIMING

    def test_segment_timing_info_alias(self) -> None:
        assert KinematicSequenceInfo is SegmentTimingInfo

    def test_citations_immutable(self) -> None:
        # MethodCitation is frozen
        import dataclasses

        try:
            CITATION_CRUNCH_FACTOR.year = 9999  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("expected FrozenInstanceError")

    def test_spinal_load_has_notes(self) -> None:
        assert CITATION_SPINAL_LOAD.notes is not None

    def test_method_citation_optional_fields(self) -> None:
        c = MethodCitation(name="x", authors="y", year=2020, title="t")
        assert c.doi is None
        assert c.notes is None


class TestSimpleDataclasses:
    def test_peak_info_defaults(self) -> None:
        p = PeakInfo(value=1.0, time=0.5, index=2)
        assert p.prominence is None
        assert p.width is None

    def test_summary_statistics_fields(self) -> None:
        s = SummaryStatistics(
            mean=1.0,
            median=1.0,
            std=0.5,
            min=0.0,
            max=2.0,
            range=2.0,
            min_time=0.0,
            max_time=1.0,
            rms=1.0,
        )
        assert s.range == 2.0


class TestValidateTimingCrossEngine:
    def test_passes_within_tolerance(self) -> None:
        a = np.array([0.0, 0.1, 0.2])
        b = np.array([0.001, 0.1015, 0.2])
        result = validate_timing_cross_engine(a, b)
        assert result["passed"] is True
        assert isinstance(result["max_diff_s"], float)

    def test_fails_outside_tolerance(self) -> None:
        a = np.array([0.0, 0.1])
        b = np.array([0.0, 0.5])
        result = validate_timing_cross_engine(a, b)
        assert result["passed"] is False
        assert result["max_diff_s"] == 0.4

    def test_length_mismatch(self) -> None:
        result = validate_timing_cross_engine(np.array([0.0, 0.1]), np.array([0.0]))
        assert result["passed"] is False
        assert result["max_diff_s"] == float("inf")

    def test_empty(self) -> None:
        result = validate_timing_cross_engine(np.array([]), np.array([]))
        assert result["passed"] is True
        assert result["max_diff_s"] == 0.0

    def test_custom_tolerance(self) -> None:
        a = np.array([0.0])
        b = np.array([0.02])
        result = validate_timing_cross_engine(a, b, tolerance_s=0.05)
        assert result["passed"] is True

    def test_default_tolerance_value(self) -> None:
        assert TIMING_TOLERANCE_S == 0.005

    def test_none_input_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="must be provided"):
            validate_timing_cross_engine(None, np.array([0.0]))  # type: ignore[arg-type]


class TestValidateAngleCrossEngine:
    def test_passes_within_tolerance(self) -> None:
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([10.5, 20.0, 31.0])
        result = validate_angle_cross_engine(a, b)
        assert result["passed"] is True

    def test_fails_outside_tolerance(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 5.0])
        result = validate_angle_cross_engine(a, b)
        assert result["passed"] is False
        assert result["max_diff_deg"] == 5.0

    def test_shape_mismatch(self) -> None:
        result = validate_angle_cross_engine(np.array([[0.0, 1.0]]), np.array([0.0]))
        assert result["passed"] is False
        assert result["max_diff_deg"] == float("inf")

    def test_empty(self) -> None:
        result = validate_angle_cross_engine(np.array([]), np.array([]))
        assert result["passed"] is True
        assert result["max_diff_deg"] == 0.0

    def test_default_tolerance(self) -> None:
        assert ANGLE_TOLERANCE_DEG == 2.0

    def test_none_input_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="must be provided"):
            validate_angle_cross_engine(None, np.array([0.0]))  # type: ignore[arg-type]
