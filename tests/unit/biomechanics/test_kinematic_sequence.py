"""Tests for src.shared.python.biomechanics.kinematic_sequence (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.biomechanics.kinematic_sequence import (
    KinematicSequenceResult,
    SegmentPeak,
    SegmentTimingAnalyzer,
    SegmentTimingResult,
)


def _make_velocities(n_samples: int = 100) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Create synthetic segment velocities with identifiable peaks."""
    times = np.linspace(0.0, 1.0, n_samples)
    # Hip peaks at t=0.3, shoulder at t=0.5, wrist at t=0.7
    hip = np.exp(-((times - 0.3) ** 2) / 0.01)
    shoulder = np.exp(-((times - 0.5) ** 2) / 0.01)
    wrist = np.exp(-((times - 0.7) ** 2) / 0.01)
    return {"hip": hip, "shoulder": shoulder, "wrist": wrist}, times


class TestSegmentTimingAnalyzer:
    def test_construction_no_expected_order(self) -> None:
        analyzer = SegmentTimingAnalyzer()
        assert analyzer.expected_order is None

    def test_construction_with_expected_order(self) -> None:
        analyzer = SegmentTimingAnalyzer(["hip", "shoulder", "wrist"])
        assert analyzer.expected_order == ["hip", "shoulder", "wrist"]

    def test_analyze_returns_result(self) -> None:
        velocities, times = _make_velocities()
        analyzer = SegmentTimingAnalyzer()
        result = analyzer.analyze(velocities, times)
        assert isinstance(result, SegmentTimingResult)

    def test_analyze_returns_peaks_for_each_segment(self) -> None:
        velocities, times = _make_velocities()
        analyzer = SegmentTimingAnalyzer()
        result = analyzer.analyze(velocities, times)
        assert len(result.peaks) == len(velocities)

    def test_peaks_have_non_negative_velocity(self) -> None:
        velocities, times = _make_velocities()
        analyzer = SegmentTimingAnalyzer()
        result = analyzer.analyze(velocities, times)
        for peak in result.peaks:
            assert peak.peak_velocity >= 0.0

    def test_sequence_consistency_in_0_1(self) -> None:
        velocities, times = _make_velocities()
        analyzer = SegmentTimingAnalyzer()
        result = analyzer.analyze(velocities, times)
        assert 0.0 <= result.sequence_consistency <= 1.0

    def test_timing_gaps_finite(self) -> None:
        velocities, times = _make_velocities()
        analyzer = SegmentTimingAnalyzer()
        result = analyzer.analyze(velocities, times)
        for gap_val in result.timing_gaps.values():
            assert np.isfinite(gap_val)

    def test_sequence_order_contains_all_segments(self) -> None:
        velocities, times = _make_velocities()
        analyzer = SegmentTimingAnalyzer()
        result = analyzer.analyze(velocities, times)
        assert set(result.sequence_order) == set(velocities.keys())

    def test_correct_order_gives_valid_sequence(self) -> None:
        velocities, times = _make_velocities()
        # Peaks occur in order: hip, shoulder, wrist
        analyzer = SegmentTimingAnalyzer(["hip", "shoulder", "wrist"])
        result = analyzer.analyze(velocities, times)
        assert result.is_valid_sequence is True

    def test_wrong_order_gives_invalid_sequence(self) -> None:
        velocities, times = _make_velocities()
        # Reversed order should be invalid
        analyzer = SegmentTimingAnalyzer(["wrist", "shoulder", "hip"])
        result = analyzer.analyze(velocities, times)
        assert result.is_valid_sequence is False

    def test_empty_times_raises(self) -> None:
        analyzer = SegmentTimingAnalyzer()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            analyzer.analyze({}, np.array([]))

    def test_kinematic_sequence_single_segment(self) -> None:
        times = np.linspace(0.0, 1.0, 50)
        velocities = {"hip": np.sin(np.pi * times)}
        analyzer = SegmentTimingAnalyzer()
        result = analyzer.analyze(velocities, times)
        assert len(result.peaks) == 1


class TestSegmentPeak:
    def test_kinematic_sequence_construction(self) -> None:
        peak = SegmentPeak(name="hip", peak_velocity=3.5, time=0.3, index=30)
        assert peak.name == "hip"
        assert peak.peak_velocity == pytest.approx(3.5)
        assert peak.time == pytest.approx(0.3)
        assert peak.index == 30

    def test_default_normalized_velocity(self) -> None:
        peak = SegmentPeak(name="hip", peak_velocity=3.5, time=0.3, index=30)
        assert peak.normalized_velocity == pytest.approx(0.0)

    def test_speed_gain_default_none(self) -> None:
        peak = SegmentPeak(name="hip", peak_velocity=3.5, time=0.3, index=30)
        assert peak.speed_gain is None


class TestKinematicSequenceResultAlias:
    def test_alias_is_same_as_result(self) -> None:
        assert KinematicSequenceResult is SegmentTimingResult
