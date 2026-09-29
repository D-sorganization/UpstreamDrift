"""Tests for src.shared.python.analysis.basic_stats (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.analysis.basic_stats import BasicStatsMixin


class _Concrete(BasicStatsMixin):
    """Minimal concrete subclass for testing BasicStatsMixin."""

    def __init__(self, n: int = 50) -> None:
        self.times = np.linspace(0.0, 1.0, n)
        # Sinusoidal club head speed
        self.club_head_speed = np.abs(np.sin(np.linspace(0, np.pi, n))) * 40.0


class TestComputeSummaryStats:
    def setup_method(self) -> None:
        self.obj = _Concrete(n=50)

    def test_returns_summary_statistics(self) -> None:
        data = np.linspace(1.0, 10.0, 50)
        stats = self.obj.compute_summary_stats(data)
        assert stats is not None

    def test_mean_correct(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = self.obj.compute_summary_stats(data)
        assert stats.mean == pytest.approx(3.0)

    def test_min_correct(self) -> None:
        data = np.array([5.0, 2.0, 8.0, 1.0, 4.0])
        stats = self.obj.compute_summary_stats(data)
        assert stats.min == pytest.approx(1.0)

    def test_max_correct(self) -> None:
        data = np.array([5.0, 2.0, 8.0, 1.0, 4.0])
        stats = self.obj.compute_summary_stats(data)
        assert stats.max == pytest.approx(8.0)

    def test_std_non_negative(self) -> None:
        data = np.random.randn(30)
        stats = self.obj.compute_summary_stats(data)
        assert stats.std >= 0.0

    def test_range_non_negative(self) -> None:
        data = np.random.randn(30)
        stats = self.obj.compute_summary_stats(data)
        assert stats.range >= 0.0

    def test_rms_non_negative(self) -> None:
        data = np.random.randn(30)
        stats = self.obj.compute_summary_stats(data)
        assert stats.rms >= 0.0

    def test_range_equals_max_minus_min(self) -> None:
        data = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        stats = self.obj.compute_summary_stats(data)
        assert stats.range == pytest.approx(stats.max - stats.min)

    def test_constant_data_std_zero(self) -> None:
        data = np.ones(20) * 7.0
        stats = self.obj.compute_summary_stats(data)
        assert stats.std == pytest.approx(0.0, abs=1e-12)

    def test_constant_data_range_zero(self) -> None:
        data = np.ones(20) * 7.0
        stats = self.obj.compute_summary_stats(data)
        assert stats.range == pytest.approx(0.0, abs=1e-12)

    def test_empty_data_raises(self) -> None:
        from src.shared.python.core.contracts.exceptions import PreconditionError

        with pytest.raises((PreconditionError, AssertionError)):
            self.obj.compute_summary_stats(np.array([]))

    def test_min_time_within_range(self) -> None:
        data = np.linspace(5.0, 1.0, 50)  # decreasing → min at end
        stats = self.obj.compute_summary_stats(data)
        assert self.obj.times[0] <= stats.min_time <= self.obj.times[-1]

    def test_max_time_within_range(self) -> None:
        data = np.linspace(1.0, 5.0, 50)  # increasing → max at end
        stats = self.obj.compute_summary_stats(data)
        assert self.obj.times[0] <= stats.max_time <= self.obj.times[-1]

    def test_rms_all_ones(self) -> None:
        data = np.ones(20)
        stats = self.obj.compute_summary_stats(data)
        assert stats.rms == pytest.approx(1.0)

    def test_median_correct(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        stats = self.obj.compute_summary_stats(data)
        assert stats.median == pytest.approx(3.0)


class TestFindPeaksInData:
    def setup_method(self) -> None:
        n = 200
        self.obj = _Concrete(n=n)
        # Signal with clear peaks at t ≈ 0.25 and t ≈ 0.75
        t = np.linspace(0, 1, n)
        self.obj.times = t
        self.two_peak_data = np.exp(-((t - 0.25) ** 2) / 0.001) + np.exp(
            -((t - 0.75) ** 2) / 0.001
        )

    def test_basic_stats_returns_list(self) -> None:
        peaks = self.obj.find_peaks_in_data(self.two_peak_data)
        assert isinstance(peaks, list)

    def test_finds_two_peaks(self) -> None:
        peaks = self.obj.find_peaks_in_data(self.two_peak_data)
        assert len(peaks) == 2

    def test_peak_value_positive(self) -> None:
        peaks = self.obj.find_peaks_in_data(self.two_peak_data)
        for p in peaks:
            assert p.value > 0.0

    def test_peak_time_in_range(self) -> None:
        peaks = self.obj.find_peaks_in_data(self.two_peak_data)
        for p in peaks:
            assert self.obj.times[0] <= p.time <= self.obj.times[-1]

    def test_no_peaks_empty_list(self) -> None:
        flat_data = np.ones(200)
        peaks = self.obj.find_peaks_in_data(flat_data)
        assert peaks == []

    def test_height_filter(self) -> None:
        # With a high threshold, no peaks should survive
        peaks = self.obj.find_peaks_in_data(self.two_peak_data, height=2.0)
        assert peaks == []
