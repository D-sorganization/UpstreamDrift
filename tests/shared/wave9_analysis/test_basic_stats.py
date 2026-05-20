"""Tests for src/shared/python/analysis/basic_stats.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.basic_stats import BasicStatsMixin
from src.shared.python.analysis.dataclasses import PeakInfo, SummaryStatistics


class _Holder(BasicStatsMixin):
    def __init__(
        self,
        times: np.ndarray,
        club_head_speed: np.ndarray | None = None,
    ) -> None:
        self.times = times
        self.club_head_speed = club_head_speed


@pytest.fixture
def holder() -> _Holder:
    times = np.linspace(0.0, 1.0, 11)
    chs = np.array([0, 1, 2, 3, 5, 8, 5, 3, 2, 1, 0], dtype=float)
    return _Holder(times=times, club_head_speed=chs)


class TestComputeSummaryStats:
    def test_basic_values(self, holder: _Holder) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        holder.times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        stats = holder.compute_summary_stats(data)
        assert isinstance(stats, SummaryStatistics)
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.range == 4.0
        assert stats.min_time == pytest.approx(0.0)
        assert stats.max_time == pytest.approx(0.4)
        assert stats.std >= 0
        assert stats.rms >= 0

    def test_constant_array(self) -> None:
        h = _Holder(times=np.array([0.0, 0.5, 1.0]))
        stats = h.compute_summary_stats(np.array([2.0, 2.0, 2.0]))
        assert stats.std == 0.0
        assert stats.range == 0.0
        assert stats.rms == pytest.approx(2.0)

    def test_empty_data_raises(self, holder: _Holder) -> None:
        with pytest.raises(ValueError):
            holder.compute_summary_stats(np.array([]))

    def test_none_data_raises(self, holder: _Holder) -> None:
        with pytest.raises(ValueError, match="data must be provided"):
            holder.compute_summary_stats(None)  # type: ignore[arg-type]

    def test_negative_values(self) -> None:
        h = _Holder(times=np.array([0.0, 1.0, 2.0]))
        stats = h.compute_summary_stats(np.array([-3.0, 0.0, 3.0]))
        assert stats.min == -3.0
        assert stats.max == 3.0
        assert stats.range == 6.0


class TestFindPeaksInData:
    def test_finds_peaks(self, holder: _Holder) -> None:
        data = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        holder.times = np.linspace(0.0, 6.0, 7)
        peaks = holder.find_peaks_in_data(data)
        assert len(peaks) == 3
        for p in peaks:
            assert isinstance(p, PeakInfo)
        assert peaks[-1].value == 3.0

    def test_with_height(self, holder: _Holder) -> None:
        data = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        holder.times = np.linspace(0.0, 6.0, 7)
        peaks = holder.find_peaks_in_data(data, height=1.5)
        assert all(p.value >= 1.5 for p in peaks)
        assert len(peaks) == 2

    def test_with_prominence(self, holder: _Holder) -> None:
        data = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        holder.times = np.linspace(0.0, 6.0, 7)
        peaks = holder.find_peaks_in_data(data, prominence=0.5)
        for p in peaks:
            assert p.prominence is not None

    def test_none_data_raises(self, holder: _Holder) -> None:
        with pytest.raises(ValueError, match="data must be provided"):
            holder.find_peaks_in_data(None)  # type: ignore[arg-type]

    def test_no_peaks(self, holder: _Holder) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        holder.times = np.linspace(0.0, 4.0, 5)
        peaks = holder.find_peaks_in_data(data)
        assert peaks == []


class TestFindClubHeadSpeedPeak:
    def test_returns_peak(self, holder: _Holder) -> None:
        peak = holder.find_club_head_speed_peak()
        assert peak is not None
        assert peak.value == 8.0
        assert peak.index == 5

    def test_none_when_no_data(self) -> None:
        h = _Holder(times=np.array([0.0]), club_head_speed=None)
        assert h.find_club_head_speed_peak() is None

    def test_none_when_empty(self) -> None:
        h = _Holder(times=np.array([0.0]), club_head_speed=np.array([]))
        assert h.find_club_head_speed_peak() is None
