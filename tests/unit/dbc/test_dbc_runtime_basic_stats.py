"""DbC runtime contract tests for analysis.basic_stats module.

Verifies require() and ensure() contracts on BasicStatsMixin.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.shared.python.analysis.basic_stats import BasicStatsMixin
from src.shared.python.core.contracts import PreconditionError


class _StubAnalyzer(BasicStatsMixin):
    """Minimal stub implementing attributes expected by BasicStatsMixin."""

    def __init__(self, times: np.ndarray) -> None:
        self.times = times
        self.club_head_speed: np.ndarray | None = None


# ── compute_summary_stats contracts ────────────────────────────────


class TestSummaryStatsPreconditions(unittest.TestCase):
    """Test require() on compute_summary_stats."""

    def test_empty_data_rejected(self) -> None:
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        with self.assertRaises(PreconditionError):
            analyzer.compute_summary_stats(np.array([]))

    def test_non_empty_data_accepted(self) -> None:
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = np.sin(2 * np.pi * t)
        result = analyzer.compute_summary_stats(data)
        self.assertIsNotNone(result)


class TestSummaryStatsPostconditions(unittest.TestCase):
    """Test ensure() postconditions on compute_summary_stats."""

    def test_std_non_negative(self) -> None:
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = np.sin(2 * np.pi * t)
        result = analyzer.compute_summary_stats(data)
        self.assertGreaterEqual(result.std, 0)

    def test_range_non_negative(self) -> None:
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = np.random.default_rng(42).standard_normal(100)
        result = analyzer.compute_summary_stats(data)
        self.assertGreaterEqual(result.range, 0)

    def test_rms_non_negative(self) -> None:
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = np.sin(2 * np.pi * t)
        result = analyzer.compute_summary_stats(data)
        self.assertGreaterEqual(result.rms, 0)

    def test_constant_signal_std_zero(self) -> None:
        """A constant signal must have std == 0."""
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = 5.0 * np.ones(100)
        result = analyzer.compute_summary_stats(data)
        self.assertAlmostEqual(result.std, 0.0)
        self.assertAlmostEqual(result.range, 0.0)

    def test_constant_signal_rms_equals_value(self) -> None:
        """For a constant 5.0 signal, RMS should be 5.0."""
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = 5.0 * np.ones(100)
        result = analyzer.compute_summary_stats(data)
        self.assertAlmostEqual(result.rms, 5.0)

    def test_min_max_consistency(self) -> None:
        """max must be >= min, so range >= 0."""
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = np.sin(2 * np.pi * 3 * t) + 2.0 * np.cos(2 * np.pi * t)
        result = analyzer.compute_summary_stats(data)
        self.assertGreaterEqual(result.max, result.min)

    def test_single_element_data(self) -> None:
        """A single-element array should have std=0, range=0."""
        t = np.array([0.0])
        analyzer = _StubAnalyzer(t)
        data = np.array([3.14])
        result = analyzer.compute_summary_stats(data)
        self.assertAlmostEqual(result.std, 0.0)
        self.assertAlmostEqual(result.range, 0.0)
        self.assertAlmostEqual(result.rms, 3.14, places=5)

    def test_negative_values(self) -> None:
        """Negative signals should still satisfy postconditions."""
        t = np.linspace(0, 1, 100)
        analyzer = _StubAnalyzer(t)
        data = -10.0 * np.ones(100)
        result = analyzer.compute_summary_stats(data)
        self.assertGreaterEqual(result.std, 0)
        self.assertGreaterEqual(result.range, 0)
        self.assertGreaterEqual(result.rms, 0)


if __name__ == "__main__":
    unittest.main()
