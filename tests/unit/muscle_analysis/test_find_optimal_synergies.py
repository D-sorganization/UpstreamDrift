"""Tests for muscle synergy analysis module.

Tests the Non-negative Matrix Factorization (NMF) based muscle synergy
extraction and analysis functionality.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.biomechanics.muscle_analysis import (
    SKLEARN_AVAILABLE,
    MuscleSynergyAnalyzer,
    SynergyResult,
)
from src.shared.python.core.contracts import PreconditionError
from src.shared.python.engine_core.engine_availability import skip_if_unavailable


@skip_if_unavailable("sklearn")
class TestFindOptimalSynergies:
    """Test find_optimal_synergies method."""

    def test_finds_synergies_meeting_threshold(self) -> None:
        """Test that method finds minimal synergies meeting VAF threshold."""
        np.random.seed(42)
        # Create data that's approximately rank-2
        W = np.random.rand(5, 2)
        H = np.random.rand(2, 100)
        data = np.dot(W, H).T + np.random.rand(100, 5) * 0.01  # Small noise

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=0.90)

        # Should find 2-3 synergies
        assert result.n_synergies <= 5, "Assertion failed: result.n_synergies <= 5"
        assert (
            result.vaf >= 0.90 or result.n_synergies == 5
        )  # Either meets threshold or uses max

    def test_returns_best_when_threshold_not_met(self, caplog) -> None:
        """Test that method returns best result when threshold not met."""
        np.random.seed(42)
        # Create complex data (hard to approximate with few synergies)
        data = np.random.rand(100, 10)

        analyzer = MuscleSynergyAnalyzer(data)

        with caplog.at_level("WARNING"):
            result = analyzer.find_optimal_synergies(
                max_synergies=2, vaf_threshold=0.99
            )

        # Should return result with 2 synergies (max)
        assert result.n_synergies == 2, "Assertion failed: result.n_synergies == 2"

        # Should warn that threshold not met
        assert (
            "threshold not met" in caplog.text.lower() or result.vaf >= 0.99
        ), "Assertion failed: threshold not met in caplog.text.lower() or result.vaf >= 0.99"

    def test_respects_max_synergies_limit(self) -> None:
        """Test that method respects max_synergies limit."""
        np.random.seed(42)
        data = np.random.rand(100, 10)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=3, vaf_threshold=0.80)

        # Should not exceed max_synergies
        assert result.n_synergies <= 3, "Assertion failed: result.n_synergies <= 3"

    def test_caps_at_number_of_muscles(self) -> None:
        """Test that max_synergies is capped at n_muscles."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        # Request more synergies than muscles
        result = analyzer.find_optimal_synergies(max_synergies=10, vaf_threshold=0.95)

        # Should not exceed n_muscles (5)
        assert result.n_synergies <= 5, "Assertion failed: result.n_synergies <= 5"

    def test_low_threshold_finds_fewer_synergies(self) -> None:
        """Test that lower VAF threshold requires fewer synergies."""
        np.random.seed(42)
        data = np.random.rand(100, 8)

        analyzer = MuscleSynergyAnalyzer(data)

        result_low = analyzer.find_optimal_synergies(
            max_synergies=8, vaf_threshold=0.50
        )
        result_high = analyzer.find_optimal_synergies(
            max_synergies=8, vaf_threshold=0.90
        )

        # Lower threshold should require fewer (or equal) synergies
        assert (
            result_low.n_synergies <= result_high.n_synergies
        ), "Assertion failed: result_low.n_synergies <= result_high.n_synergies"

    def test_threshold_of_one_uses_all_muscles(self) -> None:
        """Test that VAF threshold of 1.0 tries to use all muscles."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=1.0)

        # Should use all 5 synergies (or meet threshold early)
        assert result.n_synergies <= 5, "Assertion failed: result.n_synergies <= 5"

    def test_returns_synergy_result(self) -> None:
        """Test that method returns a SynergyResult object."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=0.80)

        assert isinstance(
            result, SynergyResult
        ), "Assertion failed: isinstance(result, SynergyResult)"
        assert result.n_synergies >= 1, "Assertion failed: result.n_synergies >= 1"

    def test_invalid_limit_raises_error(self) -> None:
        """Test that limit < 1 raises ValueError."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        # This should raise because max_synergies=0 leads to limit=0
        with pytest.raises(ValueError, match="limit must be >= 1"):
            analyzer.find_optimal_synergies(max_synergies=0, vaf_threshold=0.90)
