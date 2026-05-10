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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_data(self) -> None:
        """Test with single sample (degenerate case)."""
        data = np.array([[0.5, 0.3, 0.8]])  # 1 sample, 3 muscles
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 1, "Assertion failed: analyzer.n_samples == 1"
        assert analyzer.n_muscles == 3, "Assertion failed: analyzer.n_muscles == 3"

    def test_single_muscle_data(self) -> None:
        """Test with single muscle."""
        data = np.random.rand(100, 1)  # 100 samples, 1 muscle
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 100, "Assertion failed: analyzer.n_samples == 100"
        assert analyzer.n_muscles == 1, "Assertion failed: analyzer.n_muscles == 1"

        # Can only extract 1 synergy
        if SKLEARN_AVAILABLE:
            result = analyzer.extract_synergies(n_synergies=1)
            assert result.n_synergies == 1, "Assertion failed: result.n_synergies == 1"

    def test_all_zeros_data(self) -> None:
        """Test with all-zero data."""
        data = np.zeros((100, 5))
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 100, "Assertion failed: analyzer.n_samples == 100"
        assert analyzer.n_muscles == 5, "Assertion failed: analyzer.n_muscles == 5"

        # NMF might have issues with all-zero data, but shouldn't crash
        if SKLEARN_AVAILABLE:
            try:
                result = analyzer.extract_synergies(n_synergies=2)
                # If it succeeds, check basic properties
                assert (
                    result.n_synergies == 2
                ), "Assertion failed: result.n_synergies == 2"
            except (ValueError, RuntimeError):
                # Some NMF implementations may fail on zero data
                pass

    def test_uniform_activation_data(self) -> None:
        """Test with uniform activation (all same value)."""
        data = np.ones((100, 5)) * 0.5
        analyzer = MuscleSynergyAnalyzer(data)

        if SKLEARN_AVAILABLE:
            result = analyzer.extract_synergies(n_synergies=1)
            # Should be able to extract, though VAF might be perfect or undefined
            assert result.n_synergies == 1, "Assertion failed: result.n_synergies == 1"

    def test_very_large_number_of_muscles(self) -> None:
        """Test with large number of muscles."""
        n_muscles = 100
        data = np.random.rand(50, n_muscles)
        analyzer = MuscleSynergyAnalyzer(data)

        assert (
            analyzer.n_muscles == n_muscles
        ), "Assertion failed: analyzer.n_muscles == n_muscles"

        if SKLEARN_AVAILABLE:
            # Should be able to extract synergies
            result = analyzer.extract_synergies(n_synergies=5)
            assert result.weights.shape == (
                n_muscles,
                5,
            ), "Assertion failed: result.weights.shape == (n_muscles, 5)"

    def test_very_long_time_series(self) -> None:
        """Test with very long time series."""
        n_samples = 10000
        data = np.random.rand(n_samples, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        assert (
            analyzer.n_samples == n_samples
        ), "Assertion failed: analyzer.n_samples == n_samples"

        if SKLEARN_AVAILABLE:
            result = analyzer.extract_synergies(n_synergies=2)
            assert result.activations.shape == (
                2,
                n_samples,
            ), "Assertion failed: result.activations.shape == (2, n_samples)"
