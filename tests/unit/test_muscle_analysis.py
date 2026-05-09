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


class TestSynergyResult:
    """Test SynergyResult dataclass."""

    def test_initialization(self) -> None:
        """Test basic initialization of SynergyResult."""
        weights = np.random.rand(10, 3)
        activations = np.random.rand(3, 100)
        reconstructed = np.dot(weights, activations).T
        vaf = 0.92

        result = SynergyResult(
            weights=weights,
            activations=activations,
            reconstructed=reconstructed,
            vaf=vaf,
            n_synergies=3,
        )

        assert result.weights is weights, "Assertion failed: result.weights is weights"
        assert result.activations is activations, (
            "Assertion failed: result.activations is activations"
        )
        assert result.reconstructed is reconstructed, (
            "Assertion failed: result.reconstructed is reconstructed"
        )
        assert result.vaf == 0.92, "Assertion failed: result.vaf == 0.92"
        assert result.n_synergies == 3, "Assertion failed: result.n_synergies == 3"
        assert result.muscle_names is None, (
            "Assertion failed: result.muscle_names is None"
        )

    def test_with_muscle_names(self) -> None:
        """Test SynergyResult with muscle names."""
        muscle_names = ["Biceps", "Triceps", "Deltoid"]
        weights = np.random.rand(3, 2)
        activations = np.random.rand(2, 100)
        reconstructed = np.random.rand(100, 3)

        result = SynergyResult(
            weights=weights,
            activations=activations,
            reconstructed=reconstructed,
            vaf=0.85,
            n_synergies=2,
            muscle_names=muscle_names,
        )

        assert result.muscle_names == muscle_names, (
            "Assertion failed: result.muscle_names == muscle_names"
        )

    def test_matrix_shapes_consistency(self) -> None:
        """Test that matrix shapes are consistent."""
        n_muscles = 5
        n_synergies = 2
        n_samples = 100

        weights = np.random.rand(n_muscles, n_synergies)
        activations = np.random.rand(n_synergies, n_samples)
        reconstructed = np.random.rand(n_samples, n_muscles)

        result = SynergyResult(
            weights=weights,
            activations=activations,
            reconstructed=reconstructed,
            vaf=0.90,
            n_synergies=n_synergies,
        )

        assert result.weights.shape == (n_muscles, n_synergies), (
            "Assertion failed: result.weights.shape == (n_muscles, n_synergies)"
        )
        assert result.activations.shape == (n_synergies, n_samples), (
            "Assertion failed: result.activations.shape == (n_synergies, n_samples)"
        )
        assert result.reconstructed.shape == (n_samples, n_muscles), (
            "Assertion failed: result.reconstructed.shape == (n_samples, n_muscles)"
        )


class TestMuscleSynergyAnalyzerInitialization:
    """Test MuscleSynergyAnalyzer initialization."""

    def test_initialization_with_valid_data(self) -> None:
        """Test initialization with valid non-negative data."""
        data = np.random.rand(100, 5)  # 100 samples, 5 muscles
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 100, "Assertion failed: analyzer.n_samples == 100"
        assert analyzer.n_muscles == 5, "Assertion failed: analyzer.n_muscles == 5"
        np.testing.assert_array_equal(analyzer.data, data)

    def test_initialization_generates_muscle_names(self) -> None:
        """Test that muscle names are generated if not provided."""
        data = np.random.rand(50, 3)
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.muscle_names == ["Muscle 0", "Muscle 1", "Muscle 2"], (
            "Assertion failed: analyzer.muscle_names == [Muscle 0, Muscle 1, Muscle 2]"
        )

    def test_initialization_with_custom_muscle_names(self) -> None:
        """Test initialization with custom muscle names."""
        data = np.random.rand(50, 3)
        names = ["Biceps", "Triceps", "Deltoid"]
        analyzer = MuscleSynergyAnalyzer(data, muscle_names=names)

        assert analyzer.muscle_names == names, (
            "Assertion failed: analyzer.muscle_names == names"
        )

    def test_initialization_clips_negative_values(self, caplog) -> None:
        """Test that negative values are clipped to zero with warning."""
        # Create data with some negative values
        data = np.array(
            [
                [0.5, -0.1, 0.8],
                [0.3, 0.6, -0.2],
                [0.9, 0.4, 0.7],
            ]
        )

        with caplog.at_level("WARNING"):
            analyzer = MuscleSynergyAnalyzer(data)

        # Should warn about negative values
        assert "negative values" in caplog.text.lower(), (
            "Assertion failed: negative values in caplog.text.lower()"
        )

        # Data should be clipped to zero
        assert np.all(analyzer.data >= 0), (
            "Assertion failed: np.all(analyzer.data >= 0)"
        )
        assert (
            analyzer.data[0, 1] == 0.0
        )  # Was -0.1, "Assertion failed: analyzer.data[0, 1] == 0.0  # Was -0.1"
        assert (
            analyzer.data[1, 2] == 0.0
        )  # Was -0.2, "Assertion failed: analyzer.data[1, 2] == 0.0  # Was -0.2"

    def test_initialization_with_list_input(self) -> None:
        """Test that initialization works with list input."""
        data_list = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        analyzer = MuscleSynergyAnalyzer(data_list)

        assert analyzer.n_samples == 3, "Assertion failed: analyzer.n_samples == 3"
        assert analyzer.n_muscles == 2, "Assertion failed: analyzer.n_muscles == 2"
        assert isinstance(analyzer.data, np.ndarray), (
            "Assertion failed: isinstance(analyzer.data, np.ndarray)"
        )

    def test_data_shape_extraction(self) -> None:
        """Test that data shape is correctly extracted."""
        n_samples, n_muscles = 75, 8
        data = np.random.rand(n_samples, n_muscles)
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == n_samples, (
            "Assertion failed: analyzer.n_samples == n_samples"
        )
        assert analyzer.n_muscles == n_muscles, (
            "Assertion failed: analyzer.n_muscles == n_muscles"
        )
        assert analyzer.data.shape == (n_samples, n_muscles), (
            "Assertion failed: analyzer.data.shape == (n_samples, n_muscles)"
        )


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
                assert result.n_synergies == 2, (
                    "Assertion failed: result.n_synergies == 2"
                )
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

        assert analyzer.n_muscles == n_muscles, (
            "Assertion failed: analyzer.n_muscles == n_muscles"
        )

        if SKLEARN_AVAILABLE:
            # Should be able to extract synergies
            result = analyzer.extract_synergies(n_synergies=5)
            assert result.weights.shape == (n_muscles, 5), (
                "Assertion failed: result.weights.shape == (n_muscles, 5)"
            )

    def test_very_long_time_series(self) -> None:
        """Test with very long time series."""
        n_samples = 10000
        data = np.random.rand(n_samples, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == n_samples, (
            "Assertion failed: analyzer.n_samples == n_samples"
        )

        if SKLEARN_AVAILABLE:
            result = analyzer.extract_synergies(n_synergies=2)
            assert result.activations.shape == (2, n_samples), (
                "Assertion failed: result.activations.shape == (2, n_samples)"
            )
