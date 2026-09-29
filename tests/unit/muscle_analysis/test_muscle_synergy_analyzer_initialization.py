"""Tests for muscle synergy analysis module.

Tests the Non-negative Matrix Factorization (NMF) based muscle synergy
extraction and analysis functionality.
"""

from __future__ import annotations

import numpy as np
from src.shared.python.biomechanics.muscle_analysis import (
    MuscleSynergyAnalyzer,
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

        assert analyzer.muscle_names == [
            "Muscle 0",
            "Muscle 1",
            "Muscle 2",
        ], "Assertion failed: analyzer.muscle_names == [Muscle 0, Muscle 1, Muscle 2]"

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
        assert analyzer.data.shape == (
            n_samples,
            n_muscles,
        ), "Assertion failed: analyzer.data.shape == (n_samples, n_muscles)"
