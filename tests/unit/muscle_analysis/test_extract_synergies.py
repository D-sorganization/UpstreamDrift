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
class TestExtractSynergies:
    """Test extract_synergies method."""

    def test_extract_single_synergy(self) -> None:
        """Test extracting a single synergy."""
        # Create simple synthetic data: 1 synergy
        np.random.seed(42)
        n_samples, n_muscles = 100, 5

        # True synergy: 1 weight vector, 1 activation profile
        W_true = np.random.rand(n_muscles, 1)
        H_true = np.random.rand(1, n_samples)
        data = np.dot(W_true, H_true).T  # (n_samples, n_muscles)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.extract_synergies(n_synergies=1)

        assert result.n_synergies == 1, "Assertion failed: result.n_synergies == 1"
        assert result.weights.shape == (n_muscles, 1), (
            "Assertion failed: result.weights.shape == (n_muscles, 1)"
        )
        assert result.activations.shape == (1, n_samples), (
            "Assertion failed: result.activations.shape == (1, n_samples)"
        )
        assert result.reconstructed.shape == (n_samples, n_muscles), (
            "Assertion failed: result.reconstructed.shape == (n_samples, n_muscles)"
        )

    def test_extract_multiple_synergies(self) -> None:
        """Test extracting multiple synergies."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        result = analyzer.extract_synergies(n_synergies=3)

        assert result.n_synergies == 3, "Assertion failed: result.n_synergies == 3"
        assert result.weights.shape == (5, 3), (
            "Assertion failed: result.weights.shape == (5, 3)"
        )
        assert result.activations.shape == (3, 100), (
            "Assertion failed: result.activations.shape == (3, 100)"
        )
        assert result.reconstructed.shape == (100, 5), (
            "Assertion failed: result.reconstructed.shape == (100, 5)"
        )

    def test_vaf_is_between_zero_and_one(self) -> None:
        """Test that Variance Accounted For is between 0 and 1."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        for n_syn in [1, 2, 3, 4]:
            result = analyzer.extract_synergies(n_synergies=n_syn)
            assert 0.0 <= result.vaf <= 1.0, f"VAF out of range for {n_syn} synergies"

    def test_vaf_increases_with_more_synergies(self) -> None:
        """Test that VAF generally increases with more synergies."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        vaf_1 = analyzer.extract_synergies(n_synergies=1).vaf
        vaf_2 = analyzer.extract_synergies(n_synergies=2).vaf
        vaf_3 = analyzer.extract_synergies(n_synergies=3).vaf

        # More synergies should explain more variance
        assert (
            vaf_2 >= vaf_1 - 0.01
        )  # Allow small numerical tolerance, "Assertion failed: vaf_2 >= vaf_1 - 0.01  # Allow small numerical tolerance"
        assert vaf_3 >= vaf_2 - 0.01, "Assertion failed: vaf_3 >= vaf_2 - 0.01"

    def test_reconstruction_approximates_original(self) -> None:
        """Test that reconstruction approximates original data."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        # With enough synergies, should approximate well
        result = analyzer.extract_synergies(n_synergies=4)

        # VAF should be high
        assert result.vaf > 0.80, (
            "High number of synergies should give good reconstruction"
        )

        # Reconstruction shape should match data
        assert result.reconstructed.shape == data.shape, (
            "Assertion failed: result.reconstructed.shape == data.shape"
        )

    def test_weights_are_nonnegative(self) -> None:
        """Test that muscle weights are non-negative (NMF property)."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        result = analyzer.extract_synergies(n_synergies=2)

        assert np.all(result.weights >= 0), "Weights should be non-negative"

    def test_activations_are_nonnegative(self) -> None:
        """Test that activation profiles are non-negative (NMF property)."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        result = analyzer.extract_synergies(n_synergies=2)

        assert np.all(result.activations >= 0), "Activations should be non-negative"

    def test_invalid_number_of_synergies_too_small(self) -> None:
        """Test that n_synergies < 1 raises PreconditionError."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(PreconditionError):
            analyzer.extract_synergies(n_synergies=0)

    def test_invalid_number_of_synergies_too_large(self) -> None:
        """Test that n_synergies > n_muscles raises PreconditionError."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(PreconditionError):
            analyzer.extract_synergies(n_synergies=6)  # > 5 muscles

    def test_custom_max_iterations(self) -> None:
        """Test that custom max_iter parameter works."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        # Should not raise error with custom iterations
        result = analyzer.extract_synergies(n_synergies=2, max_iter=500)
        assert result.n_synergies == 2, "Assertion failed: result.n_synergies == 2"

    def test_result_includes_muscle_names(self) -> None:
        """Test that result includes muscle names if provided."""
        data = np.random.rand(100, 3)
        names = ["M1", "M2", "M3"]
        analyzer = MuscleSynergyAnalyzer(data, muscle_names=names)

        result = analyzer.extract_synergies(n_synergies=2)
        assert result.muscle_names == names, (
            "Assertion failed: result.muscle_names == names"
        )

    def test_synergies_with_perfect_rank_1_data(self) -> None:
        """Test synergy extraction on perfect rank-1 data."""
        np.random.seed(42)
        n_samples, n_muscles = 100, 5

        # Create perfect rank-1 data: single synergy
        W_true = np.random.rand(n_muscles, 1)
        H_true = np.random.rand(1, n_samples)
        data = np.dot(W_true, H_true).T

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.extract_synergies(n_synergies=1)

        # VAF should be very high (near perfect reconstruction)
        assert result.vaf > 0.98, (
            f"VAF should be near 1.0 for rank-1 data, got {result.vaf}"
        )
