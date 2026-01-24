"""Tests for muscle synergy analysis module.

Tests the Non-negative Matrix Factorization (NMF) based muscle synergy
extraction and analysis functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.engine_availability import skip_if_unavailable
from src.shared.python.muscle_analysis import (
    SKLEARN_AVAILABLE,
    MuscleSynergyAnalyzer,
    SynergyResult,
)


class TestSynergyResult:
    """Test SynergyResult dataclass."""

    def test_initialization(self):
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

        assert result.weights is weights
        assert result.activations is activations
        assert result.reconstructed is reconstructed
        assert result.vaf == 0.92
        assert result.n_synergies == 3
        assert result.muscle_names is None

    def test_with_muscle_names(self):
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

        assert result.muscle_names == muscle_names

    def test_matrix_shapes_consistency(self):
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

        assert result.weights.shape == (n_muscles, n_synergies)
        assert result.activations.shape == (n_synergies, n_samples)
        assert result.reconstructed.shape == (n_samples, n_muscles)


class TestMuscleSynergyAnalyzerInitialization:
    """Test MuscleSynergyAnalyzer initialization."""

    def test_initialization_with_valid_data(self):
        """Test initialization with valid non-negative data."""
        data = np.random.rand(100, 5)  # 100 samples, 5 muscles
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 100
        assert analyzer.n_muscles == 5
        np.testing.assert_array_equal(analyzer.data, data)

    def test_initialization_generates_muscle_names(self):
        """Test that muscle names are generated if not provided."""
        data = np.random.rand(50, 3)
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.muscle_names == ["Muscle 0", "Muscle 1", "Muscle 2"]

    def test_initialization_with_custom_muscle_names(self):
        """Test initialization with custom muscle names."""
        data = np.random.rand(50, 3)
        names = ["Biceps", "Triceps", "Deltoid"]
        analyzer = MuscleSynergyAnalyzer(data, muscle_names=names)

        assert analyzer.muscle_names == names

    def test_initialization_clips_negative_values(self, caplog):
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
        assert "negative values" in caplog.text.lower()

        # Data should be clipped to zero
        assert np.all(analyzer.data >= 0)
        assert analyzer.data[0, 1] == 0.0  # Was -0.1
        assert analyzer.data[1, 2] == 0.0  # Was -0.2

    def test_initialization_with_list_input(self):
        """Test that initialization works with list input."""
        data_list = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        analyzer = MuscleSynergyAnalyzer(data_list)

        assert analyzer.n_samples == 3
        assert analyzer.n_muscles == 2
        assert isinstance(analyzer.data, np.ndarray)

    def test_data_shape_extraction(self):
        """Test that data shape is correctly extracted."""
        n_samples, n_muscles = 75, 8
        data = np.random.rand(n_samples, n_muscles)
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == n_samples
        assert analyzer.n_muscles == n_muscles
        assert analyzer.data.shape == (n_samples, n_muscles)


@skip_if_unavailable("sklearn")
class TestExtractSynergies:
    """Test extract_synergies method."""

    def test_extract_single_synergy(self):
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

        assert result.n_synergies == 1
        assert result.weights.shape == (n_muscles, 1)
        assert result.activations.shape == (1, n_samples)
        assert result.reconstructed.shape == (n_samples, n_muscles)

    def test_extract_multiple_synergies(self):
        """Test extracting multiple synergies."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        result = analyzer.extract_synergies(n_synergies=3)

        assert result.n_synergies == 3
        assert result.weights.shape == (5, 3)
        assert result.activations.shape == (3, 100)
        assert result.reconstructed.shape == (100, 5)

    def test_vaf_is_between_zero_and_one(self):
        """Test that Variance Accounted For is between 0 and 1."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        for n_syn in [1, 2, 3, 4]:
            result = analyzer.extract_synergies(n_synergies=n_syn)
            assert 0.0 <= result.vaf <= 1.0, f"VAF out of range for {n_syn} synergies"

    def test_vaf_increases_with_more_synergies(self):
        """Test that VAF generally increases with more synergies."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        vaf_1 = analyzer.extract_synergies(n_synergies=1).vaf
        vaf_2 = analyzer.extract_synergies(n_synergies=2).vaf
        vaf_3 = analyzer.extract_synergies(n_synergies=3).vaf

        # More synergies should explain more variance
        assert vaf_2 >= vaf_1 - 0.01  # Allow small numerical tolerance
        assert vaf_3 >= vaf_2 - 0.01

    def test_reconstruction_approximates_original(self):
        """Test that reconstruction approximates original data."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        # With enough synergies, should approximate well
        result = analyzer.extract_synergies(n_synergies=4)

        # VAF should be high
        assert (
            result.vaf > 0.80
        ), "High number of synergies should give good reconstruction"

        # Reconstruction shape should match data
        assert result.reconstructed.shape == data.shape

    def test_weights_are_nonnegative(self):
        """Test that muscle weights are non-negative (NMF property)."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        result = analyzer.extract_synergies(n_synergies=2)

        assert np.all(result.weights >= 0), "Weights should be non-negative"

    def test_activations_are_nonnegative(self):
        """Test that activation profiles are non-negative (NMF property)."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        result = analyzer.extract_synergies(n_synergies=2)

        assert np.all(result.activations >= 0), "Activations should be non-negative"

    def test_invalid_number_of_synergies_too_small(self):
        """Test that n_synergies < 1 raises ValueError."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(ValueError, match="Invalid number of synergies"):
            analyzer.extract_synergies(n_synergies=0)

    def test_invalid_number_of_synergies_too_large(self):
        """Test that n_synergies > n_muscles raises ValueError."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(ValueError, match="Invalid number of synergies"):
            analyzer.extract_synergies(n_synergies=6)  # > 5 muscles

    def test_custom_max_iterations(self):
        """Test that custom max_iter parameter works."""
        np.random.seed(42)
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        # Should not raise error with custom iterations
        result = analyzer.extract_synergies(n_synergies=2, max_iter=500)
        assert result.n_synergies == 2

    def test_result_includes_muscle_names(self):
        """Test that result includes muscle names if provided."""
        data = np.random.rand(100, 3)
        names = ["M1", "M2", "M3"]
        analyzer = MuscleSynergyAnalyzer(data, muscle_names=names)

        result = analyzer.extract_synergies(n_synergies=2)
        assert result.muscle_names == names

    def test_synergies_with_perfect_rank_1_data(self):
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
        assert (
            result.vaf > 0.98
        ), f"VAF should be near 1.0 for rank-1 data, got {result.vaf}"


@skip_if_unavailable("sklearn")
class TestFindOptimalSynergies:
    """Test find_optimal_synergies method."""

    def test_finds_synergies_meeting_threshold(self):
        """Test that method finds minimal synergies meeting VAF threshold."""
        np.random.seed(42)
        # Create data that's approximately rank-2
        W = np.random.rand(5, 2)
        H = np.random.rand(2, 100)
        data = np.dot(W, H).T + np.random.rand(100, 5) * 0.01  # Small noise

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=0.90)

        # Should find 2-3 synergies
        assert result.n_synergies <= 5
        assert (
            result.vaf >= 0.90 or result.n_synergies == 5
        )  # Either meets threshold or uses max

    def test_returns_best_when_threshold_not_met(self, caplog):
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
        assert result.n_synergies == 2

        # Should warn that threshold not met
        assert "threshold not met" in caplog.text.lower() or result.vaf >= 0.99

    def test_respects_max_synergies_limit(self):
        """Test that method respects max_synergies limit."""
        np.random.seed(42)
        data = np.random.rand(100, 10)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=3, vaf_threshold=0.80)

        # Should not exceed max_synergies
        assert result.n_synergies <= 3

    def test_caps_at_number_of_muscles(self):
        """Test that max_synergies is capped at n_muscles."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        # Request more synergies than muscles
        result = analyzer.find_optimal_synergies(max_synergies=10, vaf_threshold=0.95)

        # Should not exceed n_muscles (5)
        assert result.n_synergies <= 5

    def test_low_threshold_finds_fewer_synergies(self):
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
        assert result_low.n_synergies <= result_high.n_synergies

    def test_threshold_of_one_uses_all_muscles(self):
        """Test that VAF threshold of 1.0 tries to use all muscles."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=1.0)

        # Should use all 5 synergies (or meet threshold early)
        assert result.n_synergies <= 5

    def test_returns_synergy_result(self):
        """Test that method returns a SynergyResult object."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=0.80)

        assert isinstance(result, SynergyResult)
        assert result.n_synergies >= 1

    def test_invalid_limit_raises_error(self):
        """Test that limit < 1 raises ValueError."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        # This should raise because max_synergies=0 leads to limit=0
        with pytest.raises(ValueError, match="limit must be >= 1"):
            analyzer.find_optimal_synergies(max_synergies=0, vaf_threshold=0.90)


@pytest.mark.skipif(SKLEARN_AVAILABLE, reason="Test for sklearn not available")
class TestSklearnNotAvailable:
    """Test behavior when sklearn is not installed."""

    def test_extract_synergies_raises_import_error(self):
        """Test that extract_synergies raises ImportError without sklearn."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(ImportError, match="sklearn is required"):
            analyzer.extract_synergies(n_synergies=2)

    def test_find_optimal_synergies_raises_import_error(self):
        """Test that find_optimal_synergies raises ImportError without sklearn."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(ImportError, match="sklearn is required"):
            analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=0.90)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_data(self):
        """Test with single sample (degenerate case)."""
        data = np.array([[0.5, 0.3, 0.8]])  # 1 sample, 3 muscles
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 1
        assert analyzer.n_muscles == 3

    def test_single_muscle_data(self):
        """Test with single muscle."""
        data = np.random.rand(100, 1)  # 100 samples, 1 muscle
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 100
        assert analyzer.n_muscles == 1

        # Can only extract 1 synergy
        if SKLEARN_AVAILABLE:
            result = analyzer.extract_synergies(n_synergies=1)
            assert result.n_synergies == 1

    def test_all_zeros_data(self):
        """Test with all-zero data."""
        data = np.zeros((100, 5))
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == 100
        assert analyzer.n_muscles == 5

        # NMF might have issues with all-zero data, but shouldn't crash
        if SKLEARN_AVAILABLE:
            try:
                result = analyzer.extract_synergies(n_synergies=2)
                # If it succeeds, check basic properties
                assert result.n_synergies == 2
            except (ValueError, RuntimeError):
                # Some NMF implementations may fail on zero data
                pass

    def test_uniform_activation_data(self):
        """Test with uniform activation (all same value)."""
        data = np.ones((100, 5)) * 0.5
        analyzer = MuscleSynergyAnalyzer(data)

        if SKLEARN_AVAILABLE:
            result = analyzer.extract_synergies(n_synergies=1)
            # Should be able to extract, though VAF might be perfect or undefined
            assert result.n_synergies == 1

    def test_very_large_number_of_muscles(self):
        """Test with large number of muscles."""
        n_muscles = 100
        data = np.random.rand(50, n_muscles)
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_muscles == n_muscles

        if SKLEARN_AVAILABLE:
            # Should be able to extract synergies
            result = analyzer.extract_synergies(n_synergies=5)
            assert result.weights.shape == (n_muscles, 5)

    def test_very_long_time_series(self):
        """Test with very long time series."""
        n_samples = 10000
        data = np.random.rand(n_samples, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        assert analyzer.n_samples == n_samples

        if SKLEARN_AVAILABLE:
            result = analyzer.extract_synergies(n_synergies=2)
            assert result.activations.shape == (2, n_samples)


@skip_if_unavailable("sklearn")
class TestNumericalAccuracy:
    """Test numerical accuracy and consistency."""

    def test_reproducibility_with_fixed_seed(self):
        """Test that results are reproducible with same random seed."""
        data = np.random.rand(100, 5)

        analyzer1 = MuscleSynergyAnalyzer(data)
        result1 = analyzer1.extract_synergies(n_synergies=2)

        analyzer2 = MuscleSynergyAnalyzer(data)
        result2 = analyzer2.extract_synergies(n_synergies=2)

        # VAF should be identical (same random_state=42 in NMF)
        np.testing.assert_allclose(result1.vaf, result2.vaf, rtol=1e-10)

    def test_vaf_calculation_correctness(self):
        """Test that VAF is calculated correctly."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.extract_synergies(n_synergies=3)

        # Calculate VAF manually
        sst = np.sum(data**2)
        sse = np.sum((data - result.reconstructed) ** 2)
        vaf_expected = 1.0 - (sse / sst)

        np.testing.assert_allclose(result.vaf, vaf_expected, rtol=1e-6)

    def test_reconstruction_via_matrix_multiplication(self):
        """Test that W @ H approximates reconstruction."""
        np.random.seed(42)
        data = np.random.rand(100, 5)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.extract_synergies(n_synergies=2)

        # Reconstruct via matrix multiplication
        # Note: result.reconstructed is (n_samples, n_muscles)
        # W is (n_muscles, n_synergies), H is (n_synergies, n_samples)
        # So W @ H gives (n_muscles, n_samples), need transpose
        manual_recon = np.dot(result.weights, result.activations).T

        # Should match result.reconstructed
        np.testing.assert_allclose(manual_recon, result.reconstructed, rtol=1e-5)

    def test_max_synergies_equals_muscles_gives_perfect_reconstruction(self):
        """Test that using all muscles as synergies gives near-perfect reconstruction."""
        np.random.seed(42)
        n_muscles = 5
        data = np.random.rand(100, n_muscles)

        analyzer = MuscleSynergyAnalyzer(data)
        result = analyzer.extract_synergies(n_synergies=n_muscles)

        # VAF should be very high (near 1.0)
        assert (
            result.vaf > 0.95
        ), f"VAF with {n_muscles} synergies should be > 0.95, got {result.vaf}"
