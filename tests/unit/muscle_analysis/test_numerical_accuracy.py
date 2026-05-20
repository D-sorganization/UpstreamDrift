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
class TestNumericalAccuracy:
    """Test numerical accuracy and consistency."""

    def test_reproducibility_with_fixed_seed(self) -> None:
        """Test that results are reproducible with same random seed."""
        data = np.random.rand(100, 5)

        analyzer1 = MuscleSynergyAnalyzer(data)
        result1 = analyzer1.extract_synergies(n_synergies=2)

        analyzer2 = MuscleSynergyAnalyzer(data)
        result2 = analyzer2.extract_synergies(n_synergies=2)

        # VAF should be identical (same random_state=42 in NMF)
        np.testing.assert_allclose(result1.vaf, result2.vaf, rtol=1e-10)

    def test_vaf_calculation_correctness(self) -> None:
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

    def test_reconstruction_via_matrix_multiplication(self) -> None:
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

    def test_max_synergies_equals_muscles_gives_perfect_reconstruction(self) -> None:
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
