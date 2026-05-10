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

    def test_muscle_analysis_initialization(self) -> None:
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
        assert (
            result.activations is activations
        ), "Assertion failed: result.activations is activations"
        assert (
            result.reconstructed is reconstructed
        ), "Assertion failed: result.reconstructed is reconstructed"
        assert result.vaf == 0.92, "Assertion failed: result.vaf == 0.92"
        assert result.n_synergies == 3, "Assertion failed: result.n_synergies == 3"
        assert (
            result.muscle_names is None
        ), "Assertion failed: result.muscle_names is None"

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

        assert (
            result.muscle_names == muscle_names
        ), "Assertion failed: result.muscle_names == muscle_names"

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

        assert result.weights.shape == (
            n_muscles,
            n_synergies,
        ), "Assertion failed: result.weights.shape == (n_muscles, n_synergies)"
        assert result.activations.shape == (
            n_synergies,
            n_samples,
        ), "Assertion failed: result.activations.shape == (n_synergies, n_samples)"
        assert result.reconstructed.shape == (
            n_samples,
            n_muscles,
        ), "Assertion failed: result.reconstructed.shape == (n_samples, n_muscles)"
