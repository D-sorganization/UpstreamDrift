"""Tests for src.shared.python.biomechanics.muscle_analysis (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "sklearn", reason="scikit-learn is required for muscle synergy analysis"
)

from src.shared.python.biomechanics.muscle_analysis import (  # noqa: E402
    MuscleSynergyAnalyzer,
    SynergyResult,
)


def _make_data(n_samples: int = 100, n_muscles: int = 4) -> np.ndarray:
    """Create synthetic non-negative activation data."""
    rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal((n_samples, n_muscles)))


class TestMuscleSynergyAnalyzerConstruction:
    def test_muscle_analysis_valid_construction(self) -> None:
        data = _make_data(100, 4)
        analyzer = MuscleSynergyAnalyzer(data)
        assert analyzer.n_muscles == 4
        assert analyzer.n_samples == 100

    def test_stores_muscle_names(self) -> None:
        data = _make_data(50, 3)
        names = ["Bicep", "Tricep", "Deltoid"]
        analyzer = MuscleSynergyAnalyzer(data, muscle_names=names)
        assert analyzer.muscle_names == names

    def test_default_muscle_names_generated(self) -> None:
        data = _make_data(50, 3)
        analyzer = MuscleSynergyAnalyzer(data)
        assert len(analyzer.muscle_names) == 3
        assert all("Muscle" in name for name in analyzer.muscle_names)

    def test_1d_data_raises(self) -> None:
        data = np.ones(50)
        with pytest.raises((ValueError, TypeError, AssertionError)):
            MuscleSynergyAnalyzer(data)

    def test_empty_data_raises(self) -> None:
        data = np.zeros((0, 4))
        with pytest.raises((ValueError, TypeError, AssertionError)):
            MuscleSynergyAnalyzer(data)

    def test_negative_values_are_clipped(self) -> None:
        data = np.array([[-1.0, 2.0], [3.0, -0.5], [0.5, 1.0]])
        analyzer = MuscleSynergyAnalyzer(data)
        assert np.all(analyzer.data >= 0.0)


class TestExtractSynergies:
    def setup_method(self) -> None:
        self.data = _make_data(n_samples=100, n_muscles=5)
        self.analyzer = MuscleSynergyAnalyzer(self.data)

    def test_returns_synergy_result(self) -> None:
        result = self.analyzer.extract_synergies(n_synergies=2)
        assert isinstance(result, SynergyResult)

    def test_n_synergies_stored(self) -> None:
        result = self.analyzer.extract_synergies(n_synergies=3)
        assert result.n_synergies == 3

    def test_weights_shape(self) -> None:
        result = self.analyzer.extract_synergies(n_synergies=2)
        # W: (n_muscles, n_synergies)
        assert result.weights.shape == (5, 2)

    def test_activations_shape(self) -> None:
        result = self.analyzer.extract_synergies(n_synergies=2)
        # H: (n_synergies, n_samples)
        assert result.activations.shape == (2, 100)

    def test_reconstructed_shape(self) -> None:
        result = self.analyzer.extract_synergies(n_synergies=2)
        assert result.reconstructed.shape == (100, 5)

    def test_vaf_in_unit_interval(self) -> None:
        result = self.analyzer.extract_synergies(n_synergies=2)
        assert 0.0 <= result.vaf <= 1.0 + 1e-6

    def test_more_synergies_higher_vaf(self) -> None:
        r2 = self.analyzer.extract_synergies(n_synergies=2)
        r4 = self.analyzer.extract_synergies(n_synergies=4)
        # More synergies should explain at least as much variance
        assert r4.vaf >= r2.vaf - 0.05  # Allow small tolerance

    def test_too_many_synergies_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, AssertionError)):
            self.analyzer.extract_synergies(n_synergies=10)  # > n_muscles=5

    def test_zero_synergies_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, AssertionError)):
            self.analyzer.extract_synergies(n_synergies=0)


class TestFindOptimalSynergies:
    def setup_method(self) -> None:
        # Use data that can be well-reconstructed
        rng = np.random.default_rng(0)
        # Create data from 2 true synergies
        W = np.abs(rng.standard_normal((4, 2)))
        H = np.abs(rng.standard_normal((2, 100)))
        self.data = (W @ H).T  # (100, 4)
        self.analyzer = MuscleSynergyAnalyzer(self.data)

    def test_returns_synergy_result(self) -> None:
        result = self.analyzer.find_optimal_synergies(max_synergies=4)
        assert isinstance(result, SynergyResult)

    def test_result_meets_threshold(self) -> None:
        result = self.analyzer.find_optimal_synergies(
            max_synergies=4, vaf_threshold=0.5
        )
        assert result.vaf >= 0.5 - 0.01  # Allow tolerance

    def test_result_n_synergies_in_range(self) -> None:
        result = self.analyzer.find_optimal_synergies(max_synergies=3)
        assert 1 <= result.n_synergies <= 3
