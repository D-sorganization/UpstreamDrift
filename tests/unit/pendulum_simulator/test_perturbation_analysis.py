"""Tests for src.shared.python.pendulum_simulator.perturbation_analysis (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_coeffs,
    variability_summary,
)


class TestPerturbTorqueCoeffs:
    def test_zero_amplitude_no_change(self) -> None:
        coeffs = [[5.0, 2.0], [3.0]]
        result = perturb_torque_coeffs(coeffs, noise_amplitude=0.0)
        assert result == coeffs

    def test_returns_list_of_lists(self) -> None:
        coeffs = [[1.0], [2.0]]
        result = perturb_torque_coeffs(coeffs, noise_amplitude=0.1)
        assert isinstance(result, list)
        assert all(isinstance(joint, list) for joint in result)

    def test_preserves_structure(self) -> None:
        coeffs = [[1.0, 2.0, 3.0], [4.0, 5.0]]
        result = perturb_torque_coeffs(coeffs, noise_amplitude=0.1, seed=42)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 2

    def test_with_seed_reproducible(self) -> None:
        coeffs = [[10.0, 2.0], [5.0]]
        r1 = perturb_torque_coeffs(coeffs, 1.0, seed=0)
        r2 = perturb_torque_coeffs(coeffs, 1.0, seed=0)
        assert r1 == r2

    def test_different_seeds_different_results(self) -> None:
        coeffs = [[10.0, 2.0], [5.0]]
        r1 = perturb_torque_coeffs(coeffs, 1.0, seed=0)
        r2 = perturb_torque_coeffs(coeffs, 1.0, seed=99)
        assert r1 != r2

    def test_noise_type_white(self) -> None:
        coeffs = [[5.0]]
        result = perturb_torque_coeffs(coeffs, 0.5, noise_type="white", seed=0)
        assert len(result) == 1

    def test_noise_type_pink(self) -> None:
        coeffs = [[5.0]]
        result = perturb_torque_coeffs(coeffs, 0.5, noise_type="pink", seed=0)
        assert len(result) == 1

    def test_perturb_mode_additive(self) -> None:
        coeffs = [[10.0]]
        result = perturb_torque_coeffs(coeffs, 0.1, perturb_mode="additive", seed=0)
        # Should differ from original if amplitude > 0
        assert isinstance(result[0][0], float)

    def test_invalid_noise_type_raises(self) -> None:
        with pytest.raises(AssertionError):
            perturb_torque_coeffs([[1.0]], 0.1, noise_type="invalid")

    def test_invalid_perturb_mode_raises(self) -> None:
        with pytest.raises(AssertionError):
            perturb_torque_coeffs([[1.0]], 0.1, perturb_mode="bad_mode")


class TestVariabilitySummary:
    def _make_results(self, n: int = 5) -> list[dict]:
        rng = np.random.default_rng(0)
        return [
            {
                "tip_speed_final": float(5.0 + rng.normal() * 0.1),
                "tip_position_final": np.array([rng.normal(), rng.normal()]),
            }
            for _ in range(n)
        ]

    def test_perturbation_analysis_returns_dict(self) -> None:
        result = variability_summary(self._make_results())
        assert isinstance(result, dict)

    def test_has_speed_keys(self) -> None:
        result = variability_summary(self._make_results())
        assert "tip_speed_mean" in result
        assert "tip_speed_std" in result
        assert "tip_speed_cv" in result

    def test_n_trials_matches(self) -> None:
        results = self._make_results(10)
        summary = variability_summary(results)
        assert summary["n_trials"] == 10

    def test_speed_mean_positive(self) -> None:
        result = variability_summary(self._make_results())
        assert result["tip_speed_mean"] > 0.0

    def test_speed_std_non_negative(self) -> None:
        result = variability_summary(self._make_results())
        assert result["tip_speed_std"] >= 0.0

    def test_speed_cv_non_negative(self) -> None:
        result = variability_summary(self._make_results())
        assert result["tip_speed_cv"] >= 0.0

    def test_empty_results_raises(self) -> None:
        with pytest.raises(AssertionError):
            variability_summary([])
