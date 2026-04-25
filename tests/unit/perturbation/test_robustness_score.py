"""Tests for src.shared.python.perturbation.robustness_score (Issues #1949, #1744)."""

from __future__ import annotations

import pytest

from src.shared.python.perturbation.robustness_score import compute_robustness_score

# ---------------------------------------------------------------------------
# compute_robustness_score
# ---------------------------------------------------------------------------


class TestComputeRobustnessScore:
    def test_zero_cv_gives_one(self) -> None:
        assert compute_robustness_score(0.0) == 1.0

    def test_result_between_zero_and_one(self) -> None:
        result = compute_robustness_score(0.5)
        assert 0.0 <= result <= 1.0

    def test_large_cv_approaches_zero(self) -> None:
        result = compute_robustness_score(1e6)
        assert result < 0.001

    def test_cv_one_gives_half(self) -> None:
        result = compute_robustness_score(1.0)
        assert abs(result - 0.5) < 1e-10

    def test_monotone_decreasing(self) -> None:
        # Higher CV → lower robustness score
        assert compute_robustness_score(0.1) > compute_robustness_score(0.5)
        assert compute_robustness_score(0.5) > compute_robustness_score(2.0)

    def test_formula_correct(self) -> None:
        cv = 3.0
        expected = 1.0 / (1.0 + cv)
        assert abs(compute_robustness_score(cv) - expected) < 1e-12

    def test_negative_cv_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            compute_robustness_score(-0.1)
