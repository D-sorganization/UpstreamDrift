"""Tests for src.shared.python.core.numerical_constants (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.core.numerical_constants import (
    CONDITION_NUMBER_CRITICAL_THRESHOLD,
    CONDITION_NUMBER_WARNING_THRESHOLD,
    EPSILON_FINITE_DIFF_JACOBIAN,
    EPSILON_MASS_MATRIX_REGULARIZATION,
    EPSILON_SINGULARITY_DETECTION,
    HUMAN_BODY_MASS_PLAUSIBLE_RANGE,
    SEGMENT_LENGTH_TO_HEIGHT_RATIO_PLAUSIBLE,
    TOLERANCE_ENERGY_CONSERVATION,
    TOLERANCE_WORK_ENERGY_MISMATCH,
)


class TestEpsilonConstants:
    def test_finite_diff_jacobian_is_positive(self) -> None:
        assert EPSILON_FINITE_DIFF_JACOBIAN > 0.0

    def test_finite_diff_jacobian_order_of_magnitude(self) -> None:
        # Should be around 1e-6 to 1e-7
        assert 1e-8 < EPSILON_FINITE_DIFF_JACOBIAN < 1e-4

    def test_singularity_detection_is_positive(self) -> None:
        assert EPSILON_SINGULARITY_DETECTION > 0.0

    def test_singularity_detection_smaller_than_finite_diff(self) -> None:
        # Singularity detection should be tighter
        assert EPSILON_SINGULARITY_DETECTION <= EPSILON_FINITE_DIFF_JACOBIAN

    def test_mass_matrix_regularization_is_positive(self) -> None:
        assert EPSILON_MASS_MATRIX_REGULARIZATION > 0.0

    def test_mass_matrix_regularization_small(self) -> None:
        assert EPSILON_MASS_MATRIX_REGULARIZATION < 1e-6


class TestToleranceConstants:
    def test_energy_conservation_is_positive(self) -> None:
        assert TOLERANCE_ENERGY_CONSERVATION > 0.0

    def test_energy_conservation_is_tight(self) -> None:
        # Should be reasonably tight
        assert TOLERANCE_ENERGY_CONSERVATION < 0.01

    def test_work_energy_mismatch_is_positive(self) -> None:
        assert TOLERANCE_WORK_ENERGY_MISMATCH > 0.0

    def test_work_energy_mismatch_is_percentage(self) -> None:
        # Stored as a fraction (0.05 = 5%)
        assert 0.0 < TOLERANCE_WORK_ENERGY_MISMATCH < 1.0


class TestConditionNumberThresholds:
    def test_warning_is_positive(self) -> None:
        assert CONDITION_NUMBER_WARNING_THRESHOLD > 0.0

    def test_critical_is_positive(self) -> None:
        assert CONDITION_NUMBER_CRITICAL_THRESHOLD > 0.0

    def test_critical_greater_than_warning(self) -> None:
        assert CONDITION_NUMBER_CRITICAL_THRESHOLD > CONDITION_NUMBER_WARNING_THRESHOLD


class TestHumanBodyConstants:
    def test_mass_range_is_tuple_of_two(self) -> None:
        assert len(HUMAN_BODY_MASS_PLAUSIBLE_RANGE) == 2

    def test_mass_range_min_positive(self) -> None:
        min_mass, _ = HUMAN_BODY_MASS_PLAUSIBLE_RANGE
        assert min_mass > 0.0

    def test_mass_range_max_gt_min(self) -> None:
        min_mass, max_mass = HUMAN_BODY_MASS_PLAUSIBLE_RANGE
        assert max_mass > min_mass

    def test_segment_ratios_is_dict(self) -> None:
        assert isinstance(SEGMENT_LENGTH_TO_HEIGHT_RATIO_PLAUSIBLE, dict)

    def test_segment_ratios_non_empty(self) -> None:
        assert len(SEGMENT_LENGTH_TO_HEIGHT_RATIO_PLAUSIBLE) > 0

    def test_segment_ratio_values_are_positive(self) -> None:
        for key, val in SEGMENT_LENGTH_TO_HEIGHT_RATIO_PLAUSIBLE.items():
            if isinstance(val, (int, float)):
                assert val > 0.0, f"Ratio for '{key}' should be positive"
            elif isinstance(val, (list, tuple)):
                assert all(v > 0.0 for v in val), (
                    f"All ratios for '{key}' should be positive"
                )
