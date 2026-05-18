"""Tests for pressure_drop_calculator fitting_loss_coefficients (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator.utils.fitting_loss_coefficients import (
    FITTING_K_FACTORS,
    equivalent_length_to_k,
    get_fitting_k_factor,
    get_multiple_fittings_k,
    k_to_equivalent_length,
    list_available_fittings,
)


class TestFittingKFactorsDict:
    def test_is_nonempty(self) -> None:
        assert len(FITTING_K_FACTORS) > 0

    def test_90_elbow_std_exists(self) -> None:
        assert "90_elbow_std" in FITTING_K_FACTORS

    def test_gate_valve_open_exists(self) -> None:
        assert "gate_valve_open" in FITTING_K_FACTORS

    def test_ball_valve_open_exists(self) -> None:
        assert "ball_valve_open" in FITTING_K_FACTORS

    def test_all_values_positive(self) -> None:
        assert all(v > 0 for v in FITTING_K_FACTORS.values())

    def test_90_elbow_std_reasonable(self) -> None:
        # K-factors for 90° elbows are typically 0.2–1.5
        k = FITTING_K_FACTORS["90_elbow_std"]
        assert 0.1 <= k <= 5.0


class TestGetFittingKFactor:
    def test_fitting_loss_coefficients_returns_float(self) -> None:
        result = get_fitting_k_factor("90_elbow_std")
        assert isinstance(result, float)

    def test_known_value(self) -> None:
        k = get_fitting_k_factor("90_elbow_std")
        assert k == pytest.approx(FITTING_K_FACTORS["90_elbow_std"])

    def test_unknown_fitting_raises(self) -> None:
        with pytest.raises(ValueError):
            get_fitting_k_factor("nonexistent_fitting_type")

    def test_ball_valve_low_resistance(self) -> None:
        k = get_fitting_k_factor("ball_valve_open")
        # Ball valves are low resistance when open
        assert k < 1.0

    def test_globe_valve_high_resistance(self) -> None:
        k = get_fitting_k_factor("globe_valve_open")
        # Globe valves are high resistance
        assert k > 1.0


class TestGetMultipleFittingsK:
    def test_single_fitting(self) -> None:
        result = get_multiple_fittings_k({"90_elbow_std": 1})
        assert result == pytest.approx(FITTING_K_FACTORS["90_elbow_std"])

    def test_multiple_of_same(self) -> None:
        k_single = get_fitting_k_factor("90_elbow_std")
        result = get_multiple_fittings_k({"90_elbow_std": 3})
        assert result == pytest.approx(k_single * 3)

    def test_multiple_types(self) -> None:
        k_elbow = get_fitting_k_factor("90_elbow_std")
        k_ball = get_fitting_k_factor("ball_valve_open")
        result = get_multiple_fittings_k({"90_elbow_std": 1, "ball_valve_open": 1})
        assert result == pytest.approx(k_elbow + k_ball)

    def test_empty_fittings_returns_zero(self) -> None:
        result = get_multiple_fittings_k({})
        assert result == pytest.approx(0.0)


class TestKToEquivalentLength:
    def test_fitting_loss_coefficients_basic_conversion(self) -> None:
        result = k_to_equivalent_length(1.0, 0.02)
        assert result == pytest.approx(50.0)

    def test_zero_friction_factor_raises(self) -> None:
        with pytest.raises(ValueError):
            k_to_equivalent_length(1.0, 0.0)

    def test_negative_friction_factor_raises(self) -> None:
        with pytest.raises(ValueError):
            k_to_equivalent_length(1.0, -0.01)

    def test_larger_k_gives_larger_length(self) -> None:
        l1 = k_to_equivalent_length(1.0, 0.02)
        l2 = k_to_equivalent_length(2.0, 0.02)
        assert l2 > l1


class TestEquivalentLengthToK:
    def test_fitting_loss_coefficients_basic_conversion(self) -> None:
        result = equivalent_length_to_k(50.0, 0.02)
        assert result == pytest.approx(1.0)

    def test_zero_friction_factor_raises(self) -> None:
        with pytest.raises(ValueError):
            equivalent_length_to_k(30.0, 0.0)

    def test_fitting_loss_coefficients_roundtrip(self) -> None:
        k_original = 0.75
        f = 0.02
        l_d = k_to_equivalent_length(k_original, f)
        k_back = equivalent_length_to_k(l_d, f)
        assert k_back == pytest.approx(k_original)


class TestListAvailableFittings:
    def test_fitting_loss_coefficients_returns_dict(self) -> None:
        result = list_available_fittings()
        assert isinstance(result, dict)

    def test_fitting_loss_coefficients_nonempty(self) -> None:
        assert len(list_available_fittings()) > 0

    def test_matches_fitting_k_factors(self) -> None:
        result = list_available_fittings()
        assert set(result.keys()) == set(FITTING_K_FACTORS.keys())
