"""Value-asserting tests for pressure-drop flow calculations (#6998).

Replaces smoke-only coverage of
``sidekick.process_calculators.pressure_drop_calculator.engine._flow_calculations``
with assertions against hand-computed / textbook reference values
(Darcy-Weisbach, regime transitions, hydrostatic head, API RP 14E
erosional velocity) plus Design-by-Contract (negatives -> ValueError).
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.sidekick.process_calculators.pressure_drop_calculator.engine._flow_calculations import (  # noqa: E501
    calculate_elevation_pressure_drop,
    calculate_erosional_velocity,
    calculate_expansion_factor,
    calculate_frictional_pressure_drop,
    classify_flow_regime,
)

# Reference constants (mirror sidekick.process_calculators.constants).
GRAVITY = 9.80665  # m/s^2
FT_S_TO_M_S = 0.3048
KG_M3_TO_LB_FT3 = 0.062428
API_C_CONTINUOUS = 100.0
API_C_INTERMITTENT = 125.0


# ---------------------------------------------------------------------------
# classify_flow_regime: Re=2300 / 4000 boundaries
# ---------------------------------------------------------------------------
class TestClassifyFlowRegime:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("reynolds", "expected"),
        [
            (0.0, "laminar"),
            (1000.0, "laminar"),
            (2299.9, "laminar"),
            (2300.0, "transitional"),  # boundary: NOT laminar (strict <)
            (3000.0, "transitional"),
            (3999.9, "transitional"),
            (4000.0, "turbulent"),  # boundary: NOT transitional (strict <)
            (10000.0, "turbulent"),
            (1e6, "turbulent"),
        ],
    )
    def test_regime_boundaries(self, reynolds: float, expected: str) -> None:
        assert classify_flow_regime(reynolds) == expected


# ---------------------------------------------------------------------------
# calculate_frictional_pressure_drop: Darcy-Weisbach
# ---------------------------------------------------------------------------
class TestFrictionalPressureDrop:
    @pytest.mark.unit
    def test_darcy_weisbach_hand_calc(self) -> None:
        # dP = f*(L/D)*(rho*V^2/2)
        #    = 0.02*(100/0.1)*(0.5*1000*2^2) = 0.02*1000*2000 = 40000 Pa
        dp = calculate_frictional_pressure_drop(
            friction_factor=0.02,
            length=100.0,
            diameter=0.1,
            density=1000.0,
            velocity=2.0,
        )
        assert dp == pytest.approx(40000.0, rel=1e-12)

    @pytest.mark.unit
    def test_scales_linearly_with_length(self) -> None:
        base = calculate_frictional_pressure_drop(0.02, 50.0, 0.1, 1000.0, 2.0)
        dbl = calculate_frictional_pressure_drop(0.02, 100.0, 0.1, 1000.0, 2.0)
        assert dbl == pytest.approx(2.0 * base, rel=1e-12)

    @pytest.mark.unit
    def test_scales_with_velocity_squared(self) -> None:
        v1 = calculate_frictional_pressure_drop(0.02, 100.0, 0.1, 1000.0, 2.0)
        v2 = calculate_frictional_pressure_drop(0.02, 100.0, 0.1, 1000.0, 4.0)
        assert v2 == pytest.approx(4.0 * v1, rel=1e-12)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"friction_factor": 0.0},
            {"friction_factor": -0.02},
            {"length": 0.0},
            {"length": -1.0},
            {"diameter": 0.0},
            {"diameter": -0.1},
            {"density": 0.0},
            {"density": -1.0},
            {"velocity": 0.0},
            {"velocity": -2.0},
        ],
    )
    def test_nonpositive_inputs_raise(self, kwargs: dict[str, float]) -> None:
        base = {
            "friction_factor": 0.02,
            "length": 100.0,
            "diameter": 0.1,
            "density": 1000.0,
            "velocity": 2.0,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            calculate_frictional_pressure_drop(**base)


# ---------------------------------------------------------------------------
# calculate_elevation_pressure_drop: hydrostatic head + sign convention
# ---------------------------------------------------------------------------
class TestElevationPressureDrop:
    @pytest.mark.unit
    def test_rise_positive_loss(self) -> None:
        # rho*g*h = 1000 * 9.80665 * 10 = 98066.5 Pa (loss)
        dp = calculate_elevation_pressure_drop(1000.0, 10.0)
        assert dp == pytest.approx(1000.0 * GRAVITY * 10.0, rel=1e-9)
        assert dp > 0

    @pytest.mark.unit
    def test_drop_negative_gain(self) -> None:
        dp = calculate_elevation_pressure_drop(1000.0, -10.0)
        assert dp == pytest.approx(-1000.0 * GRAVITY * 10.0, rel=1e-9)
        assert dp < 0

    @pytest.mark.unit
    def test_zero_elevation(self) -> None:
        assert calculate_elevation_pressure_drop(1000.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_erosional_velocity: API RP 14E  V = C / sqrt(rho)
# ---------------------------------------------------------------------------
class TestErosionalVelocity:
    @staticmethod
    def _reference(density: float, c_value: float) -> float:
        v_ft = c_value / math.sqrt(density * KG_M3_TO_LB_FT3)
        return v_ft * FT_S_TO_M_S

    @pytest.mark.unit
    def test_continuous_matches_api_formula(self) -> None:
        v = calculate_erosional_velocity(50.0, "continuous")
        assert v == pytest.approx(self._reference(50.0, API_C_CONTINUOUS), rel=1e-9)

    @pytest.mark.unit
    def test_intermittent_uses_higher_c(self) -> None:
        v = calculate_erosional_velocity(50.0, "intermittent")
        assert v == pytest.approx(self._reference(50.0, API_C_INTERMITTENT), rel=1e-9)
        # Higher C -> higher limit than continuous.
        assert v > calculate_erosional_velocity(50.0, "continuous")

    @pytest.mark.unit
    def test_unknown_service_defaults_to_continuous(self) -> None:
        v = calculate_erosional_velocity(50.0, "mystery")
        assert v == pytest.approx(calculate_erosional_velocity(50.0, "continuous"))

    @pytest.mark.unit
    def test_decreases_with_density(self) -> None:
        # V ~ 1/sqrt(rho): quadrupling density halves the velocity limit.
        v_low = calculate_erosional_velocity(25.0, "continuous")
        v_high = calculate_erosional_velocity(100.0, "continuous")
        assert v_high == pytest.approx(v_low / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# calculate_expansion_factor: Y bounds and limiting regimes
# ---------------------------------------------------------------------------
class TestExpansionFactor:
    @pytest.mark.unit
    def test_negligible_drop_is_unity(self) -> None:
        # Very small dP/P -> nearly incompressible -> Y == 1.
        y = calculate_expansion_factor(1e6, 100.0, 0.02, 100.0)
        assert y == pytest.approx(1.0)

    @pytest.mark.unit
    def test_choked_flow_returns_zero(self) -> None:
        # dP >= P1 -> outlet pressure <= 0 -> choked -> Y == 0.
        assert calculate_expansion_factor(1e6, 2e6, 0.02, 100.0) == 0.0

    @pytest.mark.unit
    def test_intermediate_drop_in_unit_interval(self) -> None:
        y = calculate_expansion_factor(1e6, 3e5, 0.02, 100.0)
        assert 0.0 < y <= 1.0

    @pytest.mark.unit
    def test_nonpositive_inlet_returns_unity(self) -> None:
        # Documented guard: invalid inlet -> safe Y = 1.0 (no correction).
        assert calculate_expansion_factor(0.0, 100.0, 0.02, 100.0) == 1.0
        assert calculate_expansion_factor(-1.0, 100.0, 0.02, 100.0) == 1.0
