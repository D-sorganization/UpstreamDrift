"""Tests for sidekick.process_calculators.pressure_drop_calculator.engine (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator.engine.pressure_drop_calculation_engine import (
    calculate_elevation_pressure_drop,
    calculate_erosional_velocity,
    classify_flow_regime,
    friction_factor_colebrook,
    friction_factor_haaland,
    friction_factor_laminar,
    friction_factor_swamee_jain,
)


class TestFrictionFactorLaminar:
    def test_typical_value(self) -> None:
        # For laminar flow, f = 64/Re
        f = friction_factor_laminar(reynolds_number=1000.0)
        assert f == pytest.approx(64.0 / 1000.0)

    def test_lower_re_higher_f(self) -> None:
        f_low = friction_factor_laminar(100.0)
        f_high = friction_factor_laminar(2000.0)
        assert f_low > f_high

    def test_positive(self) -> None:
        assert friction_factor_laminar(500.0) > 0.0


class TestFrictionFactorColebrook:
    def test_typical_turbulent_flow(self) -> None:
        f = friction_factor_colebrook(
            reynolds_number=100000.0, relative_roughness=0.0001
        )
        assert f > 0.0

    def test_smooth_pipe(self) -> None:
        f = friction_factor_colebrook(reynolds_number=100000.0, relative_roughness=0.0)
        assert f > 0.0

    def test_rougher_pipe_higher_f(self) -> None:
        f_smooth = friction_factor_colebrook(100000.0, 0.0)
        f_rough = friction_factor_colebrook(100000.0, 0.01)
        assert f_rough >= f_smooth


class TestFrictionFactorSwameeJain:
    def test_positive(self) -> None:
        f = friction_factor_swamee_jain(
            reynolds_number=100000.0, relative_roughness=0.001
        )
        assert f > 0.0

    def test_similar_to_colebrook(self) -> None:
        # Swamee-Jain is an approximation to Colebrook-White
        Re = 100000.0
        rr = 0.001
        f_sj = friction_factor_swamee_jain(Re, rr)
        f_cb = friction_factor_colebrook(Re, rr)
        assert abs(f_sj - f_cb) / f_cb < 0.05  # within 5%


class TestFrictionFactorHaaland:
    def test_positive(self) -> None:
        f = friction_factor_haaland(reynolds_number=100000.0, relative_roughness=0.001)
        assert f > 0.0


class TestClassifyFlowRegime:
    def test_laminar_low_re(self) -> None:
        regime = classify_flow_regime(500.0)
        assert "laminar" in regime.lower()

    def test_turbulent_high_re(self) -> None:
        regime = classify_flow_regime(100000.0)
        assert "turbulent" in regime.lower()

    def test_pressure_drop_engine_returns_string(self) -> None:
        regime = classify_flow_regime(2300.0)
        assert isinstance(regime, str)


class TestCalculateElevationPressureDrop:
    def test_positive_elevation_increase(self) -> None:
        # Going up → pressure drop positive
        dp = calculate_elevation_pressure_drop(density=1.2, elevation_change=10.0)
        assert dp > 0.0

    def test_zero_elevation_zero_dp(self) -> None:
        dp = calculate_elevation_pressure_drop(density=1.2, elevation_change=0.0)
        assert dp == pytest.approx(0.0)

    def test_negative_elevation_negative_dp(self) -> None:
        dp = calculate_elevation_pressure_drop(density=1.2, elevation_change=-10.0)
        assert dp < 0.0


class TestCalculateErosionalVelocity:
    def test_positive_velocity(self) -> None:
        v = calculate_erosional_velocity(density=1.2)
        assert v > 0.0

    def test_pressure_drop_engine_returns_float(self) -> None:
        v = calculate_erosional_velocity(density=10.0)
        assert isinstance(v, float)

    def test_higher_density_lower_velocity(self) -> None:
        v_low = calculate_erosional_velocity(density=1.0)
        v_high = calculate_erosional_velocity(density=100.0)
        assert v_low > v_high
