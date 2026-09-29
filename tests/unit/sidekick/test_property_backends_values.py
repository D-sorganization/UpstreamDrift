"""Value-asserting tests for thermo property backends (#7003).

Covers ``sidekick.calculators.thermo._property_backends``: input validation,
phase / quality determination, simplified property correlations, and the
Antoine-based saturation pressure/temperature path. CoolProp- and
Cantera-specific code paths are guarded with skips when those optional
libraries are absent.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.sidekick.calculators.thermo import _property_backends as pb
from src.shared.python.sidekick.calculators.thermo._constants import (
    ANTOINE_A,
    ANTOINE_B,
    ANTOINE_C_KELVIN,
    MMHG_TO_PASCAL_FACTOR,
    SPECIFIC_GAS_CONSTANT_WATER,
    TRIPLE_POINT_PRESSURE,
    TRIPLE_POINT_TEMPERATURE,
)


def _antoine_psat_pa(temperature: float) -> float:
    """Reference Antoine saturation pressure (Pa)."""
    log_p_mmhg = ANTOINE_A - ANTOINE_B / (temperature - ANTOINE_C_KELVIN)
    return 10**log_p_mmhg * MMHG_TO_PASCAL_FACTOR


class _MockWater:
    """Minimal Cantera-like water surrogate for phase determination.

    Setting ``TQ`` records the requested temperature and exposes a fixed
    saturation pressure via ``P`` so the phase branches can be exercised
    deterministically without Cantera installed.
    """

    def __init__(self, p_sat: float) -> None:
        self._p_sat = p_sat
        self.P = p_sat

    @property
    def TQ(self):  # noqa: ANN201 - mimic Cantera attribute
        return None

    @TQ.setter
    def TQ(self, value) -> None:  # noqa: ANN001
        self.P = self._p_sat


# ---------------------------------------------------------------------------
# validate_coolprop_inputs: rejects out-of-range T / P
# ---------------------------------------------------------------------------
class TestValidateCoolpropInputs:
    @pytest.mark.unit
    def test_valid_state_passes(self) -> None:
        # 400 K, 1 atm is inside the accepted window -> no raise.
        pb.validate_coolprop_inputs(400.0, 101325.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("temperature", [0.0, -10.0, 200.0, 1500.0])
    def test_bad_temperature_raises(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="Temperature"):
            pb.validate_coolprop_inputs(temperature, 101325.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("pressure", [0.0, -1.0, 1.0, 500e6])
    def test_bad_pressure_raises(self, pressure: float) -> None:
        with pytest.raises(ValueError, match="Pressure"):
            pb.validate_coolprop_inputs(400.0, pressure)

    @pytest.mark.unit
    def test_triple_point_bounds_inclusive(self) -> None:
        # Exactly at the lower bounds must be accepted.
        pb.validate_coolprop_inputs(TRIPLE_POINT_TEMPERATURE, TRIPLE_POINT_PRESSURE)


# ---------------------------------------------------------------------------
# determine_phase_and_quality: subcooled / two-phase / superheated / super-crit
# ---------------------------------------------------------------------------
class TestDeterminePhaseAndQuality:
    @pytest.mark.unit
    def test_supercritical_above_critical_temperature(self) -> None:
        phase, quality = pb.determine_phase_and_quality(_MockWater(1e5), 700.0, 1e5)
        assert phase == "supercritical"
        assert quality == pytest.approx(1.0)

    @pytest.mark.unit
    def test_supercritical_above_critical_pressure(self) -> None:
        phase, _ = pb.determine_phase_and_quality(_MockWater(1e5), 500.0, 30e6)
        assert phase == "supercritical"

    @pytest.mark.unit
    def test_subcooled_liquid_when_pressure_above_psat(self) -> None:
        # P (2e5) > P_sat (1e5) at T -> compressed/subcooled liquid.
        phase, quality = pb.determine_phase_and_quality(_MockWater(1e5), 350.0, 2e5)
        assert phase == "liquid"
        assert quality == pytest.approx(0.0)

    @pytest.mark.unit
    def test_two_phase_at_saturation(self) -> None:
        phase, quality = pb.determine_phase_and_quality(_MockWater(1e5), 350.0, 1e5)
        assert phase == "two-phase"
        assert 0.0 <= quality <= 1.0

    @pytest.mark.unit
    def test_superheated_vapor_when_pressure_below_psat(self) -> None:
        phase, quality = pb.determine_phase_and_quality(_MockWater(1e5), 350.0, 5e4)
        assert phase == "vapor"
        assert quality == pytest.approx(1.0)

    @pytest.mark.unit
    def test_non_cantera_object_returns_unknown(self) -> None:
        # Regression: a non-Cantera object raises AttributeError on .TQ; the
        # function must swallow it and report 'unknown' rather than propagate.
        phase, quality = pb.determine_phase_and_quality(None, 400.0, 1e5)
        assert phase == "unknown"
        assert quality == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_simplified_properties: ideal-gas vapor + liquid branch
# ---------------------------------------------------------------------------
class TestSimplifiedProperties:
    @pytest.mark.unit
    def test_liquid_branch_below_boiling(self) -> None:
        props = pb.calculate_simplified_properties(300.0, 1e5)
        assert props.phase == "liquid"
        assert props.quality == pytest.approx(0.0)
        assert props.density == pytest.approx(1000.0)

    @pytest.mark.unit
    def test_vapor_branch_ideal_gas_density(self) -> None:
        # rho = P / (R_water * T) for the vapor branch.
        t, p = 400.0, 1e5
        props = pb.calculate_simplified_properties(t, p)
        assert props.phase == "vapor"
        assert props.quality == pytest.approx(1.0)
        expected_rho = p / (SPECIFIC_GAS_CONSTANT_WATER * t)
        assert props.density == pytest.approx(expected_rho, rel=1e-9)

    @pytest.mark.unit
    def test_specific_volume_is_density_reciprocal(self) -> None:
        props = pb.calculate_simplified_properties(400.0, 1e5)
        assert props.specific_volume == pytest.approx(1.0 / props.density, rel=1e-9)

    @pytest.mark.unit
    def test_specific_heat_ratio_consistent(self) -> None:
        props = pb.calculate_simplified_properties(400.0, 1e5)
        assert props.specific_heat_ratio == pytest.approx(props.cp / props.cv, rel=1e-9)


# ---------------------------------------------------------------------------
# Saturation pressure / temperature (Antoine path; no Cantera)
# ---------------------------------------------------------------------------
class TestSaturationAntoine:
    @pytest.mark.unit
    def test_saturation_pressure_at_100c_near_one_atm(self) -> None:
        # Antoine for water at 373.15 K should land near atmospheric (~101 kPa).
        p = pb.get_saturation_pressure(None, 373.15)
        assert p == pytest.approx(_antoine_psat_pa(373.15), rel=1e-9)
        assert p == pytest.approx(101325.0, rel=0.01)  # within 1% of 1 atm

    @pytest.mark.unit
    def test_saturation_pressure_increases_with_temperature(self) -> None:
        assert pb.get_saturation_pressure(None, 400.0) > pb.get_saturation_pressure(
            None, 350.0
        )

    @pytest.mark.unit
    def test_saturation_temperature_round_trip(self) -> None:
        # T -> P_sat -> T must recover the original temperature.
        t0 = 373.15
        p_sat = pb.get_saturation_pressure(None, t0)
        t_back = pb.get_saturation_temperature(None, p_sat)
        assert t_back == pytest.approx(t0, abs=1e-3)

    @pytest.mark.unit
    def test_saturated_simplified_from_temp_uses_antoine(self) -> None:
        props = pb.calculate_saturated_simplified_from_temp(373.15)
        assert props.pressure == pytest.approx(_antoine_psat_pa(373.15), rel=1e-9)

    @pytest.mark.unit
    def test_saturated_simplified_from_pressure_round_trip(self) -> None:
        # Antoine inverse: pressure -> temperature consistent with forward map.
        p = _antoine_psat_pa(380.0)
        props = pb.calculate_saturated_simplified_from_pressure(p)
        assert props.temperature == pytest.approx(380.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Optional high-accuracy backends: only run when the libraries are present
# ---------------------------------------------------------------------------
class TestCoolPropOptional:
    @pytest.mark.unit
    @pytest.mark.skipif(not pb.COOLPROP_AVAILABLE, reason="CoolProp not installed")
    def test_coolprop_saturation_pressure_steam_table(self) -> None:
        # Steam table: P_sat(100 C) ~ 101.4 kPa (IAPWS).
        props = pb.calculate_saturated_coolprop_from_temp(373.15)
        assert props.pressure == pytest.approx(101.4e3, rel=0.02)

    @pytest.mark.unit
    @pytest.mark.skipif(not pb.COOLPROP_AVAILABLE, reason="CoolProp not installed")
    def test_coolprop_saturation_temperature_steam_table(self) -> None:
        # Steam table: T_sat(1 atm) ~ 373.12 K.
        props = pb.calculate_saturated_coolprop_from_pressure(101325.0)
        assert props.temperature == pytest.approx(373.12, abs=1.0)


class TestCanteraOptional:
    @pytest.mark.unit
    @pytest.mark.skipif(not pb.CANTERA_AVAILABLE, reason="Cantera not installed")
    def test_cantera_properties_runs(self) -> None:
        import cantera as ct  # type: ignore[import-not-found]

        water = ct.Water()
        props = pb.calculate_cantera_properties(water, 400.0, 1e5)
        assert props.density > 0
        assert math.isfinite(props.enthalpy)
