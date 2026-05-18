"""Tests for sidekick.calculators.conversion.flow_rate_converter (Issues #1949, #1744)."""

from __future__ import annotations

import math

import pytest
from sidekick.calculators.conversion.flow_rate_converter import (
    acfm_to_scfm,
    mass_to_mass,
    molar_to_molar,
    scfm_to_acfm,
)

# ---------------------------------------------------------------------------
# mass_to_mass
# ---------------------------------------------------------------------------


class TestMassToMass:
    def test_flow_rate_converter_identity(self) -> None:
        result = mass_to_mass(100.0, "kg/h", "kg/h")
        assert abs(result - 100.0) < 1e-10

    def test_kg_h_to_lb_hr_positive(self) -> None:
        result = mass_to_mass(1.0, "kg/h", "lb/hr")
        assert result > 0.0

    def test_kg_h_to_lb_hr_approx(self) -> None:
        # 1 kg/h ≈ 2.2046 lb/hr
        result = mass_to_mass(1.0, "kg/h", "lb/hr")
        assert abs(result - 2.2046) < 0.01

    def test_flow_rate_converter_roundtrip(self) -> None:
        original = 500.0
        lb_hr = mass_to_mass(original, "kg/h", "lb/hr")
        back = mass_to_mass(lb_hr, "lb/hr", "kg/h")
        assert abs(back - original) < 1e-6

    def test_flow_rate_converter_unknown_from_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            mass_to_mass(1.0, "INVALID_UNIT", "kg/h")

    def test_flow_rate_converter_unknown_to_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            mass_to_mass(1.0, "kg/h", "INVALID_UNIT")

    def test_infinite_value_raises(self) -> None:
        with pytest.raises(ValueError):
            mass_to_mass(math.inf, "kg/h", "lb/hr")


# ---------------------------------------------------------------------------
# molar_to_molar
# ---------------------------------------------------------------------------


class TestMolarToMolar:
    def test_flow_rate_converter_identity(self) -> None:
        result = molar_to_molar(50.0, "kmol/h", "kmol/h")
        assert abs(result - 50.0) < 1e-10

    def test_kmol_h_to_mol_s(self) -> None:
        # 1 kmol/h = 1000 mol / 3600 s ≈ 0.2778 mol/s
        result = molar_to_molar(1.0, "kmol/h", "mol/s")
        assert abs(result - 0.2778) < 0.001

    def test_flow_rate_converter_roundtrip(self) -> None:
        original = 10.0
        mol_s = molar_to_molar(original, "kmol/h", "mol/s")
        back = molar_to_molar(mol_s, "mol/s", "kmol/h")
        assert abs(back - original) < 1e-6

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            molar_to_molar(1.0, "BADUNIT", "kmol/h")


# ---------------------------------------------------------------------------
# scfm_to_acfm / acfm_to_scfm
# ---------------------------------------------------------------------------


class TestScfmAcfmConversions:
    _STD_T_K = 294.26  # SCFM standard: 70°F ≈ 294.26 K
    _ATM_PA = 101325.0

    def test_scfm_to_acfm_at_standard_conditions_approx_equal(self) -> None:
        # At T≈T_std and P≈P_std, ACFM ≈ SCFM
        result = scfm_to_acfm(1000.0, self._STD_T_K, self._ATM_PA, "SCFM")
        assert abs(result - 1000.0) < 50.0  # within 5%

    def test_scfm_to_acfm_high_temp_larger(self) -> None:
        # Higher actual T → ACFM > SCFM
        result = scfm_to_acfm(1000.0, 600.0, self._ATM_PA, "SCFM")
        assert result > 1000.0

    def test_scfm_to_acfm_positive(self) -> None:
        result = scfm_to_acfm(500.0, 400.0, self._ATM_PA, "SCFM")
        assert result > 0.0

    def test_roundtrip_scfm_acfm(self) -> None:
        original_scfm = 1000.0
        temp, pressure = 500.0, 2e5
        acfm = scfm_to_acfm(original_scfm, temp, pressure, "SCFM")
        back_scfm = acfm_to_scfm(acfm, temp, pressure, "SCFM")
        assert abs(back_scfm - original_scfm) < 0.01

    def test_flow_rate_converter_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            scfm_to_acfm(1000.0, -10.0, self._ATM_PA, "SCFM")

    def test_flow_rate_converter_zero_pressure_raises(self) -> None:
        with pytest.raises(ValueError):
            scfm_to_acfm(1000.0, 300.0, 0.0, "SCFM")
