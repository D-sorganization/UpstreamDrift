"""Tests for sidekick.process_calculators.pressure_drop_calculator.utils.flow_rate_converter
re-export module (Issues #1949, #1744).
"""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator.utils.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    STANDARD_CONDITIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
    acfm_to_scfm,
    convert_flow_rate_to_mass,
    mass_to_mass,
    molar_to_molar,
    scfm_to_acfm,
)


class TestFlowRateReexport:
    """Verify re-exported symbols are available and functional."""

    def test_mass_flow_conversions_dict(self) -> None:
        assert isinstance(MASS_FLOW_CONVERSIONS, dict)
        assert "kg/s" in MASS_FLOW_CONVERSIONS

    def test_molar_flow_conversions_dict(self) -> None:
        assert isinstance(MOLAR_FLOW_CONVERSIONS, dict)
        assert "mol/s" in MOLAR_FLOW_CONVERSIONS

    def test_volumetric_flow_conversions_dict(self) -> None:
        assert isinstance(VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S, dict)

    def test_standard_conditions_dict(self) -> None:
        assert isinstance(STANDARD_CONDITIONS, dict)
        assert "STP" in STANDARD_CONDITIONS

    def test_mass_to_mass_identity(self) -> None:
        result = mass_to_mass(100.0, "kg/h", "kg/h")
        assert result == pytest.approx(100.0)

    def test_molar_to_molar_identity(self) -> None:
        result = molar_to_molar(1.0, "mol/s", "mol/s")
        assert result == pytest.approx(1.0)

    def test_scfm_acfm_roundtrip(self) -> None:
        scfm = 100.0
        temp_k = 300.0
        pressure_pa = 101325.0
        acfm = scfm_to_acfm(scfm, temp_k, pressure_pa, standard="SCFM")
        back = acfm_to_scfm(acfm, temp_k, pressure_pa, standard="SCFM")
        assert back == pytest.approx(scfm, rel=1e-4)

    def test_convert_flow_rate_to_mass_kg_s(self) -> None:
        result = convert_flow_rate_to_mass(1.0, "kg/s", 28.0, 300.0, 101325.0)
        assert result == pytest.approx(1.0, rel=1e-3)
