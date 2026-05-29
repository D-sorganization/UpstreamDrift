"""Tests for sidekick.process_calculators.syngas_compression_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.syngas_compression_calculator import (
    CompressionStage,
    SyngasCompressionEngine,
)

# ---------------------------------------------------------------------------
# SyngasCompressionEngine.calculate_water_dropout
# ---------------------------------------------------------------------------


class TestCalculateWaterDropout:
    _ENGINE = SyngasCompressionEngine()

    def test_syngas_compression_returns_dict(self) -> None:
        result = self._ENGINE.calculate_water_dropout(350.0, 5.0, 2.0)
        assert isinstance(result, dict)

    def test_required_keys(self) -> None:
        result = self._ENGINE.calculate_water_dropout(350.0, 5.0, 2.0)
        for key in (
            "water_vapor_pressure",
            "relative_humidity",
            "water_dropout",
            "condensation_rate",
        ):
            assert key in result, f"Missing: {key}"

    def test_no_dropout_below_saturation(self) -> None:
        # Low water content at high T → no dropout
        result = self._ENGINE.calculate_water_dropout(400.0, 1.0, 0.1)
        assert result["water_dropout"] == 0.0

    def test_syngas_compression_zero_pressure_raises(self) -> None:
        with pytest.raises(ValueError):
            self._ENGINE.calculate_water_dropout(350.0, 0.0, 2.0)

    def test_syngas_compression_vapor_pressure_positive(self) -> None:
        result = self._ENGINE.calculate_water_dropout(350.0, 5.0, 2.0)
        assert result["water_vapor_pressure"] > 0.0


# ---------------------------------------------------------------------------
# SyngasCompressionEngine.calculate_compression_work
# ---------------------------------------------------------------------------

# Manually-constructed mixture properties bypassing calculate_mixture_properties
# (that method has a known bug: passes auto_normalize=True to a function
# that doesn't accept it — tracked separately)
_MIX_PROPS = {
    "molecular_weight": 15.0,  # g/mol — H2-rich syngas
    "heat_capacity_ratio": 1.38,
    "critical_temperature": 200.0,
    "critical_pressure": 40.0,
    "mole_fractions": {"H2": 0.5, "CO": 0.3, "CO2": 0.1, "N2": 0.1},
}


class TestCalculateCompressionWork:
    _ENGINE = SyngasCompressionEngine()

    def _make_stage(self) -> CompressionStage:
        return CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=5.0,
            inlet_temperature=300.0,
            efficiency=0.85,
            compression_type="isentropic",
        )

    def test_syngas_compression_returns_dict(self) -> None:
        result = self._ENGINE.calculate_compression_work(
            self._make_stage(), 1000.0, _MIX_PROPS
        )
        assert isinstance(result, dict)

    def test_power_key_present(self) -> None:
        result = self._ENGINE.calculate_compression_work(
            self._make_stage(), 1000.0, _MIX_PROPS
        )
        # Result includes power_hp and heat_rise
        assert (
            "power_hp" in result or "power_kw" in result or "compression_work" in result
        )

    def test_zero_inlet_pressure_raises(self) -> None:
        stage = CompressionStage(
            inlet_pressure=0.0,
            outlet_pressure=5.0,
            inlet_temperature=300.0,
            efficiency=0.85,
            compression_type="isentropic",
        )
        with pytest.raises(ValueError):
            self._ENGINE.calculate_compression_work(stage, 1000.0, _MIX_PROPS)

    def test_zero_outlet_pressure_raises(self) -> None:
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=0.0,
            inlet_temperature=300.0,
            efficiency=0.85,
            compression_type="isentropic",
        )
        with pytest.raises(ValueError):
            self._ENGINE.calculate_compression_work(stage, 1000.0, _MIX_PROPS)
