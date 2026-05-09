"""Tests for pressure_drop_calculator.pressure_drop_interface (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
    calculate_pressure_drop,
    get_roughness,
    list_gas_components,
    list_materials,
)


class TestListFunctions:
    def test_list_materials_nonempty(self) -> None:
        mats = list(list_materials())
        assert len(mats) > 0

    def test_list_gas_components_is_dict(self) -> None:
        comps = list_gas_components()
        assert isinstance(comps, dict)

    def test_list_gas_components_nonempty(self) -> None:
        comps = list_gas_components()
        assert len(comps) > 0


class TestGetRoughness:
    def test_known_material(self) -> None:
        mats = list(list_materials())
        r = get_roughness(mats[0])
        assert r > 0.0

    def test_unknown_material_raises(self) -> None:
        with pytest.raises(ValueError):
            get_roughness("UnobtainiumXYZ")


class TestCalculatePressureDrop:
    def test_pressure_drop_interface_returns_dict(self) -> None:
        result = calculate_pressure_drop(
            pipe_diameter=0.05,
            pipe_length=10.0,
            flow_rate=100.0,
            flow_unit="kg/h",
        )
        assert isinstance(result, dict)

    def test_has_pressure_drop_key(self) -> None:
        result = calculate_pressure_drop(
            pipe_diameter=0.05,
            pipe_length=10.0,
            flow_rate=100.0,
            flow_unit="kg/h",
        )
        assert "pressure_drop_pa" in result

    def test_has_friction_factor(self) -> None:
        result = calculate_pressure_drop(
            pipe_diameter=0.1,
            pipe_length=50.0,
            flow_rate=500.0,
            flow_unit="kg/h",
        )
        assert "friction_factor" in result

    def test_pressure_drop_positive(self) -> None:
        result = calculate_pressure_drop(
            pipe_diameter=0.05,
            pipe_length=10.0,
            flow_rate=100.0,
            flow_unit="kg/h",
        )
        assert result["pressure_drop_pa"] >= 0.0
