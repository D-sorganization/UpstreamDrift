"""Tests for sidekick.calculators.mechanical.trc_geometry (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.calculators.mechanical.trc_geometry import (
    LayerConfig,
    TRCGeometryEngine,
    VesselDimensions,
    VesselGeometryResult,
)


def _make_dims() -> VesselDimensions:
    return VesselDimensions(
        cylinder_height=100.0,
        cylinder_diameter=60.0,
        cone_height=40.0,
        cone_bottom_diameter=30.0,
        cone_interior_hole=10.0,
        top_refractory_thickness=5.0,
    )


def _make_steel_layer() -> LayerConfig:
    return LayerConfig(name="Steel", thickness=0.25, density=490.0, color="#808080")


class TestLayerConfigPostInit:
    def test_top_section_name_default(self) -> None:
        layer = LayerConfig(name="Outer", thickness=1.0, density=200.0, color="#FFF")
        assert layer.top_section_name == "Outer Top"

    def test_custom_top_section_name_preserved(self) -> None:
        layer = LayerConfig(
            name="Outer",
            thickness=1.0,
            density=200.0,
            color="#FFF",
            top_section_name="CustomTop",
        )
        assert layer.top_section_name == "CustomTop"

    def test_thickness_is_float(self) -> None:
        layer = LayerConfig(name="A", thickness=2, density=100, color="#000")
        assert isinstance(layer.thickness, float)


class TestTRCGeometryEngineCalculateGeometry:
    def test_empty_layers_returns_result(self) -> None:
        engine = TRCGeometryEngine()
        result = engine.calculate_geometry(_make_dims(), [])
        assert isinstance(result, VesselGeometryResult)

    def test_empty_layers_zero_volume(self) -> None:
        engine = TRCGeometryEngine()
        result = engine.calculate_geometry(_make_dims(), [])
        assert result.total_volume_ft3 == 0.0

    def test_single_layer_positive_volume(self) -> None:
        engine = TRCGeometryEngine()
        result = engine.calculate_geometry(_make_dims(), [_make_steel_layer()])
        assert result.total_volume_ft3 > 0.0

    def test_single_layer_positive_mass(self) -> None:
        engine = TRCGeometryEngine()
        result = engine.calculate_geometry(_make_dims(), [_make_steel_layer()])
        assert result.total_mass_lb > 0.0

    def test_invisible_layer_not_counted(self) -> None:
        engine = TRCGeometryEngine()
        visible_layer = LayerConfig(
            name="Steel", thickness=0.25, density=490.0, color="#808080", visible=True
        )
        invisible_layer = LayerConfig(
            name="Lead", thickness=1.0, density=710.0, color="#444", visible=False
        )
        result_one = engine.calculate_geometry(_make_dims(), [visible_layer])
        result_two = engine.calculate_geometry(
            _make_dims(), [visible_layer, invisible_layer]
        )
        # Adding invisible layer should not change the result
        assert abs(result_one.total_mass_lb - result_two.total_mass_lb) < 1e-6

    def test_more_layers_more_mass(self) -> None:
        engine = TRCGeometryEngine()
        one_layer = engine.calculate_geometry(_make_dims(), [_make_steel_layer()])
        two_layers = engine.calculate_geometry(
            _make_dims(),
            [
                _make_steel_layer(),
                LayerConfig("Refractory", 2.0, 120.0, "#FF8800"),
            ],
        )
        assert two_layers.total_mass_lb > one_layer.total_mass_lb

    def test_surface_area_non_negative(self) -> None:
        engine = TRCGeometryEngine()
        result = engine.calculate_geometry(_make_dims(), [_make_steel_layer()])
        assert result.outside_surface_area_ft2 >= 0.0

    def test_invalid_diameter_raises(self) -> None:
        engine = TRCGeometryEngine()
        bad_dims = VesselDimensions(
            cylinder_height=100.0,
            cylinder_diameter=0.0,  # invalid
            cone_height=40.0,
            cone_bottom_diameter=30.0,
            cone_interior_hole=10.0,
            top_refractory_thickness=5.0,
        )
        with pytest.raises(AssertionError):
            engine.calculate_geometry(bad_dims, [_make_steel_layer()])
