"""Comprehensive tests for upstream_drift_tools.calculators.mechanical.trc_geometry.

Covers:
- LayerConfig dataclass initialisation and validation
- VesselDimensions data structure
- Pure geometry functions (cylinder, cone, surface area, interior void)
- TRCGeometryEngine.calculate_geometry: full calculations with varying layers
- TRCGeometryEngine.calculate_residence_time: edge cases
- Feature flags (display_lid, display_cylinder, display_cone)
"""

from __future__ import annotations

import math

import pytest
from upstream_drift_tools.calculators.mechanical.trc_geometry import (
    LayerConfig,
    LayerResult,
    TRCGeometryEngine,
    VesselDimensions,
    VesselGeometryResult,
    _calculate_interior_void,
    _calculate_layer_cone_volume,
    _calculate_layer_cylinder_volume,
    _calculate_layer_surface_area,
)

_PI = math.pi
_PI_OVER_3 = math.pi / 3.0
_CUBIC_INCHES_TO_FT3 = 1.0 / 1728.0
_SQUARE_INCHES_TO_FT2 = 1.0 / 144.0


# ── LayerConfig ─────────────────────────────────────────────────────────


class TestLayerConfig:
    """Test LayerConfig dataclass behaviour."""

    def test_default_top_section_name(self) -> None:
        lc = LayerConfig(name="Shell", thickness=0.5, density=490.0, color="grey")
        assert lc.top_section_name == "Shell Top"

    def test_custom_top_section_name(self) -> None:
        lc = LayerConfig(
            name="Shell",
            thickness=0.5,
            density=490.0,
            color="grey",
            top_section_name="Custom",
        )
        assert lc.top_section_name == "Custom"

    def test_numeric_coercion(self) -> None:
        lc = LayerConfig(name="Shell", thickness="2", density="100", color="grey")  # type: ignore[arg-type]
        assert isinstance(lc.thickness, float)
        assert isinstance(lc.density, float)

    def test_default_visible(self) -> None:
        lc = LayerConfig(name="Shell", thickness=1, density=100, color="grey")
        assert lc.visible is True

    def test_default_transparency(self) -> None:
        lc = LayerConfig(name="Shell", thickness=1, density=100, color="grey")
        assert lc.transparency == pytest.approx(0.3)


# ── VesselDimensions ────────────────────────────────────────────────────


class TestVesselDimensions:
    """Test VesselDimensions defaults."""

    def test_default_display_flags(self) -> None:
        vd = VesselDimensions(
            cylinder_height=100.0,
            cylinder_diameter=50.0,
            cone_height=20.0,
            cone_bottom_diameter=10.0,
            cone_interior_hole=5.0,
            top_refractory_thickness=2.0,
        )
        assert vd.display_lid is True
        assert vd.display_cylinder is True
        assert vd.display_cone is True


# ── Pure geometry functions ─────────────────────────────────────────────


class TestCylinderVolume:
    """Test _calculate_layer_cylinder_volume."""

    def test_known_annular_volume(self) -> None:
        # Annular ring: R=10, r=8, h=100  → π(100−64)×100 = π×3600
        vol = _calculate_layer_cylinder_volume(100.0, 64.0, 100.0)
        assert vol == pytest.approx(_PI * 3600.0)

    def test_zero_thickness_gives_zero(self) -> None:
        # Same radii → no volume
        vol = _calculate_layer_cylinder_volume(100.0, 100.0, 100.0)
        assert vol == pytest.approx(0.0)

    def test_zero_height_gives_zero(self) -> None:
        vol = _calculate_layer_cylinder_volume(100.0, 64.0, 0.0)
        assert vol == pytest.approx(0.0)


class TestConeVolume:
    """Test _calculate_layer_cone_volume."""

    def test_positive_volume(self) -> None:
        vol = _calculate_layer_cone_volume(
            current_radius=25.0,
            current_radius_sq=625.0,
            inner_radius=24.5,
            inner_radius_sq=600.25,
            cone_bottom_radius=5.0,
            radius_offset=0.0,
            layer_thickness=0.5,
            interior_hole_radius=2.5,
            cone_height_factor=_PI_OVER_3 * 20.0,
        )
        assert vol > 0

    def test_zero_produces_no_extra_volume(self) -> None:
        # Same inner/outer → zero volume
        vol = _calculate_layer_cone_volume(
            current_radius=25.0,
            current_radius_sq=625.0,
            inner_radius=25.0,
            inner_radius_sq=625.0,
            cone_bottom_radius=5.0,
            radius_offset=0.0,
            layer_thickness=0.0,
            interior_hole_radius=2.5,
            cone_height_factor=_PI_OVER_3 * 20.0,
        )
        assert vol == pytest.approx(0.0)


class TestSurfaceArea:
    """Test _calculate_layer_surface_area."""

    def test_cylinder_only(self) -> None:
        area = _calculate_layer_surface_area(
            current_radius=25.0,
            cylinder_height=100.0,
            cone_bottom_radius=5.0,
            radius_offset=0.0,
            interior_hole_radius=2.5,
            cone_height=20.0,
            display_cylinder=True,
            display_cone=False,
        )
        expected = 2.0 * _PI * 25.0 * 100.0 * _SQUARE_INCHES_TO_FT2
        assert area == pytest.approx(expected)

    def test_cone_only(self) -> None:
        area = _calculate_layer_surface_area(
            current_radius=25.0,
            cylinder_height=100.0,
            cone_bottom_radius=5.0,
            radius_offset=0.0,
            interior_hole_radius=2.5,
            cone_height=20.0,
            display_cylinder=False,
            display_cone=True,
        )
        assert area > 0

    def test_neither_gives_zero(self) -> None:
        area = _calculate_layer_surface_area(
            current_radius=25.0,
            cylinder_height=100.0,
            cone_bottom_radius=5.0,
            radius_offset=0.0,
            interior_hole_radius=2.5,
            cone_height=20.0,
            display_cylinder=False,
            display_cone=False,
        )
        assert area == 0.0


class TestInteriorVoid:
    """Test _calculate_interior_void."""

    def test_positive_void(self) -> None:
        vol = _calculate_interior_void(
            last_inner_radius=20.0,
            half_cylinder_diameter=25.0,
            cone_bottom_radius=5.0,
            interior_hole_radius=2.5,
            interior_height=100.0,
            cone_height_factor=_PI_OVER_3 * 20.0,
            display_cylinder=True,
            display_cone=True,
        )
        assert vol > 0

    def test_no_cylinder_no_cone(self) -> None:
        vol = _calculate_interior_void(
            last_inner_radius=20.0,
            half_cylinder_diameter=25.0,
            cone_bottom_radius=5.0,
            interior_hole_radius=2.5,
            interior_height=100.0,
            cone_height_factor=_PI_OVER_3 * 20.0,
            display_cylinder=False,
            display_cone=False,
        )
        assert vol == 0.0

    def test_cylinder_only_void(self) -> None:
        vol = _calculate_interior_void(
            last_inner_radius=20.0,
            half_cylinder_diameter=25.0,
            cone_bottom_radius=5.0,
            interior_hole_radius=2.5,
            interior_height=100.0,
            cone_height_factor=_PI_OVER_3 * 20.0,
            display_cylinder=True,
            display_cone=False,
        )
        expected = _PI * 400.0 * 100.0
        assert vol == pytest.approx(expected)


# ── TRCGeometryEngine ──────────────────────────────────────────────────


class TestTRCGeometryEngine:
    """Test the main engine calculations."""

    @pytest.fixture()
    def engine(self) -> TRCGeometryEngine:
        return TRCGeometryEngine()

    @pytest.fixture()
    def dimensions(self) -> VesselDimensions:
        return VesselDimensions(
            cylinder_height=100.0,
            cylinder_diameter=50.0,
            cone_height=20.0,
            cone_bottom_diameter=10.0,
            cone_interior_hole=5.0,
            top_refractory_thickness=2.0,
        )

    @pytest.fixture()
    def single_layer(self) -> list[LayerConfig]:
        return [
            LayerConfig(name="Shell", thickness=0.5, density=490.0, color="grey"),
        ]

    @pytest.fixture()
    def two_layers(self) -> list[LayerConfig]:
        return [
            LayerConfig(name="Shell", thickness=0.5, density=490.0, color="grey"),
            LayerConfig(
                name="Refractory",
                thickness=4.0,
                density=150.0,
                color="orange",
            ),
        ]

    # ── calculate_geometry ──────────────────────────────────────────────

    def test_empty_layers_returns_zero_result(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
    ) -> None:
        result = engine.calculate_geometry(dimensions, [])
        assert result.total_volume_ft3 == 0.0
        assert result.total_mass_lb == 0.0
        assert len(result.layers) == 0

    def test_single_layer_positive_results(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
        single_layer: list[LayerConfig],
    ) -> None:
        result = engine.calculate_geometry(dimensions, single_layer)
        assert result.total_volume_ft3 > 0
        assert result.total_mass_lb > 0
        assert len(result.layers) == 1
        assert result.layers[0].name == "Shell"

    def test_two_layers_both_reported(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
        two_layers: list[LayerConfig],
    ) -> None:
        result = engine.calculate_geometry(dimensions, two_layers)
        assert len(result.layers) == 2
        names = [lr.name for lr in result.layers]
        assert "Shell" in names
        assert "Refractory" in names

    def test_interior_volume_positive(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
        two_layers: list[LayerConfig],
    ) -> None:
        result = engine.calculate_geometry(dimensions, two_layers)
        assert result.interior_volume_ft3 > 0

    def test_void_dimensions_set(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
        two_layers: list[LayerConfig],
    ) -> None:
        result = engine.calculate_geometry(dimensions, two_layers)
        assert result.void_radius_inches > 0
        assert result.void_diameter_inches > 0
        assert result.void_diameter_inches == pytest.approx(
            result.void_radius_inches * 2.0,
        )

    def test_interior_height_set(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
        two_layers: list[LayerConfig],
    ) -> None:
        result = engine.calculate_geometry(dimensions, two_layers)
        # Interior height = cylinder height - top refractory thickness (with lid)
        expected = dimensions.cylinder_height - dimensions.top_refractory_thickness
        assert result.interior_height_inches == pytest.approx(expected)

    def test_hidden_layer_excluded(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
    ) -> None:
        layers = [
            LayerConfig(
                name="Hidden",
                thickness=1.0,
                density=100.0,
                color="grey",
                visible=False,
            ),
        ]
        result = engine.calculate_geometry(dimensions, layers)
        assert len(result.layers) == 0
        assert result.total_volume_ft3 == 0.0

    def test_zero_thickness_layer_excluded(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
    ) -> None:
        layers = [
            LayerConfig(name="Zero", thickness=0.0, density=100.0, color="grey"),
        ]
        result = engine.calculate_geometry(dimensions, layers)
        assert len(result.layers) == 0

    def test_no_lid_increases_interior_height(
        self,
        engine: TRCGeometryEngine,
        single_layer: list[LayerConfig],
    ) -> None:
        dims_with_lid = VesselDimensions(
            cylinder_height=100.0,
            cylinder_diameter=50.0,
            cone_height=20.0,
            cone_bottom_diameter=10.0,
            cone_interior_hole=5.0,
            top_refractory_thickness=2.0,
            display_lid=True,
        )
        dims_no_lid = VesselDimensions(
            cylinder_height=100.0,
            cylinder_diameter=50.0,
            cone_height=20.0,
            cone_bottom_diameter=10.0,
            cone_interior_hole=5.0,
            top_refractory_thickness=2.0,
            display_lid=False,
        )
        r1 = engine.calculate_geometry(dims_with_lid, single_layer)
        r2 = engine.calculate_geometry(dims_no_lid, single_layer)
        assert r2.interior_height_inches > r1.interior_height_inches

    def test_more_layers_less_interior_volume(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
        single_layer: list[LayerConfig],
        two_layers: list[LayerConfig],
    ) -> None:
        r1 = engine.calculate_geometry(dimensions, single_layer)
        r2 = engine.calculate_geometry(dimensions, two_layers)
        # Adding refractory reduces interior void
        assert r2.interior_volume_ft3 < r1.interior_volume_ft3

    def test_mass_proportional_to_density(
        self,
        engine: TRCGeometryEngine,
        dimensions: VesselDimensions,
    ) -> None:
        light = [LayerConfig(name="A", thickness=1.0, density=50.0, color="r")]
        heavy = [LayerConfig(name="B", thickness=1.0, density=200.0, color="b")]
        r_l = engine.calculate_geometry(dimensions, light)
        r_h = engine.calculate_geometry(dimensions, heavy)
        # Same geometry, 4x density → 4x mass
        assert r_h.total_mass_lb == pytest.approx(r_l.total_mass_lb * 4.0, rel=1e-6)

    # ── calculate_residence_time ────────────────────────────────────────

    def test_residence_time_basic(self, engine: TRCGeometryEngine) -> None:
        assert engine.calculate_residence_time(100.0, 100.0) == pytest.approx(60.0)

    def test_residence_time_zero_flow(self, engine: TRCGeometryEngine) -> None:
        assert engine.calculate_residence_time(100.0, 0.0) == 0.0

    def test_residence_time_negative_flow(self, engine: TRCGeometryEngine) -> None:
        assert engine.calculate_residence_time(100.0, -10.0) == 0.0

    def test_residence_time_zero_volume(self, engine: TRCGeometryEngine) -> None:
        assert engine.calculate_residence_time(0.0, 100.0) == 0.0

    def test_residence_time_large_volume(self, engine: TRCGeometryEngine) -> None:
        # 1000 ft3 / 50 acfm = 20 min = 1200 s
        assert engine.calculate_residence_time(1000.0, 50.0) == pytest.approx(1200.0)


# ── VesselGeometryResult defaults ───────────────────────────────────────


class TestResultDataclasses:
    """Test result dataclass defaults."""

    def test_default_result(self) -> None:
        r = VesselGeometryResult()
        assert r.layers == []
        assert r.total_volume_ft3 == 0.0
        assert r.total_mass_lb == 0.0

    def test_layer_result_defaults(self) -> None:
        lr = LayerResult(name="test", volume_ft3=1.0, mass_lb=2.0, density=100.0)
        assert lr.outer_surface_area_ft2 == 0.0
