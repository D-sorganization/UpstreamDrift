"""Tests for sidekick.calculators.mechanical.trc_geometry (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.calculators.mechanical.trc_geometry import (
    LayerConfig,
    TRCGeometryEngine,
    VesselDimensions,
    VesselGeometryResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_dims(**overrides: float | bool) -> VesselDimensions:
    d = VesselDimensions(
        cylinder_height=120.0,
        cylinder_diameter=60.0,
        cone_height=36.0,
        cone_bottom_diameter=12.0,
        cone_interior_hole=6.0,
        top_refractory_thickness=4.0,
    )
    for k, v in overrides.items():
        setattr(d, k, v)
    return d


def _layer(
    name: str = "Steel", thickness: float = 2.0, density: float = 490.0
) -> LayerConfig:
    return LayerConfig(name=name, thickness=thickness, density=density, color="gray")


# ---------------------------------------------------------------------------
# LayerConfig
# ---------------------------------------------------------------------------


class TestLayerConfig:
    def test_trc_geometry_name_stored(self) -> None:
        lc = _layer("Refractory")
        assert lc.name == "Refractory"

    def test_default_top_section_name(self) -> None:
        lc = _layer("Steel")
        assert lc.top_section_name == "Steel Top"

    def test_thickness_is_float(self) -> None:
        lc = LayerConfig(name="A", thickness="3", density="490", color="red")  # type: ignore[arg-type]
        assert isinstance(lc.thickness, float)


# ---------------------------------------------------------------------------
# TRCGeometryEngine.calculate_geometry
# ---------------------------------------------------------------------------


class TestCalculateGeometry:
    _ENGINE = TRCGeometryEngine()

    def test_returns_vessel_geometry_result(self) -> None:
        result = self._ENGINE.calculate_geometry(_simple_dims(), [_layer()])
        assert isinstance(result, VesselGeometryResult)

    def test_empty_layers_returns_zero_mass(self) -> None:
        result = self._ENGINE.calculate_geometry(_simple_dims(), [])
        assert result.total_mass_lb == 0.0

    def test_positive_mass_with_layer(self) -> None:
        result = self._ENGINE.calculate_geometry(_simple_dims(), [_layer()])
        assert result.total_mass_lb > 0.0

    def test_positive_volume_with_layer(self) -> None:
        result = self._ENGINE.calculate_geometry(_simple_dims(), [_layer()])
        assert result.total_volume_ft3 > 0.0

    def test_layers_populated(self) -> None:
        result = self._ENGINE.calculate_geometry(
            _simple_dims(), [_layer("A"), _layer("B")]
        )
        assert len(result.layers) == 2

    def test_invisible_layer_excluded(self) -> None:
        visible = _layer("outer")
        invisible = LayerConfig(
            name="hidden", thickness=2.0, density=490.0, color="x", visible=False
        )
        result = self._ENGINE.calculate_geometry(_simple_dims(), [visible, invisible])
        assert len(result.layers) == 1

    def test_zero_diameter_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._ENGINE.calculate_geometry(
                _simple_dims(cylinder_diameter=0.0), [_layer()]
            )

    def test_zero_height_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._ENGINE.calculate_geometry(
                _simple_dims(cylinder_height=0.0), [_layer()]
            )


# ---------------------------------------------------------------------------
# TRCGeometryEngine.calculate_residence_time
# ---------------------------------------------------------------------------


class TestCalculateResidenceTime:
    _ENGINE = TRCGeometryEngine()

    def test_positive_flow_gives_positive_time(self) -> None:
        t = self._ENGINE.calculate_residence_time(100.0, 10.0)
        assert t > 0.0

    def test_formula(self) -> None:
        # 120 ft3 / 10 acfm = 12 min = 720 s
        t = self._ENGINE.calculate_residence_time(120.0, 10.0)
        assert abs(t - 720.0) < 0.01

    def test_zero_flow_returns_zero(self) -> None:
        t = self._ENGINE.calculate_residence_time(100.0, 0.0)
        assert t == 0.0

    def test_larger_volume_longer_time(self) -> None:
        t1 = self._ENGINE.calculate_residence_time(50.0, 5.0)
        t2 = self._ENGINE.calculate_residence_time(200.0, 5.0)
        assert t2 > t1
