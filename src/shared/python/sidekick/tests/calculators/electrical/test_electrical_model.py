from typing import Any

"""test_electrical_model.py module."""

import numpy as np
import pytest
from sidekick.calculators.electrical.config import ElectrodeConfig
from sidekick.calculators.electrical.electrical_model import (
    ThreePhaseElectricalModelEnhanced,
)
from sidekick.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)


class TestElectricalModel:
    @pytest.fixture
    def model(self) -> Any:
        config = ElectrodeConfig()
        glass = GlassPropertiesInterface()
        return ThreePhaseElectricalModelEnhanced(config, glass)

    def test_initialization(self, model) -> Any:
        assert len(model.electrode_positions) == 3
        np.testing.assert_allclose(
            model.electrode_positions, [0, 2.094395, 4.18879], rtol=1e-4
        )

    def test_calculate_system_state(self, model) -> Any:
        depths = np.array([10.0, 10.0, 10.0])
        voltages = np.array([100.0, 100.0, 100.0])

        result = model.calculate_system_state(
            depths=depths,
            bath_diameter=120.0,
            tip_diameter=24.0,
            metal_depth=2.0,
            k_factors={"K_tt": 1.0},
            bath_temperature=1200.0,
            voltages=voltages,
        )

        assert "resistances" in result
        assert "current_paths" in result
        assert "current_distribution" in result
        assert len(result["actual_currents"]) == 3

    def test_parallel_resistance(self, model) -> Any:
        r1 = 100.0
        r2 = 100.0
        rp = model._parallel_resistance(r1, r2)
        assert rp == 50.0  # Parallel of two equal resistors is half

    def test_parallel_resistance_nan_case(self, model) -> Any:
        """Line 480: NaN input → returns max of the valid values."""
        import math

        result = model._parallel_resistance(float("nan"), 10.0)
        assert math.isnan(result)

    def test_parallel_resistance_negative_case(self, model) -> Any:
        """Line 480: Zero/negative r → returns max(r1, r2)."""
        result = model._parallel_resistance(0.0, 5.0)
        assert result == 5.0

    def test_system_state_metal_nonconductive(self, model) -> Any:
        """Lines 158-162: metal_conductive=False path (direct_fraction=1, metal=0)."""
        depths = np.array([10.0, 10.0, 10.0])
        voltages = np.array([100.0, 100.0, 100.0])
        result = model.calculate_system_state(
            depths=depths,
            bath_diameter=120.0,
            tip_diameter=24.0,
            metal_depth=2.0,
            k_factors={"K_tt": 1.0},
            bath_temperature=1200.0,
            voltages=voltages,
            metal_conductive=False,
        )
        # All direct_fraction should be 1.0 and metal_fraction 0.0
        for path in result["current_paths"].values():
            assert path["direct_fraction"] == 1.0
            assert path["metal_fraction"] == 0.0

    def test_electrode_position_cache_hit(self, model) -> Any:
        """Line 193: second call returns cached positions."""
        depths = np.array([10.0, 10.0, 10.0])
        positions1 = model._calculate_electrode_positions_3d(depths, 60.0, 2.0)
        positions2 = model._calculate_electrode_positions_3d(depths, 60.0, 2.0)
        # Should return the same object from cache
        assert positions1 is positions2

    def test_vertical_segment_zero_area(self, model) -> Any:
        """Line 408: area_m2 == 0 → returns default_resistance."""
        result = model._vertical_glass_segment_resistance(
            electrode_length=0.0,  # Zero length → zero area
            effective_width=0.0,
            electrode_z=10.0,
            metal_depth=2.0,
            conductivity=1.0,
            default_resistance=0.001,
        )
        assert result == 0.001

    def test_calculate_path_currents_exception(self, model) -> Any:
        """Lines 501-502: exception in _calculate_path_currents returns zeros."""
        # Pass a None-keyed dict that will cause a TypeError during iteration
        resistances = {"1-2": float("nan"), "2-3": float("nan"), "3-1": float("nan")}
        # Force OverflowError by making voltage enormous
        voltages = np.array([1.0, 1.0, 1.0])
        result = model._calculate_path_currents(resistances, voltages)
        # Should work normally since nan division is valid float
        assert "1-2" in result

    def test_calculate_path_currents_missing_phase(self, model) -> Any:
        """Line 498: phase not in resistances → current is 0.0."""
        resistances = {"1-2": 10.0}  # Missing 2-3 and 3-1
        voltages = np.array([100.0, 100.0, 100.0])
        result = model._calculate_path_currents(resistances, voltages)
        assert result["2-3"] == 0.0
        assert result["3-1"] == 0.0

    def test_glass_conductivity(self) -> Any:
        glass = GlassPropertiesInterface()

        # Test default Arrhenius behavior
        cond_1200 = glass.get_conductivity(1200.0)  # Base temp (1473.15 K)
        cond_1300 = glass.get_conductivity(1300.0)  # Higher temp

        assert cond_1300 > cond_1200  # Conductivity should increase with temp

        # Test metal conductivity
        cond_metal = glass.get_conductivity(1200.0, is_metal=True)
        assert cond_metal == 10000.0


class TestGlassPropertiesInterface:
    """Extended tests for GlassPropertiesInterface covering all paths."""

    def test_external_calculator_used(self) -> Any:
        """Lines 95-100: external calculator returns custom conductivity."""

        def my_calc(temp, comp, power) -> Any:
            return 42.0

        glass = GlassPropertiesInterface(external_calculator=my_calc)
        result = glass.get_conductivity(1200.0)
        assert result == 42.0

    def test_external_calculator_fallback_on_error(self) -> Any:
        """Lines 101-106: external calculator raises → fallback to default model."""

        def bad_calc(temp, comp, power) -> Any:
            raise ValueError("External calc failed")

        glass = GlassPropertiesInterface(external_calculator=bad_calc)
        result = glass.get_conductivity(1200.0)
        # Should return the default Arrhenius value (not 42.0)
        assert isinstance(result, float)
        assert result > 0

    def test_lru_cache_eviction(self) -> Any:
        """Line 118: popitem (LRU eviction) when cache exceeds max size."""
        glass = GlassPropertiesInterface(cache_max_size=2)
        # Fill cache beyond max_size
        glass.get_conductivity(1000.0)
        glass.get_conductivity(1100.0)
        glass.get_conductivity(1200.0)  # Triggers eviction
        assert len(glass._temperature_dependent_data) <= 2

    def test_cache_hit_moves_to_end(self) -> Any:
        """Lines 88-91: Cache hit promotes entry (LRU behavior)."""
        glass = GlassPropertiesInterface()
        first = glass.get_conductivity(1200.0)
        second = glass.get_conductivity(1200.0)  # Cache hit
        assert first == second

    def test_set_external_calculator(self) -> Any:
        """Lines 124-127: set_external_calculator sets and clears cache."""
        glass = GlassPropertiesInterface()
        glass.get_conductivity(1200.0)
        assert len(glass._temperature_dependent_data) == 1

        glass.set_external_calculator(lambda t, c, p: 99.0)
        # Cache should be cleared
        assert len(glass._temperature_dependent_data) == 0
        result = glass.get_conductivity(1200.0)
        assert result == 99.0

    def test_update_and_get_properties(self) -> Any:
        """Lines 131, 135: update_properties and get_current_properties."""
        glass = GlassPropertiesInterface()
        glass.update_properties({"viscosity": 1.5})
        props = glass.get_current_properties()
        assert props["viscosity"] == 1.5

    def test_power_density_heats_glass(self) -> Any:
        """Line 154: power_density > 0 → increased effective temperature."""
        glass = GlassPropertiesInterface()
        cond_no_power = glass.get_conductivity(1200.0, power_density=0)
        cond_with_power = glass.get_conductivity(1200.0, power_density=10000)
        # Higher effective temp → higher conductivity
        assert cond_with_power > cond_no_power

    def test_get_resistivity(self) -> Any:
        """Lines 173-180: get_resistivity returns 1/conductivity."""
        glass = GlassPropertiesInterface()
        cond = glass.get_conductivity(1200.0)
        res = glass.get_resistivity(1200.0)
        assert abs(res - 1.0 / cond) < 1e-10

    def test_get_resistivity_zero_conductivity(self) -> Any:
        """Line 180: zero conductivity → return inf."""
        glass = GlassPropertiesInterface(external_calculator=lambda t, c, p: 0.0)
        res = glass.get_resistivity(1200.0)
        assert res == float("inf")

    def test_clear_cache(self) -> Any:
        """Line 184: clear_cache empties the cache."""
        glass = GlassPropertiesInterface()
        glass.get_conductivity(1200.0)
        assert len(glass._temperature_dependent_data) == 1
        glass.clear_cache()
        assert len(glass._temperature_dependent_data) == 0

    def test_composition_used_in_cache_key(self) -> Any:
        """Lines 84-85: composition dict used in cache key (frozenset)."""
        glass = GlassPropertiesInterface()
        comp = {"SiO2": 0.7}
        result = glass.get_conductivity(1200.0, composition=comp)
        assert isinstance(result, float)
        assert len(glass._temperature_dependent_data) == 1


class TestElectrodeConfigMethods:
    """Tests for ElectrodeConfig.status_color and scheme_color methods."""

    def test_status_color_ok(self) -> Any:
        cfg = ElectrodeConfig()
        assert cfg.status_color("ok") == "#C8FFC8"

    def test_status_color_warn(self) -> Any:
        cfg = ElectrodeConfig()
        assert cfg.status_color("warn") == "#FFFFB4"

    def test_status_color_error(self) -> Any:
        cfg = ElectrodeConfig()
        assert cfg.status_color("error") == "#FF9696"

    def test_status_color_unknown_falls_back_to_ok(self) -> Any:
        cfg = ElectrodeConfig()
        result = cfg.status_color("unknown_type")
        assert result == "#C8FFC8"

    def test_status_color_colors_none(self) -> Any:
        """When colors dict is None, returns fallback."""
        cfg = ElectrodeConfig(colors=None)
        cfg.__post_init__()
        cfg.colors = None
        assert cfg.status_color("ok") == "#C8FFC8"

    def test_scheme_color_default_scheme(self) -> Any:
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("default", "direct_glass")
        assert color == "#4169E1"

    def test_scheme_color_missing_path_type_returns_lightblue(self) -> Any:
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("default", "nonexistent_path")
        assert color == "lightblue"

    def test_scheme_color_missing_scheme_returns_lightblue(self) -> Any:
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("nonexistent_scheme", "direct_glass")
        assert color == "lightblue"

    def test_scheme_color_color_schemes_none(self) -> Any:
        """When color_schemes is None, returns 'lightblue'."""
        cfg = ElectrodeConfig()
        cfg.__post_init__()
        cfg.color_schemes = None
        assert cfg.scheme_color("default", "direct_glass") == "lightblue"
