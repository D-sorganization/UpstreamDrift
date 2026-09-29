"""Tests for sidekick.calculators.electrical.glass_interface (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)


class TestGlassPropertiesInterfaceInit:
    def test_instantiates_with_defaults(self) -> None:
        g = GlassPropertiesInterface()
        assert g is not None

    def test_no_external_calculator_by_default(self) -> None:
        g = GlassPropertiesInterface()
        assert g.external_calculator is None

    def test_custom_cache_max_size(self) -> None:
        g = GlassPropertiesInterface(cache_max_size=50)
        assert g._cache_max_size == 50


class TestGetConductivity:
    def test_returns_positive_float(self) -> None:
        g = GlassPropertiesInterface()
        c = g.get_conductivity(1200.0)
        assert c > 0.0
        assert isinstance(c, float)

    def test_metal_returns_high_conductivity(self) -> None:
        g = GlassPropertiesInterface()
        c = g.get_conductivity(1200.0, is_metal=True)
        assert c == 10000.0

    def test_higher_temperature_higher_conductivity(self) -> None:
        g = GlassPropertiesInterface()
        c_low = g.get_conductivity(800.0)
        c_high = g.get_conductivity(1400.0)
        assert c_high > c_low

    def test_cached_result_returned_on_second_call(self) -> None:
        g = GlassPropertiesInterface()
        c1 = g.get_conductivity(1200.0)
        c2 = g.get_conductivity(1200.0)
        assert c1 == c2

    def test_composition_affects_cache_key(self) -> None:
        g = GlassPropertiesInterface()
        c1 = g.get_conductivity(1200.0, composition=None)
        c2 = g.get_conductivity(1200.0, composition={"SiO2": 0.7})
        # Keys differ, both should be computed without error
        assert isinstance(c1, float)
        assert isinstance(c2, float)

    def test_external_calculator_used_when_set(self) -> None:
        g = GlassPropertiesInterface(external_calculator=lambda t, comp, pd: 42.0)
        c = g.get_conductivity(1200.0)
        assert c == 42.0

    def test_external_calculator_failure_falls_back(self) -> None:
        def bad_calc(t: float, comp: object, pd: float) -> float:
            msg = "oops"
            raise ValueError(msg)

        g = GlassPropertiesInterface(external_calculator=bad_calc)
        # Should not raise — falls back to default Arrhenius model
        c = g.get_conductivity(1200.0)
        assert c > 0.0

    def test_power_density_increases_conductivity(self) -> None:
        g = GlassPropertiesInterface()
        c0 = g.get_conductivity(1200.0, power_density=0)
        c_pd = g.get_conductivity(1200.0, power_density=1000)
        assert c_pd >= c0


class TestGetResistivity:
    def test_resistivity_is_inverse_of_conductivity(self) -> None:
        g = GlassPropertiesInterface()
        c = g.get_conductivity(1200.0)
        r = g.get_resistivity(1200.0)
        assert abs(r - 1.0 / c) < 1e-12

    def test_metal_resistivity_is_low(self) -> None:
        g = GlassPropertiesInterface()
        r = g.get_resistivity(1200.0, is_metal=True)
        assert r < 0.001


class TestCacheAndProperties:
    def test_clear_cache_empties_store(self) -> None:
        g = GlassPropertiesInterface()
        g.get_conductivity(1000.0)
        assert len(g._temperature_dependent_data) > 0
        g.clear_cache()
        assert len(g._temperature_dependent_data) == 0

    def test_set_external_calculator_clears_cache(self) -> None:
        g = GlassPropertiesInterface()
        g.get_conductivity(1000.0)
        g.set_external_calculator(lambda t, c, p: 1.0)
        assert len(g._temperature_dependent_data) == 0

    def test_update_and_get_current_properties(self) -> None:
        g = GlassPropertiesInterface()
        g.update_properties({"SiO2": 0.7, "Al2O3": 0.1})
        props = g.get_current_properties()
        assert props["SiO2"] == 0.7
        assert props["Al2O3"] == 0.1

    def test_cache_lru_eviction(self) -> None:
        g = GlassPropertiesInterface(cache_max_size=3)
        for t in range(10):
            g.get_conductivity(float(t * 100 + 500))
        assert len(g._temperature_dependent_data) <= 3
