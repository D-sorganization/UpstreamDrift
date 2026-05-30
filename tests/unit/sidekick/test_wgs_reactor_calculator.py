"""Tests for sidekick.process_calculators.wgs_reactor_calculator (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.process_calculators.wgs_reactor_calculator import (
    WGSReactorEngine,
)

# ---------------------------------------------------------------------------
# WGSReactorEngine.calculate_equilibrium_constant
# ---------------------------------------------------------------------------


class TestCalculateEquilibriumConstant:
    _ENGINE = WGSReactorEngine()

    def test_returns_positive_value(self) -> None:
        K = self._ENGINE.calculate_equilibrium_constant(600.0)
        assert K > 0.0

    def test_lower_temperature_higher_k(self) -> None:
        # WGS is exothermic — K decreases with temperature
        K_low = self._ENGINE.calculate_equilibrium_constant(400.0)
        K_high = self._ENGINE.calculate_equilibrium_constant(900.0)
        assert K_low > K_high

    def test_k_at_room_temperature_large(self) -> None:
        K = self._ENGINE.calculate_equilibrium_constant(298.15)
        assert K > 1.0

    def test_k_at_high_temperature_near_one(self) -> None:
        K = self._ENGINE.calculate_equilibrium_constant(1200.0)
        # At high T, exothermic reaction K < 1 (unfavorable)
        assert K < 1.0


# ---------------------------------------------------------------------------
# WGSReactorEngine.size_wgs_reactor
# ---------------------------------------------------------------------------


class TestSizeWgsReactor:
    _ENGINE = WGSReactorEngine()

    def test_wgs_reactor_calculator_returns_dict(self) -> None:
        result = self._ENGINE.size_wgs_reactor(100.0, 80.0, 600.0, "Fe")
        assert isinstance(result, dict)

    def test_required_keys_present(self) -> None:
        result = self._ENGINE.size_wgs_reactor(100.0, 80.0, 600.0, "Fe")
        for key in (
            "reactor_volume",
            "catalyst_volume",
            "diameter",
            "length",
            "heat_duty",
            "ghsv",
        ):
            assert key in result, f"Missing key: {key}"

    def test_reactor_volume_positive(self) -> None:
        result = self._ENGINE.size_wgs_reactor(100.0, 80.0, 600.0, "Fe")
        assert result["reactor_volume"] > 0.0

    def test_catalyst_volume_less_than_reactor(self) -> None:
        result = self._ENGINE.size_wgs_reactor(100.0, 80.0, 600.0, "Fe")
        assert result["catalyst_volume"] < result["reactor_volume"]

    def test_diameter_positive(self) -> None:
        result = self._ENGINE.size_wgs_reactor(100.0, 80.0, 600.0, "Fe")
        assert result["diameter"] > 0.0

    def test_larger_feed_rate_larger_reactor(self) -> None:
        small = self._ENGINE.size_wgs_reactor(10.0, 80.0, 600.0, "Fe")
        large = self._ENGINE.size_wgs_reactor(1000.0, 80.0, 600.0, "Fe")
        assert large["reactor_volume"] > small["reactor_volume"]

    def test_higher_conversion_higher_heat_duty(self) -> None:
        low = self._ENGINE.size_wgs_reactor(100.0, 50.0, 600.0, "Fe")
        high = self._ENGINE.size_wgs_reactor(100.0, 90.0, 600.0, "Fe")
        assert high["heat_duty"] > low["heat_duty"]


# ---------------------------------------------------------------------------
# WGSReactorEngine._prepare_initial_moles (static)
# ---------------------------------------------------------------------------


class TestPrepareInitialMoles:
    def test_co_and_steam_ratio(self) -> None:
        n_CO_0, n_H2O_0, n_CO2_0, n_H2_0, n_total_0 = (
            WGSReactorEngine._prepare_initial_moles({"CO": 1.0}, steam_ratio=1.0)
        )
        assert n_CO_0 == 1.0
        # H2O = 0 + 1.0*1.0 = 1.0
        assert abs(n_H2O_0 - 1.0) < 1e-10

    def test_total_moles_correct(self) -> None:
        _, _, _, _, n_total_0 = WGSReactorEngine._prepare_initial_moles(
            {"CO": 1.0, "CO2": 0.5, "H2": 0.2}, steam_ratio=0.0
        )
        # H2O = 0 + 1*0 = 0; total = 1 + 0 + 0.5 + 0.2 = 1.7
        assert abs(n_total_0 - 1.7) < 1e-10

    def test_empty_composition_zero_moles(self) -> None:
        _, _, _, _, n_total_0 = WGSReactorEngine._prepare_initial_moles(
            {}, steam_ratio=0.0
        )
        assert n_total_0 == 0.0
