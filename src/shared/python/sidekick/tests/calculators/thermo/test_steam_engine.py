from unittest.mock import patch

import pytest
from sidekick.calculators.thermo.steam_engine import (
    SteamCalculationEngine,
    SteamProperties,
)


def test_engine_init() -> None:
    engine = SteamCalculationEngine()

    # We don't guarantee Cantera/CoolProp are installed in the test env,
    # so we just test that init doesn't crash
    assert hasattr(engine, "water")
    assert hasattr(engine, "initialized")


def test_select_best_engine() -> None:
    engine = SteamCalculationEngine()

    with patch(
        "upstream_drift_tools.calculators.thermo.steam_engine.COOLPROP_AVAILABLE", True
    ):
        assert engine._select_best_engine("coolprop") == "coolprop"

    with (
        patch(
            "upstream_drift_tools.calculators.thermo.steam_engine.COOLPROP_AVAILABLE",
            False,
        ),
        patch(
            "upstream_drift_tools.calculators.thermo.steam_engine.CANTERA_AVAILABLE",
            False,
        ),
    ):
        assert engine._select_best_engine("auto") == "simplified"


def test_antoine_equation() -> None:
    engine = SteamCalculationEngine()
    # At 100 °C, pressure should be approximately 1 atm (101325 Pa)
    p = engine._antoine_equation(100.0)
    assert 100000.0 < p < 103000.0


def test_buck_equation() -> None:
    engine = SteamCalculationEngine()
    assert engine._buck_equation(0.01) == pytest.approx(611.66, rel=0.005)
    assert engine._buck_equation(25.0) == pytest.approx(3169.9, rel=0.005)
    assert engine._buck_equation(100.0) == pytest.approx(101417.0, rel=0.005)


def test_calculate_water_vapor_pressure() -> None:
    engine = SteamCalculationEngine()
    p_antoine = engine.calculate_water_vapor_pressure(100.0, method="antoine")
    p_buck = engine.calculate_water_vapor_pressure(100.0, method="buck")

    assert p_antoine > 0
    assert p_buck > 0


def test_calculate_dew_point() -> None:
    engine = SteamCalculationEngine()
    partial_p = 3169.9
    dp = engine.calculate_dew_point(partial_p, 101325.0)

    assert dp == pytest.approx(25.0, abs=0.1)


def test_buck_and_antoine_are_consistent_from_5_to_100c() -> None:
    engine = SteamCalculationEngine()

    for temperature_c in (5.0, 25.0, 50.0, 75.0, 100.0):
        p_buck = engine.calculate_water_vapor_pressure(temperature_c, method="buck")
        p_antoine = engine.calculate_water_vapor_pressure(
            temperature_c, method="antoine"
        )
        assert p_buck == pytest.approx(p_antoine, rel=0.02)


def test_calculate_saturated_simplified_from_temp() -> None:
    engine = SteamCalculationEngine()
    # 100 C = 373.15 K
    props = engine.calculate_saturated_properties_from_temperature(
        373.15, engine="simplified"
    )

    assert props.temperature == 373.15
    assert 100000.0 < props.pressure < 103000.0


def test_calculate_saturated_simplified_from_pressure() -> None:
    engine = SteamCalculationEngine()
    # 1 atm ~ 101325 Pa -> expect ~ 373.15 K
    props = engine.calculate_saturated_properties_from_pressure(
        101325.0, engine="simplified"
    )

    assert 370.0 < props.temperature < 375.0
    assert props.pressure == 101325.0


def test_saturation_temperature_rejects_out_of_range_input() -> None:
    engine = SteamCalculationEngine()

    with pytest.raises(ValueError, match="saturation bounds"):
        engine.calculate_saturated_properties_from_temperature(
            200.0, engine="simplified"
        )


def test_saturation_pressure_rejects_nonphysical_input() -> None:
    engine = SteamCalculationEngine()

    with pytest.raises(ValueError, match="saturation bounds"):
        engine.calculate_saturated_properties_from_pressure(-5.0, engine="simplified")


def test_saturation_pressure_round_trip_uses_consistent_curve() -> None:
    engine = SteamCalculationEngine()

    props = engine.calculate_saturated_properties_from_pressure(
        101325.0, engine="simplified"
    )

    round_tripped = engine._calculate_saturated_simplified_from_temp(props.temperature)

    assert round_tripped.pressure == pytest.approx(props.pressure, rel=2e-5)


def test_calculate_properties_simplified() -> None:
    engine = SteamCalculationEngine()
    props = engine.calculate_properties(400.0, 101325.0, engine="simplified")

    assert props.temperature == 400.0
    assert props.pressure == 101325.0
    assert props.enthalpy > 0
    assert props.entropy > 0


def test_simplified_vapor_entropy_includes_pressure_dependence() -> None:
    engine = SteamCalculationEngine()

    low_pressure = engine.calculate_properties(573.15, 101325.0, engine="simplified")
    high_pressure = engine.calculate_properties(573.15, 1.0e6, engine="simplified")
    saturated_reference = engine.calculate_properties(
        373.15, 101325.0, engine="simplified"
    )

    assert low_pressure.entropy / 1000.0 == pytest.approx(8.217, rel=0.03)
    assert high_pressure.entropy / 1000.0 == pytest.approx(7.123, rel=0.03)
    assert saturated_reference.entropy / 1000.0 == pytest.approx(7.354, rel=0.03)
    assert high_pressure.entropy < low_pressure.entropy


def test_steam_properties_to_dict() -> None:
    props = SteamProperties(
        temperature=300.0,
        pressure=100000.0,
        density=1.0,
        specific_volume=1.0,
        enthalpy=2000.0,
        entropy=10.0,
        internal_energy=1900.0,
        cp=2.0,
        cv=1.5,
        speed_of_sound=300.0,
        thermal_conductivity=0.6,
        dynamic_viscosity=1e-5,
        kinematic_viscosity=1e-5,
        quality=1.0,
        phase="vapor",
    )

    d = props.to_dict()
    assert d["Temperature (K)"] == 300.0
    assert d["Pressure (Pa)"] == 100000.0
    assert d["Phase"] == "vapor"
