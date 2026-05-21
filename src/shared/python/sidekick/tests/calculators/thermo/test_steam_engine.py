from unittest.mock import patch

from upstream_drift_tools.calculators.thermo.steam_engine import (
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
    p = engine._buck_equation(100.0)
    assert p > 0


def test_calculate_water_vapor_pressure() -> None:
    engine = SteamCalculationEngine()
    p_antoine = engine.calculate_water_vapor_pressure(100.0, method="antoine")
    p_buck = engine.calculate_water_vapor_pressure(100.0, method="buck")

    assert p_antoine > 0
    assert p_buck > 0


def test_calculate_dew_point() -> None:
    engine = SteamCalculationEngine()
    partial_p = engine.calculate_water_vapor_pressure(50.0, method="buck")
    dp = engine.calculate_dew_point(partial_p, 101325.0)

    # Dew point should be close to 50
    assert 49.0 < dp < 51.0


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


def test_calculate_properties_simplified() -> None:
    engine = SteamCalculationEngine()
    props = engine.calculate_properties(400.0, 101325.0, engine="simplified")

    assert props.temperature == 400.0
    assert props.pressure == 101325.0
    assert props.enthalpy > 0
    assert props.entropy > 0


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
