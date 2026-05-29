# ruff: noqa: E501
from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
    calculate_pressure_drop,
    compare_friction_methods,
    list_fittings,
    list_flow_units,
    list_gas_components,
    list_materials,
    list_pipe_sizes,
    show_help,
    validate_inputs,
)


def test_helper_functions() -> None:
    # Just call them to ensure no crashes
    show_help()

    comps = list_gas_components()
    assert "H2" in comps

    fittings = list_fittings("elbow")
    assert any("elbow" in f.lower() for f in fittings)
    fittings_all = list_fittings()
    assert "tee_through_branch" in fittings_all

    sizes = list_pipe_sizes()
    assert "4" in sizes

    units = list_flow_units()
    assert "mass" in units
    assert "molar" in units

    mats = list_materials()
    assert "Commercial Steel" in mats


def test_compare_friction_methods() -> None:
    res = compare_friction_methods(1e5, 0.001)
    assert "colebrook" in res
    assert "swamee-jain" in res
    assert "churchill" in res
    assert "haaland" in res

    # Check transitional flow
    compare_friction_methods(3000, 0.001)
    # Check laminar flow
    compare_friction_methods(1000, 0.001)


def test_validate_inputs() -> None:
    # Test valid
    is_valid, errors, warnings = validate_inputs(
        pipe_size="4", pipe_schedule="40", flow_rate=1000, flow_unit="kg/h"
    )
    assert is_valid
    assert not errors

    # Test invalid diam
    is_valid, errors, warnings = validate_inputs(
        flow_rate=1000,
        flow_unit="kg/h",
        # missing diameter/size
    )
    assert not is_valid
    assert any("Must provide either pipe_diameter" in e for e in errors)

    # Test negative diam
    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=-1.0, flow_rate=1000, flow_unit="kg/h"
    )
    assert not is_valid
    assert any("pipe_diameter must be positive" in e for e in errors)

    # Test large diam
    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=3.0, flow_rate=1000, flow_unit="kg/h"
    )
    assert is_valid
    assert any("Large diameter" in w for w in warnings)

    # Test invalid flow
    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=0.1, flow_rate=-10, flow_unit="kg/h"
    )
    assert not is_valid

    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=0.1, flow_rate=None, flow_unit="kg/h"
    )
    assert not is_valid

    # Test invalid unit
    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=0.1, flow_rate=10, flow_unit="invalid_unit"
    )
    assert not is_valid

    # Test pressure temp
    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=0.1, flow_rate=10, flow_unit="kg/h", pressure=-1, temperature=-10
    )
    assert not is_valid

    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=0.1,
        flow_rate=10,
        flow_unit="kg/h",
        pressure=2000,
        temperature=100,
    )
    assert is_valid
    assert (
        sum(1 for w in warnings if "High pressure" in w or "Low temperature" in w) >= 2
    )

    # Composition and fittings
    is_valid, errors, warnings = validate_inputs(
        pipe_diameter=0.1,
        flow_rate=10,
        flow_unit="kg/h",
        temperature=3000,
        gas_composition={"H2": 1.5, "Unknown": 0.1},
        fittings=[{"type": "bad_fitting"}],
    )
    assert not is_valid
    assert any("Unknown gas components" in e for e in errors)
    assert any("sums to" in w for w in warnings)
    assert any("not in database" in w for w in warnings)


def test_calculate_pressure_drop() -> None:
    # Basic valid calculation
    res = calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=100.0,
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=10.0,
        temperature=500.0,
    )
    assert res is not None
    assert "pressure_drop_bar" in res
    assert res["pressure_drop_bar"] > 0

    # Test with fittings and validation error (raises ValueError)
    with pytest.raises(ValueError, match="Invalid inputs:"):
        calculate_pressure_drop(pipe_diameter=-1.0, flow_rate=1000.0, flow_unit="kg/h")
