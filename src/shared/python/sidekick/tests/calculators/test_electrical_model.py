import numpy as np
from sidekick.calculators.electrical.config import ElectrodeConfig
from sidekick.calculators.electrical.electrical_model import (
    ThreePhaseElectricalModelEnhanced,
)
from sidekick.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)


def test_electrical_model_initialization() -> None:
    config = ElectrodeConfig()
    glass = GlassPropertiesInterface()
    model = ThreePhaseElectricalModelEnhanced(config, glass)

    assert len(model.electrode_positions) == 3


def test_electrical_model_calculation() -> None:
    config = ElectrodeConfig()
    glass = GlassPropertiesInterface()
    model = ThreePhaseElectricalModelEnhanced(config, glass)

    depths = np.array([5.0, 5.0, 5.0])
    voltages = np.array([100.0, 100.0, 100.0])

    result = model.calculate_system_state(
        depths=depths,
        bath_diameter=120.0,
        tip_diameter=24.0,
        metal_depth=2.0,
        k_factors={"K_tt": 1.0, "K_vert": 1.0},
        bath_temperature=1200.0,
        voltages=voltages,
        conductive_height=2.0,
        metal_conductive=True,
    )

    # Check basic structure
    assert "resistances" in result
    assert "current_paths" in result
    assert "current_distribution" in result
    assert "electrode_positions" in result
    assert "actual_currents" in result

    # Check values
    assert len(result["electrode_positions"]) == 3
    assert result["resistances"]["1-2"] > 0
    assert result["actual_currents"]["1-2"] > 0
    assert "direct_glass_fraction" in result["current_distribution"]["1-2"]


def test_electrical_model_no_metal() -> None:
    config = ElectrodeConfig()
    glass = GlassPropertiesInterface()
    model = ThreePhaseElectricalModelEnhanced(config, glass)

    depths = np.array([5.0, 5.0, 5.0])

    result = model.calculate_system_state(
        depths=depths,
        bath_diameter=120.0,
        tip_diameter=24.0,
        metal_depth=2.0,
        k_factors={"K_tt": 1.0, "K_vert": 1.0},
        bath_temperature=1200.0,
        metal_conductive=False,
    )

    # Since metal is not conductive, via metal resistance is inf
    assert result["current_paths"]["1-2"]["via_metal"] == float("inf")
    assert result["current_paths"]["1-2"]["metal_fraction"] == 0.0
    assert result["current_paths"]["1-2"]["direct_fraction"] == 1.0
