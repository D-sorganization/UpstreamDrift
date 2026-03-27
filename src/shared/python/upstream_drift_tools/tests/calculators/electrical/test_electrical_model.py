"""test_electrical_model.py module."""

import numpy as np
import pytest
from upstream_drift_tools.calculators.electrical.config import ElectrodeConfig
from upstream_drift_tools.calculators.electrical.electrical_model import (
    ThreePhaseElectricalModelEnhanced,
)
from upstream_drift_tools.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)


class TestElectricalModel:
    @pytest.fixture
    def model(self):
        config = ElectrodeConfig()
        glass = GlassPropertiesInterface()
        return ThreePhaseElectricalModelEnhanced(config, glass)

    def test_initialization(self, model):
        assert len(model.electrode_positions) == 3
        np.testing.assert_allclose(
            model.electrode_positions, [0, 2.094395, 4.18879], rtol=1e-4
        )

    def test_calculate_system_state(self, model):
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

    def test_parallel_resistance(self, model):
        r1 = 100.0
        r2 = 100.0
        rp = model._parallel_resistance(r1, r2)
        assert rp == 50.0  # Parallel of two equal resistors is half

    def test_glass_conductivity(self):
        glass = GlassPropertiesInterface()

        # Test default Arrhenius behavior
        cond_1200 = glass.get_conductivity(1200.0)  # Base temp (1473.15 K)
        cond_1300 = glass.get_conductivity(1300.0)  # Higher temp

        assert cond_1300 > cond_1200  # Conductivity should increase with temp

        # Test metal conductivity
        cond_metal = glass.get_conductivity(1200.0, is_metal=True)
        assert cond_metal == 10000.0
