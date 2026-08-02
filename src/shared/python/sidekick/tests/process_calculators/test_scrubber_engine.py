"""Tests for Scrubber Calculation Engine."""

import unittest

from sidekick.process_calculators.scrubber.engine.scrubber_engine import (
    ScrubberEngine,
)
from sidekick.process_calculators.scrubber.models.scrubber_models import (
    ScrubberInputs,
)


class TestScrubberEngine(unittest.TestCase):
    """Test suite for ScrubberEngine."""

    def setUp(self) -> None:
        """Set up standard test inputs."""
        self.standard_inputs = ScrubberInputs(
            gas_flow_kg_hr=10000,
            inlet_temp_c=200,
            pressure_bar=1.5,
            molecular_weight=22,
            target_outlet_temp_c=38,
            packing_name="Metal Pall Rings",
            percent_of_flood=70,
            height_safety_factor=1.2,
            lg_ratio=3.0,
            caustic_concentration_wt_pct=20,
            cooling_water_inlet_temp_c=25,
            kla_hr=200,
            acid_gas_composition_ppmv={
                "HCl": 500,
                "SO2": 200,
                "H2S": 1000,
                "HF": 100,
            },
            acid_gas_removal_pct={
                "HCl": 99.0,
                "SO2": 95.0,
                "H2S": 90.0,
                "HF": 99.0,
            },
        )

    def test_standard_calculation(self) -> None:
        """Test calculation with standard inputs."""
        results = ScrubberEngine.calculate(self.standard_inputs)

        # Check basic results existence and types
        self.assertGreater(results.column_diameter_m, 0.5)
        self.assertLess(results.column_diameter_m, 5.0)
        self.assertGreater(results.packed_height_m, 0.1)
        self.assertGreater(results.pressure_drop_kpa, 0.0)
        self.assertGreater(results.naoh_pure_kg_hr, 0.0)
        self.assertGreater(results.total_heat_duty_kw, 0.0)

        # Check acid gas details
        self.assertEqual(len(results.acid_gas_details), 4)
        for detail in results.acid_gas_details:
            self.assertIn("name", detail)
            self.assertGreater(detail["ntu"], 0)
            self.assertGreater(detail["removed_kg_hr"], 0)

    def test_invalid_packing(self) -> None:
        """Test error handling for invalid packing name."""
        invalid_inputs = ScrubberInputs(
            **{**self.standard_inputs.__dict__, "packing_name": "Invalid Packing"}
        )
        with self.assertRaises(ValueError):
            ScrubberEngine.calculate(invalid_inputs)

    def test_zero_flow(self) -> None:
        """Test behavior with zero gas flow."""
        zero_inputs = ScrubberInputs(
            **{**self.standard_inputs.__dict__, "gas_flow_kg_hr": 0}
        )
        results = ScrubberEngine.calculate(zero_inputs)
        self.assertEqual(results.column_diameter_m, 0.0)
        self.assertEqual(results.packed_height_m, 0.0)

    def test_calculate_column_sizing_invalid_packing(self) -> None:
        """Test column sizing with unknown packing."""
        inputs = ScrubberInputs(
            **{**self.standard_inputs.__dict__, "packing_name": "UnknownXYZ"}
        )
        with self.assertRaisesRegex(ValueError, "Unknown packing type: UnknownXYZ"):
            ScrubberEngine._calculate_column_sizing(inputs, 1.2)

    def test_calculate_column_sizing_valid(self) -> None:
        """Test column sizing with valid packing."""
        (
            diameter_m,
            actual_area,
            gas_mass_flux,
            liquid_mass_flux,
            design_velocity,
            warnings,
        ) = ScrubberEngine._calculate_column_sizing(self.standard_inputs, 1.2)
        self.assertGreater(diameter_m, 0)
        self.assertGreater(actual_area, 0)
        self.assertGreater(gas_mass_flux, 0)
        self.assertGreater(liquid_mass_flux, 0)
        self.assertGreater(design_velocity, 0)

    def test_calculate_mass_transfer_valid(self) -> None:
        """Test mass transfer calculations."""
        # Need realistic gas/liquid mass flux
        packed_height, max_ntu, acid_gas_details, acid_gas_removed = (
            ScrubberEngine._calculate_mass_transfer(self.standard_inputs, 1.2, 1.0, 5.0)
        )
        self.assertGreater(packed_height, 0)
        self.assertGreater(max_ntu, 0)
        self.assertEqual(len(acid_gas_details), 4)

    def test_calculate_thermal_valid(self) -> None:
        """Test thermal calculations."""
        acid_gas_removed = {"HCl": 10.0, "HF": 1.0}
        naoh_pure, naoh_sol, heat_kw, cw_flow, warnings = (
            ScrubberEngine._calculate_thermal(self.standard_inputs, acid_gas_removed)
        )
        self.assertGreater(naoh_pure, 0)
        self.assertGreater(naoh_sol, 0)
        self.assertGreater(heat_kw, 0)


if __name__ == "__main__":
    unittest.main()
