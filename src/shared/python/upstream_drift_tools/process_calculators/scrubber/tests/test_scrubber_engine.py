"""Tests for Scrubber Calculation Engine."""

import unittest

from ..engine.scrubber_engine import ScrubberEngine
from ..models.scrubber_models import ScrubberInputs


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
        # Should probably return zero or handle gracefully.
        # In current implementation, it might produce results but they should be small/zero.
        results = ScrubberEngine.calculate(zero_inputs)
        self.assertEqual(results.column_diameter_m, 0.0)
        self.assertEqual(results.packed_height_m, 0.0)


if __name__ == "__main__":
    unittest.main()
