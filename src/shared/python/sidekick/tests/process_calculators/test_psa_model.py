from typing import Any

"""
Test suite for PSA Model.

This module tests the PSA calculation model against known Excel results
to ensure consistency across all implementations (Python core, Jupyter notebook, GUI).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sidekick.process_calculators.psa_package.psa_model import (
    PSAModel,
    calculate_o2_safety_analysis,
    calculate_sensitivity,
    get_flammability_status,
)

# Excel reference values for validation
EXCEL_REFERENCE = {
    "h2_recovery_pct": 79.47548460661345,
    "h2_purity_pct": 99.99943514578405,
    "total_net_product_scfm": 280.4266315767112,
    "total_exhaust_scfm": 819.5733684232889,
    "total_s2_tail_vent_scfm": 0.0,
    "s2_tail_h2_pct": 75.75448502421742,
    "s2_tail_o2_pct": 1.9744831882956226,
    # Component-level flows for H2 (index 0)
    "fresh_feed_h2": 352.8447155284472,
    "mixed_feed_h2": 402.33148862992834,
    "exhaust_h2": 72.4196679533871,
    "interstage_h2": 329.91182067654125,
    "s2_tail_h2": 49.486773101481184,
    "net_product_h2": 280.4250475750601,
    # Component-level flows for O2 (index 5)
    "fresh_feed_o2": 5.499450054994501,
    "mixed_feed_o2": 6.789285257499516,
    "exhaust_o2": 5.499321058574608,
    "interstage_o2": 1.289964198924908,
    "s2_tail_o2": 1.2898352025050153,
    "net_product_o2": 0.00012899641989272403,
}


class TestPSAModelBaseCase:
    """Test PSA model against Excel base case results."""

    @pytest.fixture
    def base_model(self) -> PSAModel:
        """Create base case model matching Excel defaults."""
        return PSAModel(
            total_feed_scfm=1100.0,
            s2_tail_recycle_frac=1.0,
            product_recycle_frac=0.0,
        )

    @pytest.fixture
    def base_results(self, base_model: PSAModel) -> Any:
        """Calculate base case results."""
        return base_model.calculate()

    def test_h2_recovery(self, base_results) -> None:
        """Test H2 recovery matches Excel."""
        assert_allclose(
            base_results.h2_recovery_pct,
            EXCEL_REFERENCE["h2_recovery_pct"],
            rtol=1e-10,
            err_msg="H2 recovery does not match Excel",
        )

    def test_h2_purity(self, base_results) -> None:
        """Test H2 purity matches Excel."""
        assert_allclose(
            base_results.h2_purity_pct,
            EXCEL_REFERENCE["h2_purity_pct"],
            rtol=1e-10,
            err_msg="H2 purity does not match Excel",
        )

    def test_net_product_flow(self, base_results) -> None:
        """Test net product flow matches Excel."""
        assert_allclose(
            base_results.total_net_product_scfm,
            EXCEL_REFERENCE["total_net_product_scfm"],
            rtol=1e-10,
            err_msg="Net product flow does not match Excel",
        )

    def test_exhaust_flow(self, base_results) -> None:
        """Test exhaust flow matches Excel."""
        assert_allclose(
            base_results.total_exhaust_scfm,
            EXCEL_REFERENCE["total_exhaust_scfm"],
            rtol=1e-10,
            err_msg="Exhaust flow does not match Excel",
        )

    def test_s2_tail_vent_flow(self, base_results) -> None:
        """Test S2 tail vent flow (should be 0 at 100% recycle)."""
        assert_allclose(
            base_results.total_s2_tail_vent_scfm,
            EXCEL_REFERENCE["total_s2_tail_vent_scfm"],
            atol=1e-10,
            err_msg="S2 tail vent flow does not match Excel",
        )

    def test_mass_balance(self, base_results) -> None:
        """Test mass balance closure."""
        assert abs(base_results.mass_balance_error) < 1e-10, (
            f"Mass balance error too large: {base_results.mass_balance_error}"
        )

    def test_s2_tail_h2_pct(self, base_results) -> None:
        """Test S2 tail H2 percentage matches Excel."""
        assert_allclose(
            base_results.s2_tail_h2_pct,
            EXCEL_REFERENCE["s2_tail_h2_pct"],
            rtol=1e-10,
            err_msg="S2 tail H2% does not match Excel",
        )

    def test_s2_tail_o2_pct(self, base_results) -> None:
        """Test S2 tail O2 percentage matches Excel."""
        assert_allclose(
            base_results.s2_tail_o2_pct,
            EXCEL_REFERENCE["s2_tail_o2_pct"],
            rtol=1e-10,
            err_msg="S2 tail O2% does not match Excel",
        )


class TestPSAModelH2Flows:
    """Test H2 component flows against Excel."""

    @pytest.fixture
    def base_results(self) -> Any:
        """Calculate base case results."""
        model = PSAModel()
        return model.calculate()

    def test_fresh_feed_h2(self, base_results) -> None:
        """Test H2 fresh feed flow."""
        assert_allclose(
            base_results.flows.fresh_feed[0],
            EXCEL_REFERENCE["fresh_feed_h2"],
            rtol=1e-10,
        )

    def test_mixed_feed_h2(self, base_results) -> None:
        """Test H2 mixed feed flow."""
        assert_allclose(
            base_results.flows.mixed_feed[0],
            EXCEL_REFERENCE["mixed_feed_h2"],
            rtol=1e-10,
        )

    def test_exhaust_h2(self, base_results) -> None:
        """Test H2 exhaust flow."""
        assert_allclose(
            base_results.flows.exhaust[0],
            EXCEL_REFERENCE["exhaust_h2"],
            rtol=1e-10,
        )

    def test_interstage_h2(self, base_results) -> None:
        """Test H2 interstage flow."""
        assert_allclose(
            base_results.flows.interstage[0],
            EXCEL_REFERENCE["interstage_h2"],
            rtol=1e-10,
        )

    def test_s2_tail_h2(self, base_results) -> None:
        """Test H2 S2 tail flow."""
        assert_allclose(
            base_results.flows.s2_tail[0],
            EXCEL_REFERENCE["s2_tail_h2"],
            rtol=1e-10,
        )

    def test_net_product_h2(self, base_results) -> None:
        """Test H2 net product flow."""
        assert_allclose(
            base_results.flows.net_product[0],
            EXCEL_REFERENCE["net_product_h2"],
            rtol=1e-10,
        )


class TestPSAModelO2Flows:
    """Test O2 component flows against Excel."""

    @pytest.fixture
    def base_results(self) -> Any:
        """Calculate base case results."""
        model = PSAModel()
        return model.calculate()

    def test_fresh_feed_o2(self, base_results) -> None:
        """Test O2 fresh feed flow."""
        assert_allclose(
            base_results.flows.fresh_feed[5],
            EXCEL_REFERENCE["fresh_feed_o2"],
            rtol=1e-10,
        )

    def test_mixed_feed_o2(self, base_results) -> None:
        """Test O2 mixed feed flow."""
        assert_allclose(
            base_results.flows.mixed_feed[5],
            EXCEL_REFERENCE["mixed_feed_o2"],
            rtol=1e-10,
        )

    def test_exhaust_o2(self, base_results) -> None:
        """Test O2 exhaust flow."""
        assert_allclose(
            base_results.flows.exhaust[5],
            EXCEL_REFERENCE["exhaust_o2"],
            rtol=1e-10,
        )

    def test_interstage_o2(self, base_results) -> None:
        """Test O2 interstage flow."""
        assert_allclose(
            base_results.flows.interstage[5],
            EXCEL_REFERENCE["interstage_o2"],
            rtol=1e-10,
        )

    def test_s2_tail_o2(self, base_results) -> None:
        """Test O2 S2 tail flow."""
        assert_allclose(
            base_results.flows.s2_tail[5],
            EXCEL_REFERENCE["s2_tail_o2"],
            rtol=1e-10,
        )

    def test_net_product_o2(self, base_results) -> None:
        """Test O2 net product flow."""
        assert_allclose(
            base_results.flows.net_product[5],
            EXCEL_REFERENCE["net_product_o2"],
            rtol=1e-10,
        )


class TestPSAModelSensitivity:
    """Test sensitivity analysis functions."""

    def test_sensitivity_h2_recovery_increases_with_recycle(self) -> None:
        """Test that H2 recovery increases with S2 tail recycle."""
        s2_range = np.array([0.0, 0.5, 1.0])
        sensitivity = calculate_sensitivity(
            s2_tail_recycle_range=s2_range,
            product_recycle_range=np.array([0.0]),
        )

        # H2 recovery should increase with recycle
        assert sensitivity["h2_recovery"][0, 0] < sensitivity["h2_recovery"][1, 0]
        assert sensitivity["h2_recovery"][1, 0] < sensitivity["h2_recovery"][2, 0]

    def test_sensitivity_product_recycle_reduces_output(self) -> None:
        """Test that product recycle reduces net output."""
        prod_range = np.array([0.0, 0.2, 0.4])
        sensitivity = calculate_sensitivity(
            s2_tail_recycle_range=np.array([1.0]),
            product_recycle_range=prod_range,
        )

        # Net product should decrease with product recycle
        assert sensitivity["net_product"][0, 0] > sensitivity["net_product"][0, 1]
        assert sensitivity["net_product"][0, 1] > sensitivity["net_product"][0, 2]

    def test_sensitivity_output_shapes(self) -> None:
        """Test that sensitivity output arrays have correct shapes."""
        s2_range = np.linspace(0, 1, 11)
        prod_range = np.array([0.0, 0.1, 0.2])

        sensitivity = calculate_sensitivity(
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
        )

        assert sensitivity["h2_recovery"].shape == (11, 3)
        assert sensitivity["h2_purity"].shape == (11, 3)
        assert sensitivity["net_product"].shape == (11, 3)
        assert sensitivity["s2_tail_o2"].shape == (11, 3)

    def test_sensitivity_defaults(self) -> None:
        """Test sensitivity calculation with default None arguments."""
        sensitivity = calculate_sensitivity(total_feed=1000.0)
        assert sensitivity["h2_recovery"].shape == (11, 1)


class TestO2SafetyAnalysis:
    """Test O2 safety analysis functions."""

    def test_o2_increases_with_lower_s1_removal(self) -> None:
        """Test that S2 tail O2 increases with lower S1 removal efficiency."""
        s1_range = np.array([95.0, 80.0, 50.0])
        o2_analysis = calculate_o2_safety_analysis(
            inlet_o2_pcts=np.array([0.5]),
            stage1_o2_removal_range=s1_range,
        )

        # S2 tail O2 should increase as S1 removal decreases
        assert o2_analysis["s2_tail_o2"][0, 0] < o2_analysis["s2_tail_o2"][1, 0]
        assert o2_analysis["s2_tail_o2"][1, 0] < o2_analysis["s2_tail_o2"][2, 0]

    def test_o2_increases_with_inlet_o2(self) -> None:
        """Test that S2 tail O2 increases with inlet O2."""
        inlet_o2 = np.array([0.5, 2.0, 5.0])
        o2_analysis = calculate_o2_safety_analysis(
            inlet_o2_pcts=inlet_o2,
            stage1_o2_removal_range=np.array([80.0]),
        )

        # S2 tail O2 should increase with inlet O2
        assert o2_analysis["s2_tail_o2"][0, 0] < o2_analysis["s2_tail_o2"][0, 1]
        assert o2_analysis["s2_tail_o2"][0, 1] < o2_analysis["s2_tail_o2"][0, 2]

    def test_current_operation_safe(self) -> None:
        """Test that current operation (0.5% inlet, 81% removal) is safe."""
        o2_analysis = calculate_o2_safety_analysis(
            inlet_o2_pcts=np.array([0.5]),
            stage1_o2_removal_range=np.array([81.0]),
        )

        # S2 tail O2 should be below danger threshold (2%)
        assert o2_analysis["s2_tail_o2"][0, 0] < 2.0

    def test_o2_analysis_defaults(self) -> None:
        """Test O2 safety analysis with default None arguments."""
        o2_analysis = calculate_o2_safety_analysis(total_feed=1000.0)
        assert len(o2_analysis["inlet_o2_pcts"]) == 4
        assert len(o2_analysis["stage1_o2_removal"]) == 10


class TestFlammabilityStatus:
    """Test flammability status determination."""

    def test_safe_low_o2(self) -> None:
        """Test Safe-Low O2 status."""
        status = get_flammability_status(h2_pct=50.0, o2_pct=0.05)
        assert status == "Safe-Low O2"

    def test_safe_below_lfl(self) -> None:
        """Test Safe-Below LFL status."""
        status = get_flammability_status(h2_pct=2.0, o2_pct=5.0)
        assert status == "Safe-Below LFL"

    def test_caution_rich(self) -> None:
        """Test Caution-Rich status."""
        status = get_flammability_status(h2_pct=80.0, o2_pct=1.0)
        assert status == "Caution-Rich"

    def test_flammable(self) -> None:
        """Test FLAMMABLE status."""
        status = get_flammability_status(h2_pct=30.0, o2_pct=1.0)
        assert status == "FLAMMABLE"

    def test_critical(self) -> None:
        """Test CRITICAL status."""
        status = get_flammability_status(h2_pct=30.0, o2_pct=5.0)
        assert status == "CRITICAL"


class TestPSAModelEdgeCases:
    """Test PSA model edge cases."""

    def test_zero_recycle(self) -> None:
        """Test model with zero recycle."""
        model = PSAModel(
            total_feed_scfm=1100.0,
            s2_tail_recycle_frac=0.0,
            product_recycle_frac=0.0,
        )
        results = model.calculate()

        # Should still produce valid results
        assert results.h2_recovery_pct > 0
        assert results.h2_purity_pct > 99.99
        assert results.total_net_product_scfm > 0
        assert abs(results.mass_balance_error) < 1e-10

    def test_full_product_recycle(self) -> None:
        """Test model with full product recycle (edge case)."""
        model = PSAModel(
            total_feed_scfm=1100.0,
            s2_tail_recycle_frac=1.0,
            product_recycle_frac=0.99,  # Near full recycle
        )
        results = model.calculate()

        # Net product should be very small (1% of gross product)
        # H2 recovery drops significantly due to product recycle
        assert results.total_net_product_scfm < 20
        assert results.h2_recovery_pct < 10  # Most H2 recycled back
        assert abs(results.mass_balance_error) < 1e-10

    def test_different_feed_rate(self) -> None:
        """Test model with different feed rate."""
        model = PSAModel(
            total_feed_scfm=2200.0,  # Double the default
            s2_tail_recycle_frac=1.0,
            product_recycle_frac=0.0,
        )
        results = model.calculate()

        # Results should scale proportionally
        base_model = PSAModel()
        base_results = base_model.calculate()

        assert_allclose(
            results.total_net_product_scfm,
            base_results.total_net_product_scfm * 2,
            rtol=1e-10,
        )
        # Recovery percentage should be the same
        assert_allclose(
            results.h2_recovery_pct,
            base_results.h2_recovery_pct,
            rtol=1e-10,
        )

    def test_composition_sums_to_100(self) -> None:
        """Test that all stream compositions sum to 100%."""
        model = PSAModel()
        results = model.calculate()

        for stream_name in [
            "fresh_feed",
            "mixed_feed",
            "exhaust",
            "interstage",
            "gross_product",
            "net_product",
        ]:
            comp = getattr(results.compositions, stream_name)
            assert_allclose(
                np.sum(comp),
                100.0,
                rtol=1e-10,
                err_msg=f"{stream_name} composition does not sum to 100%",
            )


class TestPSAModelConsistency:
    """Test consistency between different calculation paths."""

    def test_flow_conservation_per_component(self) -> None:
        """Test mass conservation for each component."""
        model = PSAModel()
        results = model.calculate()

        for i in range(len(results.component_names)):
            # Fresh feed = Exhaust + S2 Tail Vent + Net Product
            balance = (
                results.flows.fresh_feed[i]
                - results.flows.exhaust[i]
                - results.flows.s2_tail_vent[i]
                - results.flows.net_product[i]
            )
            assert abs(balance) < 1e-10, (
                f"Mass balance error for {results.component_names[i]}: {balance}"
            )

    def test_mixed_feed_balance(self) -> None:
        """Test mixed feed balance."""
        model = PSAModel()
        results = model.calculate()

        for i in range(len(results.component_names)):
            # Mixed feed = Fresh feed + S2 tail recycle + Product recycle
            calculated_mixed = (
                results.flows.fresh_feed[i]
                + results.flows.s2_tail_recycle[i]
                + results.flows.product_recycle[i]
            )
            assert_allclose(
                results.flows.mixed_feed[i],
                calculated_mixed,
                rtol=1e-10,
                err_msg=f"Mixed feed balance error for {results.component_names[i]}",
            )

    def test_interstage_balance(self) -> None:
        """Test interstage balance."""
        model = PSAModel()
        results = model.calculate()

        for i in range(len(results.component_names)):
            # Interstage = Mixed feed - Exhaust
            calculated_interstage = (
                results.flows.mixed_feed[i] - results.flows.exhaust[i]
            )
            assert_allclose(
                results.flows.interstage[i],
                calculated_interstage,
                rtol=1e-10,
            )

    def test_gross_product_balance(self) -> None:
        """Test gross product balance."""
        model = PSAModel()
        results = model.calculate()

        for i in range(len(results.component_names)):
            # Gross product = Interstage - S2 tail
            calculated_gross = results.flows.interstage[i] - results.flows.s2_tail[i]
            assert_allclose(
                results.flows.gross_product[i],
                calculated_gross,
                rtol=1e-10,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
