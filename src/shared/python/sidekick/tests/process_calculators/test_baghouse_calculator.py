import pytest
from upstream_drift_tools.process_calculators.baghouse_calculator import (
    BaghouseCalculator,
    BaghouseResult,
)


@pytest.fixture
def baghouse_calc() -> BaghouseCalculator:
    return BaghouseCalculator()


def test_calculate_baghouse(baghouse_calc: BaghouseCalculator) -> None:
    result = baghouse_calc.calculate(
        gas_flow_kg_s=1.0,
        inlet_temp_k=400.0,
        pressure_pa=101325.0,
        composition={"CH4": 0.8, "C2H6": 0.2},
        solid_carbon_in_kg_hr=10.0,
        ash_in_kg_hr=5.0,
        carbon_removal_efficiency=0.99,
        ash_removal_efficiency=0.95,
        heat_loss_w=5000.0,
        drum_volume_m3=2.0,
        solid_density_kg_m3=800.0,
        bag_area_ft2=1000.0,
    )

    assert isinstance(result, BaghouseResult)
    assert result.carbon_removed_rate > 0.0
    assert result.ash_removed_rate > 0.0
    assert result.total_solids_removed_rate > 0.0
    assert result.drum_fill_time_hours > 0.0

    comp = result.ash_stream_composition
    assert "carbon_fraction" in comp
    assert "ash_fraction" in comp
    assert comp["carbon_fraction"] + comp["ash_fraction"] == pytest.approx(1.0)


def test_calculate_baghouse_zero_solids(baghouse_calc: BaghouseCalculator) -> None:
    result = baghouse_calc.calculate(
        gas_flow_kg_s=1.0,
        inlet_temp_k=400.0,
        pressure_pa=101325.0,
        composition={"CH4": 1.0},
        solid_carbon_in_kg_hr=0.0,
        ash_in_kg_hr=0.0,
        carbon_removal_efficiency=0.99,
        ash_removal_efficiency=0.95,
        heat_loss_w=5000.0,
        drum_volume_m3=2.0,
        solid_density_kg_m3=800.0,
        bag_area_ft2=1000.0,
    )

    assert result.carbon_removed_rate == 0.0
    assert result.ash_removed_rate == 0.0
    assert result.drum_fill_time_hours == float("inf")


def test_calculate_baghouse_invalid(baghouse_calc: BaghouseCalculator) -> None:
    with pytest.raises(AssertionError):
        baghouse_calc.calculate(
            gas_flow_kg_s=-1.0,
            inlet_temp_k=400.0,
            pressure_pa=101325.0,
            composition={"CH4": 1.0},
            solid_carbon_in_kg_hr=0.0,
            ash_in_kg_hr=0.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            heat_loss_w=5000.0,
            drum_volume_m3=2.0,
            solid_density_kg_m3=800.0,
            bag_area_ft2=1000.0,
        )
