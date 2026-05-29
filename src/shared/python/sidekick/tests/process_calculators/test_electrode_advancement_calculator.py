from sidekick.process_calculators.electrode_advancement_calculator import (
    ElectrodeAdvancementCalculator,
)


def test_electrode_advancement() -> None:
    calc = ElectrodeAdvancementCalculator()

    assert calc.consumption_rate == 0.5

    cons = calc.calculate_consumption(current_ka=10.0, time_hrs=2.0)
    assert cons == 10.0
