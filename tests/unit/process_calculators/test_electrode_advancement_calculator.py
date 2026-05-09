"""Tests for upstream_drift_tools.process_calculators.electrode_advancement_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
    ElectrodeAdvancementCalculator,
)


class TestElectrodeAdvancementCalculator:
    def test_electrode_advancement_calculator_construction(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        assert calc is not None

    def test_default_consumption_rate(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        assert calc.consumption_rate == pytest.approx(0.5)

    def test_calculate_consumption_basic(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        # rate=0.5, current=10 kA, time=2 hrs → 0.5 * 10 * 2 = 10
        result = calc.calculate_consumption(current_ka=10.0, time_hrs=2.0)
        assert result == pytest.approx(10.0)

    def test_calculate_consumption_zero_current(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        result = calc.calculate_consumption(current_ka=0.0, time_hrs=5.0)
        assert result == pytest.approx(0.0)

    def test_calculate_consumption_zero_time(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        result = calc.calculate_consumption(current_ka=5.0, time_hrs=0.0)
        assert result == pytest.approx(0.0)

    def test_calculate_consumption_proportional_to_current(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        low = calc.calculate_consumption(current_ka=5.0, time_hrs=1.0)
        high = calc.calculate_consumption(current_ka=10.0, time_hrs=1.0)
        assert high == pytest.approx(2 * low)

    def test_calculate_consumption_proportional_to_time(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        short = calc.calculate_consumption(current_ka=5.0, time_hrs=1.0)
        long_ = calc.calculate_consumption(current_ka=5.0, time_hrs=4.0)
        assert long_ == pytest.approx(4 * short)

    def test_consumption_rate_modifiable(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        calc.consumption_rate = 1.0
        result = calc.calculate_consumption(current_ka=3.0, time_hrs=2.0)
        assert result == pytest.approx(6.0)
