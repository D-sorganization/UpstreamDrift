from typing import Any

import pytest
from upstream_drift_tools.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)


def test_glass_interface_initialization() -> None:
    interface = GlassPropertiesInterface()
    assert interface.get_current_properties() == {}

    interface.update_properties({"test": 123})
    assert interface.get_current_properties() == {"test": 123}


def test_glass_interface_conductivity() -> None:
    interface = GlassPropertiesInterface()

    # Test metal conductivity
    cond_metal = interface.get_conductivity(1000.0, is_metal=True)
    assert cond_metal == 10000.0

    # Test glass conductivity
    cond_glass_1 = interface.get_conductivity(
        1200.0
    )  # Matches reference temp 1473.15 K
    assert cond_glass_1 == pytest.approx(1.0, abs=0.01)

    # Test caching
    cond_glass_2 = interface.get_conductivity(1200.0)
    assert cond_glass_1 == cond_glass_2

    interface.clear_cache()


def test_glass_interface_resistivity() -> None:
    interface = GlassPropertiesInterface()
    res = interface.get_resistivity(1200.0)
    assert res == pytest.approx(1.0, abs=0.01)


def test_glass_interface_external_calculator() -> None:
    def ext_calc(t, c, p) -> Any:
        return 50.0

    interface = GlassPropertiesInterface(external_calculator=ext_calc)

    cond = interface.get_conductivity(1000.0, composition={"SiO2": 1.0})
    assert cond == 50.0

    interface.set_external_calculator(lambda t, c, p: 100.0)
    cond2 = interface.get_conductivity(1000.0)
    assert cond2 == 100.0
