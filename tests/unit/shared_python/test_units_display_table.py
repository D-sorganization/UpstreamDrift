"""Small lengths and the display-unit table in ``ui/units.py`` (#14646).

Clients that cannot import Python (the putting studio, the React page) need
the same conversions as the desktop. ``display_unit_table`` gives the factor
from each SI unit to the display unit and its suffix, built from the existing
helpers, and the API serves it. Small lengths (strike offsets) are shown in
millimetres or inches.
"""

from __future__ import annotations

import pytest

from src.shared.python.ui.units import (
    UnitSystem,
    display_unit_table,
    distance_suffix,
    from_display_small_length,
    small_length_suffix,
    speed_suffix,
    to_display_distance,
    to_display_small_length,
    to_display_speed,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("system", "shown", "suffix"),
    [(UnitSystem.METRIC, 10.0, " mm"), (UnitSystem.IMPERIAL, 0.3937007874, " in")],
)
def test_small_length_is_mm_or_inches(
    system: UnitSystem, shown: float, suffix: str
) -> None:
    assert to_display_small_length(0.01, system) == pytest.approx(shown, rel=1e-9)
    assert small_length_suffix(system) == suffix
    assert from_display_small_length(shown, system) == pytest.approx(0.01, rel=1e-9)


def test_inches_are_a_twelfth_of_the_shared_foot() -> None:
    feet = to_display_distance(0.01, UnitSystem.IMPERIAL, unit="ft")
    assert to_display_small_length(0.01, UnitSystem.IMPERIAL) == pytest.approx(
        12.0 * feet, rel=1e-12
    )


@pytest.mark.parametrize("system", list(UnitSystem))
def test_table_is_built_from_the_helpers(system: UnitSystem) -> None:
    assert display_unit_table(system) == {
        "distance": {
            "factor": to_display_distance(1.0, system, unit="ft"),
            "suffix": distance_suffix(system, unit="ft"),
        },
        "speed": {
            "factor": to_display_speed(1.0, system),
            "suffix": speed_suffix(system),
        },
        "small_length": {
            "factor": to_display_small_length(1.0, system),
            "suffix": small_length_suffix(system),
        },
    }


def test_small_length_rejects_a_non_finite_value() -> None:
    with pytest.raises(ValueError, match="finite"):
        to_display_small_length(float("nan"), UnitSystem.METRIC)
