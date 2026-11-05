"""Golden parity: units.golden.json (generated from units.py) vs units.py (#14458)."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest

from src.shared.python.ui import units
from src.shared.python.ui.units import UnitSystem

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[3] / "ui/src/utils/units.golden.json").read_text(
        encoding="utf-8"
    )
)
CONVERSIONS = {
    "to_display_distance_ft": (partial(units.to_display_distance, unit="ft"), "si"),
    "from_display_distance_ft": (
        partial(units.from_display_distance, unit="ft"),
        "display",
    ),
    "to_display_speed": (units.to_display_speed, "si"),
    "from_display_speed": (units.from_display_speed, "display"),
}


@pytest.mark.parametrize("key", CONVERSIONS)
def test_conversions_match_golden(key: str) -> None:
    convert, source = CONVERSIONS[key]
    target = "display" if source == "si" else "si"
    assert GOLDEN[key], f"golden {key} is empty"
    for case in GOLDEN[key]:
        got = convert(case[source], UnitSystem(case["system"]))
        assert got == pytest.approx(case[target], rel=1e-12), case


@pytest.mark.parametrize("system", list(UnitSystem))
def test_suffixes_match_golden(system: UnitSystem) -> None:
    ft = units.distance_suffix(system, unit="ft")
    assert ft == GOLDEN["distance_suffix_ft"][system.value]
    assert units.speed_suffix(system) == GOLDEN["speed_suffix"][system.value]
