"""Flight model registry labels must not imply unimplemented models (#15443 F6).

The registry names five constant Cd/Cl presets after researchers (Nathan,
Ballantyne, J. Cole, Rospie DL, Charry L3) and one after MacDonald and Hanzely.
None of them reproduces the published model, so each description must say what
it actually implements. Names stay unchanged for API and UI compatibility.
"""

from __future__ import annotations

import pytest

from src.shared.python.physics.flight_models import (
    FlightModelRegistry,
    FlightModelType,
)

pytestmark = pytest.mark.unit

PRESET_TYPES = (
    FlightModelType.NATHAN,
    FlightModelType.BALLANTYNE,
    FlightModelType.JCOLE,
    FlightModelType.ROSPIE_DL,
    FlightModelType.CHARRY_L3,
    FlightModelType.MACDONALD_HANZELY,
)
DISCLAIMER = "not the published"


@pytest.mark.parametrize("model_type", PRESET_TYPES)
def test_named_preset_description_denies_published_model(
    model_type: FlightModelType,
) -> None:
    model = FlightModelRegistry.get_model(model_type)
    assert DISCLAIMER in model.description.lower(), model.description


@pytest.mark.parametrize("model_type", PRESET_TYPES)
def test_named_preset_description_states_what_it_implements(
    model_type: FlightModelType,
) -> None:
    model = FlightModelRegistry.get_model(model_type)
    assert "constant" in model.description.lower(), model.description


def test_waterloo_penner_description_states_calibration() -> None:
    model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
    assert "calibrat" in model.description.lower(), model.description


def test_registry_names_are_unchanged() -> None:
    names = {m.name for m in FlightModelRegistry.get_all_models()}
    assert {"Nathan", "Ballantyne", "J. Cole", "Rospie DL", "Charry L3"} <= names
    assert {"Waterloo/Penner", "MacDonald-Hanzely"} <= names
