"""``ui/src/utils/putterHead.golden.json`` matches the shared core (#14462).

``putterHead.test.ts`` holds the React side to the same file.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.api.routes.putting_green import Putt3DSimulationRequest
from src.shared.python.putting_dynamics.putter_presets import BALANCE_PRESETS
from src.shared.python.putting_dynamics.types import (
    TYPICAL_HEAD_MASS_KG,
    TYPICAL_HEAD_MOI_KG_M2,
)

pytestmark = pytest.mark.unit

_GOLDEN = Path(__file__).resolve().parents[3] / "ui/src/utils/putterHead.golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def _request_bounds(field: str) -> list[float]:
    """The ``ge``/``le`` limits of one ``Putt3DSimulationRequest`` field."""
    metadata = Putt3DSimulationRequest.model_fields[field].metadata
    lo = next(m.ge for m in metadata if hasattr(m, "ge"))
    hi = next(m.le for m in metadata if hasattr(m, "le"))
    return [lo, hi]


def test_typical_head_matches_core(golden: dict) -> None:
    assert golden["typical"] == {
        "head_mass_kg": TYPICAL_HEAD_MASS_KG,
        "head_moi_kg_m2": TYPICAL_HEAD_MOI_KG_M2,
    }


@pytest.mark.parametrize("field", ["head_mass_kg", "head_moi_kg_m2"])
def test_input_bounds_match_simulate_request(golden: dict, field: str) -> None:
    assert golden["bounds"][field] == _request_bounds(field)


def test_presets_match_core(golden: dict) -> None:
    assert golden["presets"] == [
        dataclasses.asdict(p) for p in BALANCE_PRESETS.values()
    ]
