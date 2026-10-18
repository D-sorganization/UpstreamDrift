"""NSX-100 (#12729): the catalogue of allowed sensitivity options.

Both UIs build the sensitivity request from this catalogue only; nothing is
invented on the UI side.
"""

from __future__ import annotations

import json

import pytest

from src.shared.python.loop_nullspace import sensitivity_options as opts
from src.shared.python.loop_nullspace.derivative_capability import default_registry
from src.shared.python.loop_nullspace.strategy_spec import CATALOGUE
from src.shared.python.loop_nullspace.variance_records import Mechanism
from src.shared.python.perturbation.disturbance import InjectionStage, Spectrum

pytestmark = pytest.mark.unit


def test_catalogue_is_json_and_lists_the_backend_enums() -> None:
    cat = opts.option_catalogue()
    json.dumps(cat)
    assert [s["id"] for s in cat["stages"]] == [s.value for s in InjectionStage]
    assert [s["id"] for s in cat["spectra"]] == [s.value for s in Spectrum]
    assert [m["id"] for m in cat["mechanisms"]] == [m.value for m in Mechanism]
    assert [s["id"] for s in cat["strategies"]] == [s.value for s in CATALOGUE]


def test_strategies_carry_their_availability_and_reason() -> None:
    cat = opts.option_catalogue()
    for entry in cat["strategies"]:
        source = CATALOGUE[next(k for k in CATALOGUE if k.value == entry["id"])]
        assert entry["name"] == source.name
        assert entry["availability"] == source.availability.value
        if source.availability.value != "implemented":
            assert entry["reason"]


def test_default_outcome_priority_is_face_angle_first() -> None:
    cat = opts.option_catalogue()
    assert cat["default_outcome_priority"] == [opts.FACE_OUTCOME]
    assert opts.FACE_OUTCOME in [o["id"] for o in cat["outcomes"]]


def test_engines_come_from_the_capability_registry_with_reasons() -> None:
    cat = opts.option_catalogue()
    reg = default_registry()
    by_id = {e["id"]: e for e in cat["engines"]}
    assert "pinocchio" in by_id and "simscape" in by_id
    sim = by_id["simscape"]
    assert sim["available"] is False
    assert sim["reason"] == reg.declaration("simscape").reason
    assert by_id["pinocchio"]["available"] is True
    assert by_id["pinocchio"]["reason"] == ""
