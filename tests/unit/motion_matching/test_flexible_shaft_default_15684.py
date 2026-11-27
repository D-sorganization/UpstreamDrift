"""The flexible shaft is the default of the current golfer model (#15684).

Golf shafts bend, so the canonical golfer model carries a chain shaft (Stiff
by default). A rigid club is only an explicit ballpark fallback, logged, and
the canonical gate refuses it.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from src.shared.python.model_appearance import golfer_model_versions as gmv
from src.shared.python.model_appearance.canonical_golfer_gate import (
    NonCanonicalGolferError,
    require_canonical_golfer,
)
from src.shared.python.motion_matching.club_shaft_chain import (
    passive_coordinates,
    shaft_model,
)
from src.shared.python.motion_matching.execution import golfer_reference_specs as refs
from src.shared.python.motion_matching.execution.spec_revisions import (
    current_shaft,
)

pytestmark = pytest.mark.unit


def test_the_current_model_names_a_stiff_chain_shaft() -> None:
    current = gmv.current_golfer_model()
    assert current.shaft_model == "chain"
    assert current.shaft_flex == "Stiff"
    assert current.version == "full_body_skinned_flex_v2"


def _registry_with(tmp_path: Path, **shaft: str) -> Path:
    raw = json.loads(gmv.SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["current"]["shaft"].update(shaft)
    path = tmp_path / "golfer_model_versions.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_the_registry_refuses_an_unknown_or_rigid_current_shaft(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="shaft"):
        gmv.load_golfer_model_versions(_registry_with(tmp_path, model="noodle"))
    with pytest.raises(ValueError, match="rigid"):
        gmv.load_golfer_model_versions(_registry_with(tmp_path, model="rigid"))
    with pytest.raises(ValueError, match="flex"):
        gmv.load_golfer_model_versions(_registry_with(tmp_path, flex=""))


def test_current_shaft_is_the_registry_chain_unless_ballpark_rigid(caplog) -> None:
    assert current_shaft(None) == ("chain", {"flex": "Stiff"})
    with caplog.at_level(logging.WARNING):
        assert current_shaft("rigid") == ("rigid", {})
    assert "ballpark" in caplog.text
    with pytest.raises(ValueError, match="shaft_model"):
        current_shaft("noodle")


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_the_reference_specs_carry_the_stiff_chain_shaft(club: str) -> None:
    document = refs.reference_spec(club)
    assert shaft_model(document) == "chain"
    assert document["club"]["shaft"]["flex"] == "Stiff"
    assert passive_coordinates(document)
    assert require_canonical_golfer(document) == gmv.current_golfer_model()


def test_the_canonical_gate_refuses_a_rigid_club() -> None:
    document = copy.deepcopy(refs.reference_spec("driver"))
    document["club"].pop("shaft")
    with pytest.raises(NonCanonicalGolferError, match="flexible") as caught:
        require_canonical_golfer(document)
    assert caught.value.part == "shaft"


def test_an_overlay_sets_a_whole_value_it_cannot_recurse_into() -> None:
    parent = {"a": [1, 2], "b": {"x": 1, "y": 2}, "c": 0}
    child = {"a": [1, 2, 3], "b": {"x": 1}, "c": 0}
    ops = refs.overlay_between(parent, child)
    assert ops == [
        {"path": ["a"], "value": [1, 2, 3]},
        {"path": ["b"], "value": {"x": 1}},
    ]
    assert refs.apply_overlay(parent, ops) == child
    with pytest.raises(ValueError, match="root"):
        refs.overlay_between({"a": 1}, {})
