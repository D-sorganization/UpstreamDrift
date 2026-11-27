"""Current-model golfer reference specs (#15648).

The driver and 7-iron reference specs on the current golfer model are a
recorded overlay on the legacy reference specs, checked by hash, and must stay
what ``with_current_golfer_model`` makes of their parents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.model_appearance import golfer_model_versions as gmv
from src.shared.python.motion_matching.execution import golfer_reference_specs as refs
from src.shared.python.motion_matching.execution.spec_revisions import (
    with_current_golfer_model,
)

pytestmark = pytest.mark.unit


def test_overlay_round_trips_a_nested_document() -> None:
    parent = {"a": [1.0, {"b": 2}], "c": "x"}
    child = {"a": [1.5, {"b": 2, "d": [3]}], "c": "y", "e": None}
    ops = refs.overlay_between(parent, child)
    assert refs.apply_overlay(parent, ops) == child
    assert parent == {"a": [1.0, {"b": 2}], "c": "x"}


def test_overlay_refuses_what_a_set_cannot_express() -> None:
    """Only a key dropped at the root has no path to set (#15684)."""
    with pytest.raises(ValueError):
        refs.overlay_between({"a": 1, "b": 2}, {"a": 1})


def test_apply_overlay_refuses_an_empty_or_dangling_path() -> None:
    with pytest.raises(ValueError, match="empty"):
        refs.apply_overlay({}, [{"path": [], "value": 1}])
    with pytest.raises(ValueError, match="parent"):
        refs.apply_overlay({}, [{"path": ["a", "b"], "value": 1}])


def test_the_driver_and_iron7_reference_specs_are_on_the_current_model() -> None:
    assert refs.reference_clubs() == ("driver", "iron7")
    current = gmv.current_golfer_model()
    for club in refs.reference_clubs():
        document = refs.reference_spec(club)
        assert gmv.deprecated_parts(document) == ()
        assert document["subject"]["hand_visual_backend"] == current.hand_visual_backend
        assert document["subject"]["body_visual_backend"] == current.body_visual_backend
        revision = document["spec_revision"]["configuration"]
        assert revision == current.spec_revision_configuration


def test_a_changed_overlay_fails_its_recorded_hash(tmp_path: Path) -> None:
    settings = json.loads(refs.SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["specs"]["driver"]["overlay"][0]["value"] = "tampered"
    bad = tmp_path / "golfer_reference_specs.json"
    bad.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        refs.reference_spec("driver", path=bad)
    with pytest.raises(ValueError, match="no reference spec"):
        refs.reference_spec("putter", path=bad)


def test_write_reference_spec_writes_the_document(tmp_path: Path) -> None:
    out = refs.write_reference_spec("iron7", tmp_path / "spec.json")
    assert json.loads(out.read_text(encoding="utf-8")) == refs.reference_spec("iron7")


def test_the_registry_points_at_the_reference_specs() -> None:
    registry = json.loads(gmv.SETTINGS_PATH.read_text(encoding="utf-8"))
    pointer = registry["current"]["reference_specs"]
    assert (refs.REPO_ROOT / pointer).resolve() == refs.SETTINGS_PATH.resolve()
    recorded = json.loads(refs.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert recorded["golfer_model_version"] == gmv.current_golfer_model().version


def _numeric_drift(ops: list[dict[str, Any]]) -> float:
    """Largest numeric change in ``ops``; ``inf`` for any non-numeric change."""
    worst = 0.0
    for op in ops:
        if not isinstance(op["value"], float):
            return float("inf")
        worst = max(worst, abs(op["value"] - op["recorded"]))
    return worst


@pytest.mark.slow  # builds the skinned hand pair per club
@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_the_reference_spec_is_fresh(club: str) -> None:
    """Regenerating from the parent agrees to 1e-9 m (platform float drift)."""
    recorded = refs.reference_spec(club)
    entry = json.loads(refs.SETTINGS_PATH.read_text(encoding="utf-8"))["specs"][club]
    parent = json.loads((refs.REPO_ROOT / entry["parent"]).read_text("utf-8"))
    fresh, _ = with_current_golfer_model(parent, dict(parent["subject"]))
    ops = refs.overlay_between(recorded, fresh)
    for op in ops:
        node: Any = recorded
        for key in op["path"]:
            node = node[key]
        op["recorded"] = node
    assert _numeric_drift(ops) <= 1e-9, "regenerate the reference spec overlay"
