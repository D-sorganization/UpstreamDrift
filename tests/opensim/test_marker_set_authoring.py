"""Deterministic MarkerSet authoring on the packaged golf_humanoid model."""

from defusedxml import ElementTree as ET
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import marker_set

pytestmark = pytest.mark.unit
OSIM = (
    Path(__file__).resolve().parents[2]
    / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
)


def _placements() -> dict[str, marker_set.MarkerPlacement]:
    return {
        "WaistLeft": marker_set.MarkerPlacement("pelvis", (0.05, 0.02, 0.12)),
        "Marker_3:3:1": marker_set.MarkerPlacement("Club", (0.0, -0.9, 0.0)),
    }


def test_attach_marker_set_writes_opensim_markers(tmp_path: Path) -> None:
    tree = ET.parse(OSIM)
    marker_set.attach_marker_set(tree, _placements())
    model = tree.getroot().find("Model")
    assert model is not None
    markers = model.findall("MarkerSet/objects/Marker")
    assert [m.get("name") for m in markers] == ["WaistLeft", "Marker_3:3:1"]
    first = markers[0]
    assert first.findtext("socket_parent_frame") == "/bodyset/pelvis"
    assert first.findtext("location") == "0.05 0.02 0.12"
    assert first.findtext("fixed") == "false"
    out = tmp_path / "with_markers.osim"
    marker_set.write_model(tree, out)
    again = ET.parse(out).getroot().find("Model")
    assert again is not None and len(again.findall("MarkerSet/objects/Marker")) == 2
    # Idempotent: attaching again replaces rather than duplicates.
    marker_set.attach_marker_set(tree, _placements())
    assert len(model.findall("MarkerSet/objects/Marker")) == 2


def test_attach_rejects_unknown_body_and_bad_offsets() -> None:
    tree = ET.parse(OSIM)
    with pytest.raises(ValueError):
        marker_set.attach_marker_set(
            tree, {"X": marker_set.MarkerPlacement("no_such_body", (0, 0, 0))}
        )
    with pytest.raises(ValueError):
        marker_set.MarkerPlacement("pelvis", (0.0, float("nan"), 0.0))
    with pytest.raises(ValueError):
        marker_set.attach_marker_set(tree, {})


def test_unlock_coordinates_flips_locked_flags_only_for_named() -> None:
    tree = ET.parse(OSIM)
    for coord in tree.findall(".//Coordinate"):
        if coord.get("name") in ("arm_flex_r", "lumbar_rotation", "elbow_flex_r"):
            locked_elem = coord.find("locked")
            if locked_elem is not None:
                locked_elem.text = "true"
    locked_before = marker_set.locked_coordinates(tree)
    assert "arm_flex_r" in locked_before and "lumbar_rotation" in locked_before
    changed = marker_set.unlock_coordinates(tree, ("arm_flex_r", "lumbar_rotation"))
    assert changed == ("arm_flex_r", "lumbar_rotation")
    after = marker_set.locked_coordinates(tree)
    assert "arm_flex_r" not in after and "lumbar_rotation" not in after
    assert "elbow_flex_r" in after
    with pytest.raises(ValueError):
        marker_set.unlock_coordinates(tree, ("no_such_coordinate",))
    assert marker_set.unlock_coordinates(tree, ("arm_flex_r",)) == ()
