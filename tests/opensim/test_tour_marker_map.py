"""Tour capture labels map onto golf_humanoid bodies for OpenSim tracking."""

from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import marker_map
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    tracked_labels,
)

pytestmark = pytest.mark.unit
OSIM = (
    Path(__file__).resolve().parents[2]
    / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
)


def test_every_tracked_label_maps_to_a_model_body() -> None:
    mapping = marker_map.GOLF_HUMANOID_MARKER_BODIES
    assert set(mapping) == set(tracked_labels())
    assert marker_map.body_for("LKneeOut") == "femur_l"
    assert marker_map.body_for("RAnkleOut") == "tibia_r"
    assert marker_map.body_for("LToeIn") == "calcn_l"
    assert marker_map.body_for("Marker_3:3:2") == "Club"
    assert marker_map.body_for("HeadTop") == "torso"


def test_unassigned_labels_are_rejected() -> None:
    for label in MARKER_SEGMENTS["unassigned"]:
        with pytest.raises(ValueError):
            marker_map.body_for(label)
    with pytest.raises(ValueError):
        marker_map.body_for("NotALabel")


def test_mapped_bodies_exist_in_the_packaged_model() -> None:
    bodies = marker_map.model_body_names(OSIM)
    assert {"pelvis", "torso", "Club", "calcn_l", "femur_r"} <= bodies
    missing = marker_map.bodies_missing_from(bodies)
    assert missing == ()


def test_labels_per_body_groups_the_full_body() -> None:
    per_body = marker_map.labels_per_body()
    assert set(per_body["Club"]) == {
        "Marker_2:2:1",
        "Marker_2:2:2",
        "Marker_2:2:3",
        "Marker_3:3:1",
        "Marker_3:3:2",
        "Marker_3:3:3",
    }
    assert per_body["pelvis"] == ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
    assert sum(len(v) for v in per_body.values()) == len(tracked_labels())
