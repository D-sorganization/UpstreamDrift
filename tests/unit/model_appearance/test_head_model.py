"""Parametric visual head: schema round-trip, geometry, anchor, gaze channel."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    document_to_dict,
    head_frame_in_body,
    head_override_rotation,
    place_parts,
    resolve_head_anchor,
)
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit
SPECS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
LENGTH = 0.2429


def test_schema_round_trip_with_head_and_body_model() -> None:
    raw = {
        "schema_version": "appearance-v1",
        "body_model": "ellipsoid",
        "head": {
            "headwear": "cap",
            "headwear_material": "cap_white",
            "scale": 1.1,
            "forward_axis": "+y",
            "up_axis": "+z",
            "orientation_override": {
                "frame": "world",
                "yaw_rad": 0.3,
                "channel": "gaze",
            },
        },
    }
    doc = document_from_dict(raw)
    again = document_from_dict(document_to_dict(doc))
    assert again == doc
    assert doc.head.orientation_override.yaw_rad == 0.3


@pytest.mark.parametrize(
    "bad",
    [
        {"head": {"headwear": "wig"}},
        {"head": {"forward_axis": "+x", "up_axis": "-x"}},
        {"head": {"headwear": "cap", "headwear_material": "nope"}},
        {"body_model": "voxels"},
        {"head": {"orientation_override": {"pitch_rad": 5.0}}},
    ],
)
def test_invalid_head_fields_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        document_from_dict({"schema_version": "appearance-v1", **bad})


def test_default_document_round_trips_unchanged() -> None:
    doc = document_from_dict({"schema_version": "appearance-v1"})
    assert doc.head.enabled and doc.body_model == "ellipsoid"
    assert document_from_dict(document_to_dict(doc)) == doc


@pytest.mark.parametrize("headwear", ["none", "hair", "cap"])
def test_head_meshes_are_closed_and_have_a_face(headwear: str) -> None:
    parts = {
        p.name: p for p in build_head_parts(LENGTH, HeadSettings(headwear=headwear))
    }
    for part in parts.values():
        assert part.mesh.volume() > 0, part.name
    assert {
        "neck",
        "skull",
        "nose",
        "mouth",
        "eye_l",
        "eye_r",
        "ear_l",
        "ear_r",
    } <= set(parts)
    assert ("hair" in parts) == (headwear == "hair") and ("cap" in parts) == (
        headwear == "cap"
    )
    skull = parts["skull"].mesh.vertices
    assert skull[:, 2].max() == pytest.approx(LENGTH, abs=1e-3)  # vertex height
    # face looks along +x: nose and eyes in front of the skull centre, ears behind
    assert parts["nose"].mesh.vertices[:, 0].max() > skull[:, 0].max() * 0.95
    assert parts["eye_l"].mesh.vertices[:, 0].mean() > 0.05
    assert parts["ear_l"].mesh.vertices[:, 0].mean() < 0.0
    # left/right mirror about y = 0
    left, right = parts["eye_l"].mesh.vertices, parts["eye_r"].mesh.vertices
    assert left[:, 1].mean() == pytest.approx(-right[:, 1].mean(), abs=1e-9)


def test_neck_base_is_at_the_neck_point_within_5_mm() -> None:
    neck = {p.name: p for p in build_head_parts(LENGTH)}["neck"].mesh.vertices
    assert abs(neck[:, 2].min()) < 0.005
    assert np.hypot(*neck[neck[:, 2] < 0.0].mean(axis=0)[:2]) < 0.005


def test_orientation_follows_forward_axis() -> None:
    anchor = resolve_head_anchor(
        json.loads((SPECS / "full_body_spec_anthro_driver.json").read_text())
    )
    assert anchor is not None
    cfg = HeadSettings(forward_axis="+y", up_axis="+z")
    parts = place_parts(
        build_head_parts(anchor.length_m, cfg), head_frame_in_body(anchor, cfg)
    )
    eye = {p.name: p for p in parts}["eye_l"].mesh.vertices.mean(axis=0)
    assert eye[1] > 0.05 and abs(eye[0]) < 0.05


def test_anchor_uses_the_spec_head_body_and_none_without_one() -> None:
    driver = json.loads((SPECS / "full_body_spec_anthro_driver.json").read_text())
    anchor = resolve_head_anchor(driver)
    assert anchor.body.endswith("/Head") and anchor.source == "head_body"
    assert anchor.length_m == pytest.approx(
        driver["anthropometry"]["segments"]["head"]["length_m"]
    )
    assert (
        resolve_head_anchor(json.loads((SPECS / "full_body_spec_v1.json").read_text()))
        is None
    )


def test_gaze_rotation_conventions() -> None:
    assert np.allclose(head_override_rotation(0, 0, 0), np.eye(3))
    forward = np.array([1.0, 0.0, 0.0])
    assert head_override_rotation(np.pi / 2, 0, 0) @ forward == pytest.approx(
        [0, 1, 0], abs=1e-12
    )
    assert (head_override_rotation(0, 0.3, 0) @ forward)[
        2
    ] > 0  # positive pitch looks up
    up = np.array([0.0, 0.0, 1.0])
    assert (head_override_rotation(0, 0, 0.3) @ up)[1] < 0  # roll toward right shoulder
    with pytest.raises(ValueError):
        head_override_rotation(float("nan"), 0, 0)
