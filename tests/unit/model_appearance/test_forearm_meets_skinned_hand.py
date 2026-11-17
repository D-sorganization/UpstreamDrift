"""The forearm's wrist end sized from the skinned hand's own wrist (#15262).

The forearm ended 29 mm in radius whatever hand was drawn; the skinned
hand's wrist is about 25 mm by 18 mm, so the forearm overlapped it as a
thicker stub. With the skinned hands drawn, the forearm's wrist radius is
read from the hand mesh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
)
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(600)]
SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
)


def _wrist_radii(spec: dict) -> list[float]:
    """Forearm skin radius at the wrist joint, per arm."""
    club = ca.assembly_from_spec(spec)
    doc = document_from_dict({"schema_version": "appearance-v1"})
    meshes = {m.body: m for m in body_meshes(spec, doc, club) if m.label == "skin0"}
    out = []
    for cap in derive_visual_skeleton(spec).capsules:
        carries_wrist = any(
            j["parent"] == cap.body and library.classify_body(j["child"]) != "forearm"
            for j in spec["joints"]
        )
        if library.classify_body(cap.body) != "forearm" or not carries_wrist:
            continue
        line = np.subtract(cap.end_m, cap.start_m)
        axis = line / np.linalg.norm(line)
        v = meshes[cap.body].mesh_in_body_frame.vertices - cap.start_m
        ring = v[np.abs(v @ axis - cap.length_m()) < 1e-9]
        out.append(
            float(np.linalg.norm(ring - np.outer(ring @ axis, axis), axis=1).max())
        )
    assert len(out) == 2
    return out


def test_the_hand_wrist_radius_lies_between_its_thickness_and_width() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    club = ca.assembly_from_spec(spec)
    assert club is not None
    radius = sch.wrist_radius_m(club)
    assert 0.017 <= radius <= 0.026
    assert radius < library.PART_RADIUS_M["forearm"] * library.distal_scale("forearm")


def test_the_forearm_ends_at_the_skinned_hand_wrist_radius() -> None:
    base = json.loads(SPEC.read_text(encoding="utf-8"))
    skinned = apply_skinned_hand_wrists(base)
    club = ca.assembly_from_spec(skinned)
    assert club is not None
    for radius in _wrist_radii(skinned):
        assert radius == pytest.approx(sch.wrist_radius_m(club), abs=1e-6)
    # Other hands keep the library's wrist.
    plain = library.PART_RADIUS_M["forearm"] * library.distal_scale("forearm")
    for radius in _wrist_radii(base):
        assert radius == pytest.approx(plain, abs=1e-6)
