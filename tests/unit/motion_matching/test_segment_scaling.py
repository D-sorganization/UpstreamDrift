"""Tests for segment length scaling of a full-body specification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import segment_scaling as module
from src.shared.python.motion_matching.full_body_spec import (
    canonical_sha256,
    validate_full_body_spec,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)


def _joint(document: dict, name: str) -> dict:
    return next(j for j in document["joints"] if j["name"] == name)


def test_scaling_moves_distal_joints_solids_and_attachments() -> None:
    spec = json.loads(SPEC.read_text())
    scaled = module.scale_segments(spec, {"femur_r": 1.1, "tibia_l": 0.9})
    validate_full_body_spec(scaled, json.loads(UPPER.read_text()))
    assert canonical_sha256(scaled) != canonical_sha256(spec)
    old, new = _joint(spec, "knee_r"), _joint(scaled, "knee_r")
    np.testing.assert_allclose(
        np.array(new["parent_to_base"])[:3, 3],
        1.1 * np.array(old["parent_to_base"])[:3, 3],
    )
    np.testing.assert_allclose(
        np.array(new["parent_to_base"])[:3, :3], np.array(old["parent_to_base"])[:3, :3]
    )
    assert new["child_to_follower"] == old["child_to_follower"]
    assert _joint(scaled, "hip_r") == _joint(spec, "hip_r")  # parent is the pelvis
    old_ankle, new_ankle = _joint(spec, "ankle_l"), _joint(scaled, "ankle_l")
    np.testing.assert_allclose(
        np.array(new_ankle["parent_to_base"])[:3, 3],
        0.9 * np.array(old_ankle["parent_to_base"])[:3, 3],
    )
    femur_old = next(b for b in spec["bodies"] if b["name"] == "femur_r")["solids"][0]
    femur_new = next(b for b in scaled["bodies"] if b["name"] == "femur_r")["solids"][0]
    np.testing.assert_allclose(femur_new["com_m"], 1.1 * np.array(femur_old["com_m"]))
    np.testing.assert_allclose(
        femur_new["inertia_com_kg_m2"], 1.21 * np.array(femur_old["inertia_com_kg_m2"])
    )
    assert femur_new["mass_kg"] == femur_old["mass_kg"]
    assert scaled["contact"] == spec["contact"]  # spheres sit on the calcanei
    assert (
        scaled["marker_attachments"]["WaistLeft"]
        == spec["marker_attachments"]["WaistLeft"]
    )
    assert "femur_r x1.1000" in scaled["provenance"]
    with pytest.raises(ValueError):
        module.scale_segments(spec, {"nope": 1.0})
    with pytest.raises(ValueError):
        module.scale_segments(spec, {"femur_r": 0.0})


def test_marker_offsets_and_spheres_on_scaled_bodies_follow() -> None:
    spec = json.loads(SPEC.read_text())
    spec["marker_attachments"]["RKneeOut"] = {
        "body": "femur_r",
        "offset_m": [0.0, -0.4, 0.06],
    }
    scaled = module.scale_segments(spec, {"femur_r": 2.0, "calcn_r": 1.5})
    np.testing.assert_allclose(
        scaled["marker_attachments"]["RKneeOut"]["offset_m"], [0.0, -0.8, 0.12]
    )
    heel = next(s for s in scaled["contact"]["spheres"] if s["name"] == "heel_r")
    heel_old = next(s for s in spec["contact"]["spheres"] if s["name"] == "heel_r")
    np.testing.assert_allclose(
        heel["position_m"], 1.5 * np.array(heel_old["position_m"])
    )
    assert heel["radius_m"] == heel_old["radius_m"]
