"""Tests for the anthropometric candidate transform of the full-body document."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching import anthropometric_candidate as module
from src.shared.python.motion_matching.anthropometry import segment_parameters
from src.shared.python.motion_matching.full_body_spec import (
    canonical_sha256,
    validate_full_body_spec,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)


def test_zero_pose_positions_match_mujoco() -> None:
    spec = json.loads(SPEC.read_text())
    anchors = module.zero_pose_joint_positions(spec)
    adapter = NativeMujocoFullBodyModel(SPEC.read_bytes())
    adapter.frame_poses(dict.fromkeys(adapter.coordinate_order, 0.0))
    for joint in spec["joints"]:
        first = joint["primitives"][0]["coordinate"]
        world = adapter.data.xanchor[adapter.model.joint(first).id]
        np.testing.assert_allclose(anchors[joint["name"]], world, atol=1e-9)


def test_candidate_lengths_masses_and_inertias() -> None:
    spec = json.loads(SPEC.read_text())
    stature, mass = 1.71, 78.0
    cand = module.anthropometric_candidate(spec, stature_m=stature, mass_kg=mass)
    assert cand["subject"]["mass_kg"] == mass
    assert canonical_sha256(cand) != canonical_sha256(spec)
    with pytest.raises(ValueError):
        validate_full_body_spec(
            cand, json.loads(UPPER.read_text())
        )  # unqualified by design
    anchors = module.zero_pose_joint_positions(cand)

    def length(a: str, b: str) -> float:
        return float(np.linalg.norm(anchors[a] - anchors[b]))

    names = {j["name"].rsplit("/", 1)[-1]: j["name"] for j in cand["joints"]}
    joints = {j["name"] for j in cand["joints"]}
    shoulder = next(j for j in joints if "Left Shoulder Joint" in j)
    elbow = next(j for j in joints if "Left Elbow Joint" in j)
    forearm = next(j for j in joints if "Left Forearm" in j)
    wrist = next(j for j in joints if "Left Wrist" in j)
    scap = next(j for j in joints if "Left Scapula" in j)
    assert length(shoulder, elbow) == pytest.approx(
        segment_parameters(stature, mass, "upper_arm").length_m, abs=1e-6
    )
    assert length(elbow, forearm) + length(forearm, wrist) == pytest.approx(
        segment_parameters(stature, mass, "forearm").length_m, abs=2e-3
    )
    assert length(scap, shoulder) == pytest.approx(
        module.BIACROMIAL_FRACTION_OF_STATURE * stature / 2, abs=1e-6
    )
    total = sum(s["mass_kg"] for b in cand["bodies"] for s in b["solids"])
    assert total == pytest.approx(mass, rel=0.10)
    for body in cand["bodies"]:
        for solid in body["solids"]:
            inertia = np.array(solid["inertia_com_kg_m2"])
            if solid["mass_kg"] > 0:
                assert np.all(np.linalg.eigvalsh(inertia) > 0), body["name"]
    thigh = next(b for b in cand["bodies"] if b["name"] == "femur_r")["solids"][0]
    expected = segment_parameters(stature, mass, "thigh")
    assert thigh["mass_kg"] == pytest.approx(expected.mass_kg)
    femur = length("hip_r", "knee_r")  # inertia uses the model's own segment length
    longitudinal = module.DE_LEVA_MALE["thigh"].radii[2]
    assert sorted(np.linalg.eigvalsh(np.array(thigh["inertia_com_kg_m2"])))[
        0
    ] == pytest.approx(expected.mass_kg * (longitudinal * femur) ** 2, rel=1e-6)
    # The candidate still builds in MuJoCo and keeps 41 coordinates.
    adapter = NativeMujocoFullBodyModel(json.dumps(cand).encode())
    assert len(adapter.coordinate_order) == 41
    assert float(np.sum(adapter.model.body_mass)) == pytest.approx(total)
    assert names  # joint names resolved
    with pytest.raises(ValueError):
        module.anthropometric_candidate(spec, stature_m=0.0, mass_kg=mass)
