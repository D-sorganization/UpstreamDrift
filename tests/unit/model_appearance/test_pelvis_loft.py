"""The pelvis visual is lofted on the hip landmarks, not on a capsule (HRG-C1).

Issue #13164 (epic #13160). The pelvis used to be lofted along the body's
longest visual capsule; after a functional hip calibration that capsule is the
left hip, so the pelvis was drawn 51 mm off the midline, almost on the right
femoral head, and overlapping the chest. These tests pin the replacement:
a solid centred on the hip joint centre (HJC) midpoint, oriented by the hip
line, running from below the HJCs up to the lumbar-joint plane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull, cKDTree

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import BodyMesh, body_meshes
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.skinned_body import spec_body_poses

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
ANTHRO = MODELS / "full_body_spec_anthro_driver.json"
DOC = document_from_dict({"schema_version": "appearance-v1"})
# Best-fit golfer hip joint centres (pelvis frame, m): left is 0.149 m from the
# origin, so the old longest-capsule rule picked it.
BEST_FIT_HIPS = {"hip_r": (-0.009, -0.102, -0.059), "hip_l": (-0.015, 0.114, -0.098)}
# Symmetric, but wider than the lumbar capsule (0.138 m): the old rule picked
# the first hip capsule and drew the pelvis lopsided.
SYMMETRIC_HIPS = {"hip_r": (0.0, -0.12, -0.08), "hip_l": (0.0, 0.12, -0.08)}


def _with_hips(hips: dict[str, tuple[float, float, float]]) -> dict[str, Any]:
    spec = json.loads(ANTHRO.read_text(encoding="utf-8"))
    for joint in spec["joints"]:
        if joint["name"] in hips:
            t = np.asarray(joint["parent_to_base"], float)
            t[:3, 3] = hips[joint["name"]]
            joint["parent_to_base"] = t.tolist()
    return spec


def _pelvis_body(spec: dict[str, Any]) -> str:
    return next(b for b in body_sections(spec) if b.endswith("LowerTorso"))


def _skin(spec: dict[str, Any], body: str) -> BodyMesh:
    (skin,) = [
        m for m in body_meshes(spec, DOC) if m.body == body and m.label == "skin0"
    ]
    return skin


def _hjcs(spec: dict[str, Any], body: str) -> np.ndarray:
    return np.array(
        [
            np.asarray(j["parent_to_base"], float)[:3, 3]
            for j in spec["joints"]
            if j["parent"] == body and library.classify_body(j["child"]) == "thigh"
        ]
    )


def _up(spec: dict[str, Any], body: str) -> np.ndarray:
    section = body_sections(spec)[body]
    return np.cross(section.depth_axis, section.ml_axis)


def test_symmetric_hips_give_a_left_right_symmetric_pelvis() -> None:
    spec = _with_hips(SYMMETRIC_HIPS)
    body = _pelvis_body(spec)
    verts = _skin(spec, body).mesh_in_body_frame.vertices
    ml = body_sections(spec)[body].ml_axis
    mid = _hjcs(spec, body).mean(axis=0)
    # Mirror every vertex through the plane normal to ML at the HJC midpoint:
    # the mirrored cloud must coincide with the cloud (nearest vertex < 1 mm).
    mirrored = verts - 2.0 * np.outer((verts - mid) @ ml, ml)
    gaps, _ = cKDTree(verts).query(mirrored)
    assert gaps.max() < 1e-3, f"left/right asymmetry {1e3 * gaps.max():.2f} mm"


def test_pelvis_is_centred_laterally_on_the_hjc_midpoint_when_asymmetric() -> None:
    spec = _with_hips(BEST_FIT_HIPS)
    body = _pelvis_body(spec)
    verts = _skin(spec, body).mesh_in_body_frame.vertices
    ml = body_sections(spec)[body].ml_axis
    centre = float(np.ptp(verts @ ml)) / 2.0 + float((verts @ ml).min())
    mid = float(_hjcs(spec, body).mean(axis=0) @ ml)
    assert abs(centre - mid) < 5e-3, f"lateral offset {1e3 * (centre - mid):.1f} mm"


@pytest.mark.parametrize("hips", [BEST_FIT_HIPS, SYMMETRIC_HIPS], ids=["fit", "sym"])
def test_both_hjcs_are_inside_the_skin(
    hips: dict[str, tuple[float, float, float]], capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _with_hips(hips)
    body = _pelvis_body(spec)
    verts = _skin(spec, body).mesh_in_body_frame.vertices
    planes = ConvexHull(verts).equations  # the solid is convex: exact clearance
    clearance = [
        float(-(planes[:, :3] @ p + planes[:, 3]).max()) for p in _hjcs(spec, body)
    ]
    with capsys.disabled():
        print(f"\nHJC clearance to skin (mm): {[round(1e3 * c, 1) for c in clearance]}")
    assert min(clearance) > 0.0, clearance


def test_pelvis_ends_at_the_lumbar_plane_and_does_not_overlap_the_chest() -> None:
    spec = _with_hips(BEST_FIT_HIPS)
    pelvis = _pelvis_body(spec)
    chest = next(b for b in body_sections(spec) if b.endswith("COMRod"))
    poses = spec_body_poses(spec)  # q = 0
    up = np.array([0.0, 0.0, 1.0])  # gravity is -z in this spec

    def axial(body: str) -> tuple[float, float]:
        v = _skin(spec, body).mesh_in_body_frame.vertices
        world = v @ poses[body][:3, :3].T + poses[body][:3, 3]
        return float((world @ up).min()), float((world @ up).max())

    p_lo, p_hi = axial(pelvis)
    c_lo, _ = axial(chest)
    lumbar = next(
        poses[j["child"]][:3, 3] @ up
        for j in spec["joints"]
        if j["parent"] == pelvis and library.classify_body(j["child"]) == "torso"
    )
    assert p_hi <= lumbar + 1e-6, (p_hi, lumbar)
    assert p_hi <= c_lo + 1e-6, f"pelvis/chest overlap {1e3 * (p_hi - c_lo):.1f} mm"
    hjc_lo = min(
        (poses[pelvis][:3, :3] @ h + poses[pelvis][:3, 3]) @ up
        for h in _hjcs(spec, pelvis)
    )
    assert p_lo < hjc_lo, "the pelvis must run below the HJC line"


def test_pelvis_width_covers_the_hjcs_plus_the_femoral_heads() -> None:
    # HRG-C4 (#13187) replaced "HJCs plus whole femur radii" (C1) with femoral
    # head containment; the width rules are pinned in test_pelvis_width.py.
    spec = _with_hips(BEST_FIT_HIPS)
    body = _pelvis_body(spec)
    verts = _skin(spec, body).mesh_in_body_frame.vertices
    ml = body_sections(spec)[body].ml_axis
    hjc = _hjcs(spec, body) @ ml
    need = float(np.ptp(hjc)) + 2 * (0.025 + 0.010)
    assert float(np.ptp(verts @ ml)) >= need


def test_hip_blend_radius_is_at_most_1p05_times_the_femur_radius() -> None:
    spec = json.loads(ANTHRO.read_text(encoding="utf-8"))
    blends = [
        m
        for m in body_meshes(spec, DOC)
        if m.label.startswith("blend") and library.classify_body(m.body) == "thigh"
    ]
    assert len(blends) == 2
    femur_r = library.PART_RADIUS_M["thigh"]
    for blend in blends:
        verts = blend.mesh_in_body_frame.vertices
        radius = float(np.linalg.norm(verts - verts.mean(axis=0), axis=1).max())
        assert radius <= 1.05 * femur_r + 1e-9, (blend.body, radius, femur_r)
    assert library.blend_scale(True, "thigh") <= 1.05
    assert library.blend_scale(False, "thigh") <= 1.05


def test_upper_arm_blend_is_unchanged() -> None:
    assert library.blend_scale(True, "upper_arm") == pytest.approx(
        library.BLEND_RADIUS_SCALE * library.GARMENT_THICKNESS
    )


def test_split_ellipsoid_rejects_bad_preconditions() -> None:
    from src.shared.python.model_appearance import geometry

    base = {
        "centre": np.zeros(3),
        "up": np.array([0.0, 0.0, 1.0]),
        "width": np.array([0.0, 1.0, 0.0]),
        "half_width": 0.2,
        "half_depth": 0.1,
        "half_up": 0.3,
        "half_down": 0.08,
        "top": 0.2,
    }
    mesh = geometry.split_ellipsoid_mesh(geometry.SplitEllipsoid(**base))
    assert mesh.volume() > 0
    assert float(mesh.vertices[:, 2].max()) == pytest.approx(0.2)
    assert float(mesh.vertices[:, 2].min()) == pytest.approx(-0.08)
    for patch in ({"top": 0.4}, {"half_width": -1.0}, {"width": base["up"]}):
        with pytest.raises(ValueError):
            geometry.split_ellipsoid_mesh(geometry.SplitEllipsoid(**{**base, **patch}))
