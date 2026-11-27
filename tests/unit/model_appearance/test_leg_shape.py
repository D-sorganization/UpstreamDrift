"""Leg shape against the CC0 body sections (#15299, epic #15150). Visual only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.shared.python.model_appearance import document_from_dict, geometry, library
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.skinned_body import spec_body_poses

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (
        ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_text(encoding="utf-8")
)
ADDRESS = json.loads(
    (ROOT / "tests/fixtures/model_appearance/address_pose.json").read_text("utf-8")
)
DOC = document_from_dict({"schema_version": "appearance-v1"})
# Girth over stature of the athletic male CC0 body (measured at 1.82 m).
TARGET = {
    ("femur_r", 0.35): 591.0 / 1820.0,
    ("femur_r", 0.95): 392.0 / 1820.0,
    ("tibia_r", 0.28): 399.0 / 1820.0,
    ("tibia_r", 0.84): 230.0 / 1820.0,
}
GIRTH_TOLERANCE = 0.08
CHILD = {"femur_r": "tibia_r", "tibia_r": "talus_r"}


def _world(label: str, q: dict[str, float] | None = None) -> dict[str, Any]:
    poses = spec_body_poses(SPEC, q)
    out = {}
    for mesh in body_meshes(SPEC, DOC):
        if mesh.label == label and mesh.body[:5] in ("femur", "tibia"):
            pose = poses[mesh.body]
            verts = mesh.mesh_in_body_frame.vertices @ pose[:3, :3].T + pose[:3, 3]
            out[mesh.body] = (verts, np.asarray(mesh.mesh_in_body_frame.faces))
    return out


def _section(body: str, frac: float, label: str = "skin0") -> np.ndarray:
    """Section of ``body`` at ``frac`` of its length: columns (forward, lateral), m."""
    poses = spec_body_poses(SPEC)
    start, end = poses[body][:3, 3], poses[CHILD[body]][:3, 3]
    axis = (end - start) / np.linalg.norm(end - start)
    verts, faces = _world(label)[body]
    signed = (verts - start - frac * (end - start)) @ axis
    points = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = signed[faces[:, i]], signed[faces[:, j]]
        hit = a * b < 0
        t = (a[hit] / (a[hit] - b[hit]))[:, None]
        points.append(verts[faces[hit, i]] * (1 - t) + verts[faces[hit, j]] * t)
    rel = np.vstack(points) - start
    front = np.array([1.0, 0.0, 0.0]) - axis[0] * axis
    front /= np.linalg.norm(front)
    return np.column_stack([rel @ front, rel @ np.cross(axis, front)])


def _girth(body: str, frac: float) -> float:
    return float(ConvexHull(_section(body, frac)).area)


@pytest.mark.parametrize(("body", "frac"), sorted(TARGET))
def test_leg_girth_matches_the_measured_body(body: str, frac: float) -> None:
    target = TARGET[(body, frac)] * SPEC["subject"]["stature_m"]
    assert _girth(body, frac) == pytest.approx(target, rel=GIRTH_TOLERANCE)


def test_legs_taper_from_the_belly_to_the_far_joint() -> None:
    for body, fracs in (("femur_r", (0.4, 0.6, 0.8, 0.95)), ("tibia_r", (0.4, 0.84))):
        girths = [_girth(body, f) for f in fracs]
        assert girths == sorted(girths, reverse=True), (body, girths)
    assert _girth("femur_r", 0.3) > _girth("femur_r", 0.6)  # thickest near the hip


def test_calf_is_behind_the_shin_and_quadriceps_ahead_of_the_thigh() -> None:
    calf = _section("tibia_r", 0.30)[:, 0]
    assert -calf.min() > 2.0 * calf.max()
    thigh = _section("femur_r", 0.35)[:, 0]
    assert thigh.max() > 1.3 * -thigh.min()


def test_knee_has_no_pinch() -> None:
    assert _girth("tibia_r", 0.05) > 0.85 * _girth("femur_r", 0.95)


@pytest.mark.parametrize("pose", [None, ADDRESS], ids=["standing", "address"])
def test_thighs_do_not_interpenetrate(pose: dict[str, float] | None) -> None:
    legs = _world("garment0", pose)
    right, left = legs["femur_r"][0][:, 1], legs["femur_l"][0][:, 1]
    gap = max(left.min() - right.max(), right.min() - left.max())
    assert gap >= 0.0, gap


def test_trousers_cover_the_leg_skin() -> None:
    skin, garment = _world("skin0"), _world("garment0")
    for body in skin:
        assert (garment[body][0].max(0) >= skin[body][0].max(0)).all(), body
        assert (garment[body][0].min(0) <= skin[body][0].min(0)).all(), body


def test_leg_radius_scales_with_stature() -> None:
    at = library.LIMB_REFERENCE_STATURE_M
    assert library.stature_scale("thigh", at) == pytest.approx(1.0)
    assert library.stature_scale("shin", 0.9 * at) == pytest.approx(0.9)
    assert library.stature_scale("forearm", 0.9 * at) == 1.0  # not a scaled part
    assert library.stature_scale("thigh", None) == 1.0  # no subject stature
    for bad in (0.0, -1.7, float("nan"), "tall"):
        with pytest.raises((ValueError, TypeError)):
            library.stature_scale("thigh", bad)  # type: ignore[arg-type]


def test_loft_offset_moves_the_centreline_only_near_its_position() -> None:
    start, end, side = np.zeros(3), np.array([0.0, 0.0, 0.4]), np.array([0.02, 0, 0])
    plain = geometry.lofted_segment(start, end, 0.05)
    bowed = geometry.lofted_segment(
        start, end, 0.05, geometry.LoftOptions(offset=(side, 0.3, 0.3))
    )
    moved = bowed.vertices - plain.vertices
    near = np.abs(plain.vertices[:, 2] - 0.12) < 0.02
    far = plain.vertices[:, 2] > 0.26
    assert moved[near, 0].max() == pytest.approx(0.02, abs=2e-3)
    assert np.abs(moved[far]).max() < 1e-12
    with pytest.raises(ValueError, match="offset"):
        geometry.lofted_segment(
            start, end, 0.05, geometry.LoftOptions(offset=(side, 1.5, 0.3))
        )
