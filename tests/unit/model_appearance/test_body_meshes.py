"""Engine-neutral body meshes: the shared appearance and anthropometry spec.

R8 (#12685, epic #12678). ``model_appearance.body_meshes`` resolves a spec and
an appearance document into skin, garment, blend and shoe meshes in the spec
body frames, so every engine emitter only translates them. These tests pin
the contract: coverage of every body, trunk sizing from the spec
``visual_hints`` semi-axes, the default charcoal trousers (pelvis and both
legs, hip to ankle, no bare-knee band), determinism and the DbC failures.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import (
    SPEC_BODY_FRAME,
    BodyMesh,
    MeshReference,
    body_meshes,
    resolve_appearance,
    section_size,
)
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.schema import AppearanceDocument, SegmentRule
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
ANTHRO = (
    MODELS / "full_body_spec_anthro_driver.json",
    MODELS / "full_body_spec_anthro_iron7.json",
)
V1 = MODELS / "full_body_spec_v1.json"
DEFAULT = {"schema_version": "appearance-v1"}
SHORTS = {"schema_version": "appearance-v1", "clothing": "golf_polo_shorts"}
TRUNK_PARTS = ("torso", "pelvis")
LEG_PARTS = ("thigh", "shin")


def _spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _doc(data: dict[str, Any] = DEFAULT) -> AppearanceDocument:
    return document_from_dict(data)


def _by_body(meshes: list[BodyMesh]) -> dict[str, list[BodyMesh]]:
    out: dict[str, list[BodyMesh]] = {}
    for mesh in meshes:
        out.setdefault(mesh.body, []).append(mesh)
    return out


def _semi_axis(half: np.ndarray, direction: np.ndarray) -> float:
    """Radius of an axis-aligned ellipsoid along a unit ``direction``."""
    d = direction / np.linalg.norm(direction)
    return float(1.0 / np.sqrt(np.sum((d / half) ** 2)))


def _hint_semi_axes(spec: dict[str, Any], body: str) -> tuple[float, float]:
    """Largest (ML, AP) semi-axis of the spec hint ellipsoids of ``body``."""
    section = body_sections(spec)[body]
    shapes = [s for s in derive_visual_skeleton(spec).shapes if s.body == body]
    assert shapes, f"{body} has no visual_hints shapes"
    half = [np.asarray(s.half_size_m) for s in shapes]
    return (
        max(_semi_axis(h, section.ml_axis) for h in half),
        max(_semi_axis(h, section.depth_axis) for h in half),
    )


@pytest.mark.parametrize("spec_path", ANTHRO, ids=lambda p: p.stem)
def test_every_non_club_non_head_body_yields_a_mesh(spec_path: Path) -> None:
    spec = _spec(spec_path)
    meshes = _by_body(body_meshes(spec, _doc()))
    drawn = {cap.body for cap in derive_visual_skeleton(spec).capsules}
    wanted = {b for b in drawn if library.classify_body(b) not in ("club", "head")}
    assert wanted, "spec has no drawable bodies"
    assert wanted <= set(meshes), sorted(wanted - set(meshes))


@pytest.mark.parametrize("spec_path", ANTHRO, ids=lambda p: p.stem)
def test_trunk_and_pelvis_extents_equal_the_spec_hint_semi_axes(
    spec_path: Path,
) -> None:
    spec = _spec(spec_path)
    sections = body_sections(spec)
    hinted = {s.body for s in derive_visual_skeleton(spec).shapes}
    seen = set()
    for body, meshes in _by_body(body_meshes(spec, _doc())).items():
        if library.classify_body(body) not in TRUNK_PARTS or body not in hinted:
            continue
        skins = [m for m in meshes if m.label.startswith("skin")]
        assert len(skins) == 1, f"{body}: one trunk solid, not {len(skins)} lofts"
        verts = skins[0].mesh_in_body_frame.vertices
        ml = float(np.ptp(verts @ sections[body].ml_axis))
        ap = float(np.ptp(verts @ sections[body].depth_axis))
        hint_ml, hint_ap = _hint_semi_axes(spec, body)
        assert ml == pytest.approx(2 * hint_ml, rel=0.05), (body, ml, hint_ml)
        assert ap == pytest.approx(2 * hint_ap, rel=0.05), (body, ap, hint_ap)
        assert ml > ap, (body, ml, ap)
        seen.add(library.classify_body(body))
    assert seen == set(TRUNK_PARTS)


@pytest.mark.parametrize("spec_path", ANTHRO, ids=lambda p: p.stem)
def test_chest_is_shallower_than_shoulder_width(spec_path: Path) -> None:
    """Owner complaint (epic #12678 item 1): chest-to-back < shoulder-to-shoulder."""
    spec = _spec(spec_path)
    chest = next(b for b in body_sections(spec) if b.endswith("COMRod"))
    size = section_size(spec, chest)
    assert size.source == "visual_hints"
    # Canonical thorax: 0.185 x 1.71 m / 2 = 0.158 m ML semi-axis (a half width).
    assert size.radius_m == pytest.approx(0.185 * 1.71 / 2, rel=1e-6)
    assert 0.5 < size.aspect < 0.8  # AP/ML: deep chest is a regression


def test_spec_without_hints_uses_the_documented_library_default() -> None:
    spec = _spec(V1)
    chest = next(b for b in body_sections(spec) if b.endswith("COMRod"))
    size = section_size(spec, chest)
    assert size.source == "library_default"
    assert size.radius_m == library.PART_RADIUS_M["torso"]
    assert size.aspect == library.PART_ASPECT["torso"]


@pytest.mark.parametrize("spec_path", (*ANTHRO, V1), ids=lambda p: p.stem)
def test_default_outfit_is_charcoal_trousers_on_pelvis_and_both_legs(
    spec_path: Path,
) -> None:
    spec = _spec(spec_path)
    doc = _doc()
    assert doc.clothing == library.DEFAULT_CLOTHING == "golf_polo_trousers"
    garments = {
        m.body: m.material_name
        for m in body_meshes(spec, doc)
        if m.label.startswith("garment")
        and library.classify_body(m.body) in ("pelvis", *LEG_PARTS)
    }
    parts = {library.classify_body(b) for b in garments}
    assert parts == {"pelvis", *LEG_PARTS}
    assert set(garments.values()) == {library.TROUSERS_MATERIAL}
    assert sum(library.classify_body(b) in LEG_PARTS for b in garments) == 4


@pytest.mark.parametrize("spec_path", (*ANTHRO, V1), ids=lambda p: p.stem)
def test_no_blue_segment_in_the_default_outfit(spec_path: Path) -> None:
    resolved = resolve_appearance(_spec(spec_path), _doc())
    for mesh in resolved.meshes:
        r, g, b, _ = resolved.materials[mesh.material_name].rgba
        assert not (b > r + 0.15 and b > g + 0.1), (mesh.body, mesh.material_name)


def _world(spec: dict[str, Any], mesh: BodyMesh) -> np.ndarray:
    pose = spec_body_poses(spec)[mesh.body]
    return mesh.mesh_in_body_frame.vertices @ pose[:3, :3].T + pose[:3, 3]


def _inside(hull_points: np.ndarray, points: np.ndarray) -> bool:
    hull = ConvexHull(hull_points)
    lhs = points @ hull.equations[:, :3].T + hull.equations[:, 3]
    return bool((lhs <= 1e-7).all())


@pytest.mark.parametrize("spec_path", (*ANTHRO, V1), ids=lambda p: p.stem)
def test_trousers_cover_hip_to_ankle_with_no_bare_knee_band(spec_path: Path) -> None:
    spec = _spec(spec_path)
    doc = _doc()
    by_body = _by_body(body_meshes(spec, doc))
    up = -np.asarray(spec["gravity_m_s2"], float)
    up /= np.linalg.norm(up)
    for side in ("l", "r"):
        chain = [
            next(b for b in by_body if library.classify_body(b) == "pelvis"),
            f"femur_{side}",
            f"tibia_{side}",
            f"calcn_{side}",
        ]
        spans = []
        for body in chain:
            garments = [m for m in by_body[body] if m.label.startswith("garment")]
            assert garments, f"{body} wears no garment"
            pts = np.vstack([_world(spec, m) for m in garments])
            spans.append((float((pts @ up).min()), float((pts @ up).max())))
            if library.classify_body(body) in LEG_PARTS:
                for skin in (
                    m for m in by_body[body] if m.material_name == doc.skin_tone
                ):
                    assert _inside(pts, _world(spec, skin)), f"bare skin on {body}"
        for (upper_lo, _), (_, lower_hi) in zip(spans, spans[1:], strict=False):
            assert upper_lo < lower_hi, ("exposed band between garments", chain, spans)


def test_shorts_still_show_the_knee() -> None:
    """The shorts preset keeps its partial thigh band (coverage is per outfit)."""
    assert library.garment_coverage("golf_polo_shorts", "thigh") == (0.0, 0.6)
    assert library.garment_coverage("golf_polo_trousers", "thigh") == (0.0, 1.0)
    assert library.garment_coverage("golf_polo_trousers", "shin") == (0.0, 1.0)


@pytest.mark.parametrize("data", (DEFAULT, SHORTS), ids=("trousers", "shorts"))
def test_pelvis_wears_the_pants_material(data: dict[str, Any]) -> None:
    spec = _spec(ANTHRO[0])
    doc = _doc(data)
    pelvis = [
        m
        for m in body_meshes(spec, doc)
        if library.classify_body(m.body) == "pelvis" and m.label.startswith("garment")
    ]
    assert pelvis
    assert {m.material_name for m in pelvis} == {library.pants_material(doc.clothing)}


def test_output_is_deterministic() -> None:
    spec, doc = _spec(ANTHRO[1]), _doc()
    first, second = body_meshes(spec, doc), body_meshes(spec, doc)
    assert [(m.body, m.label, m.material_name) for m in first] == [
        (m.body, m.label, m.material_name) for m in second
    ]
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(
            a.mesh_in_body_frame.vertices, b.mesh_in_body_frame.vertices
        )
        assert np.array_equal(a.mesh_in_body_frame.faces, b.mesh_in_body_frame.faces)


def test_unknown_material_raises() -> None:
    doc = dataclasses.replace(
        _doc(), segments=(SegmentRule(match="femur_r", material="no_such_material"),)
    )
    with pytest.raises(ValueError, match="no_such_material"):
        body_meshes(_spec(ANTHRO[0]), doc)


def test_unknown_body_part_is_other_and_never_skin() -> None:
    assert library.classify_body("mystery_widget") == "other"
    doc = _doc()
    material = library.material_name_for(doc, "mystery_widget")
    assert material == library.PLAIN_BODY_MATERIAL
    assert material not in library.SKIN_TONES
    assert library.garment_name_for(doc, "mystery_widget") is None


def test_resolved_spec_declares_its_frame_and_materials() -> None:
    resolved = resolve_appearance(_spec(ANTHRO[0]), _doc())
    assert resolved.frame == SPEC_BODY_FRAME == "spec_body"
    assert {m.material_name for m in resolved.meshes} <= set(resolved.materials)
    assert resolved.head_mesh is not None  # default CC0 head (#12949)
    body = resolved.meshes[0].body
    assert all(m.body == body for m in resolved.meshes_for(body))


def test_head_mesh_slot_requires_licence_and_provenance() -> None:
    ref = MeshReference(path="assets/head.obj", licence="CC0-1.0", provenance="x")
    resolved = resolve_appearance(_spec(ANTHRO[0]), _doc(), head_mesh=ref)
    assert resolved.head_mesh == ref
    for bad in (
        {"path": "", "licence": "CC0-1.0", "provenance": "x"},
        {"path": "a.obj", "licence": "", "provenance": "x"},
        {"path": "a.obj", "licence": "CC0-1.0", "provenance": " "},
    ):
        with pytest.raises(ValueError):
            MeshReference(**bad)
