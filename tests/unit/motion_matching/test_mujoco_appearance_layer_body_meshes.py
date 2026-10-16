"""MuJoCo is a thin translator of the shared body meshes (R8, #12685).

- **Equivalence.** A golden per-body summary (material, vertex count, bbox in
  the MJCF body frame) captured from the pre-refactor appearance layer pins
  every non-trunk body exactly; trunk bodies keep their materials and become
  one trunk solid each (the documented sizing change).
- **Frames.** The v1 chest body (COMRod) has a non-identity
  ``child_to_follower`` rotation, so its MJCF body frame is not the spec body
  frame. The chest is checked in the WORLD frame after MuJoCo compiles the
  model: wider shoulder-to-shoulder than chest-to-back, vertical long axis.
- **Physics** stays bit-identical; **palette** has one source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.engines.physics_engines.mujoco.python import visual_layer
from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import section_size

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
SPECS = {
    p.stem: p
    for p in (
        MODELS / "full_body_spec_anthro_driver.json",
        MODELS / "full_body_spec_anthro_iron7.json",
        MODELS / "full_body_spec_v1.json",
    )
}
GOLDEN = ROOT / "tests/fixtures/model_appearance/mujoco_body_mesh_summary.json"
# The golden was captured with the shorts preset, whose geometry R8 does not
# change, so every difference left is the trunk sizing change.
SHORTS = {"schema_version": "appearance-v1", "clothing": "golf_polo_shorts"}
DEFAULT = {"schema_version": "appearance-v1"}
_MESH = re.compile(r"vmesh_\d+_(\w+)")
TRUNK_PARTS = ("torso", "pelvis")


def body_summary(raw: bytes, doc: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Body -> sorted [{label, material, n_vertices, bbox_min, bbox_max}]."""
    xml, _ = exporter.export_full_body_mjcf(raw, appearance=document_from_dict(doc))
    root = ET.fromstring(xml)
    verts = {
        m.get("name"): np.array(m.get("vertex", "").split(), float).reshape(-1, 3)
        for m in root.iter("mesh")
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            match = _MESH.fullmatch(geom.get("mesh") or "")
            if match is None:
                continue
            v = verts[geom.get("mesh")]
            out.setdefault(body.get("name") or "", []).append(
                {
                    "label": match.group(1),
                    "material": geom.get("material"),
                    "n_vertices": len(v),
                    "bbox_min": np.round(v.min(axis=0), 6).tolist(),
                    "bbox_max": np.round(v.max(axis=0), 6).tolist(),
                }
            )
    return {k: sorted(v, key=lambda e: e["label"]) for k, v in sorted(out.items())}


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("stem", sorted(SPECS))
def test_non_trunk_bodies_match_the_pre_refactor_golden(stem, golden) -> None:
    now = body_summary(SPECS[stem].read_bytes(), SHORTS)
    before = golden[stem]
    assert set(now) == set(before)
    for body, entries in before.items():
        if library.classify_body(body) in TRUNK_PARTS:
            continue
        assert len(now[body]) == len(entries), body
        for got, want in zip(now[body], entries, strict=True):
            assert got["label"] == want["label"], body
            assert got["material"] == want["material"], body
            assert got["n_vertices"] == want["n_vertices"], body
            np.testing.assert_allclose(got["bbox_min"], want["bbox_min"], atol=2e-6)
            np.testing.assert_allclose(got["bbox_max"], want["bbox_max"], atol=2e-6)


@pytest.mark.parametrize("stem", sorted(SPECS))
def test_trunk_bodies_keep_materials_and_become_one_solid(stem, golden) -> None:
    now = body_summary(SPECS[stem].read_bytes(), SHORTS)
    trunk = [b for b in golden[stem] if library.classify_body(b) in TRUNK_PARTS]
    assert trunk
    for body in trunk:
        before = {e["material"] for e in golden[stem][body]}
        assert {e["material"] for e in now[body]} == before, body
        skins = [e for e in now[body] if e["label"].startswith("skin")]
        assert len(skins) == 1, (body, [e["label"] for e in now[body]])


def _world_mesh_points(mujoco, model, data, geom_id: int) -> np.ndarray:
    mesh = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    local = np.asarray(model.mesh_vert[start : start + count], float)
    rot = np.asarray(data.geom_xmat[geom_id], float).reshape(3, 3)
    return local @ rot.T + np.asarray(data.geom_xpos[geom_id], float)


def _body_xpos(mujoco, model, data, suffix: str) -> np.ndarray:
    names = [model.body(i).name for i in range(model.nbody)]
    (name,) = [n for n in names if n.endswith(suffix)]
    return np.asarray(data.xpos[model.body(name).id], float)


@pytest.mark.parametrize("stem", sorted(SPECS))
def test_chest_is_wider_than_deep_in_the_world_frame(stem) -> None:
    """Rotation trap: the chest mesh is checked after MuJoCo compiles it."""
    mujoco = pytest.importorskip("mujoco")
    raw = SPECS[stem].read_bytes()
    spec = json.loads(raw)
    xml, _ = exporter.export_full_body_mjcf(raw, appearance=document_from_dict(DEFAULT))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    up = -np.asarray(spec["gravity_m_s2"], float)
    up /= np.linalg.norm(up)
    shoulders = _body_xpos(mujoco, model, data, "LUpperArm") - _body_xpos(
        mujoco, model, data, "RUpperArm"
    )
    ml = shoulders - (shoulders @ up) * up
    ml /= np.linalg.norm(ml)
    chest = next(b["name"] for b in spec["bodies"] if b["name"].endswith("COMRod"))
    chest_id = model.body(chest).id
    skins = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == chest_id
        and "_skin" in model.geom(g).name
        and not model.geom(g).name.startswith("visual_head_")  # CC0 head (#12949)
    ]
    assert len(skins) == 1, "one trunk solid on the chest"
    pts = _world_mesh_points(mujoco, model, data, skins[0])
    # The v1 zero pose leans the trunk, so its long axis (principal direction
    # of the mesh across the world shoulder line) need not be vertical;
    # chest-to-back is then measured perpendicular to both.
    planar = pts - np.outer(pts @ ml, ml)
    planar -= planar.mean(axis=0)
    long_axis = np.linalg.eigh(planar.T @ planar)[1][:, -1]
    ap = np.cross(ml, long_axis)
    width, depth, height = (float(np.ptp(pts @ a)) for a in (ml, ap, long_axis))
    size = section_size(spec, chest)
    assert width == pytest.approx(2 * size.radius_m, rel=0.05), (width, size)
    assert depth == pytest.approx(2 * size.radius_m * size.aspect, rel=0.05)
    assert depth < width, ("chest-to-back must be < shoulder-to-shoulder", depth, width)
    assert height > width, "the trunk solid is longer than it is wide"
    tilt = np.degrees(np.arccos(abs(float(long_axis @ up))))
    assert tilt < 45.0, ("trunk long axis is not upright in the world", tilt)


@pytest.mark.parametrize("stem", sorted(SPECS))
def test_physics_is_bit_identical_with_the_shared_body_meshes(stem) -> None:
    mujoco = pytest.importorskip("mujoco")
    raw = SPECS[stem].read_bytes()
    plain = mujoco.MjModel.from_xml_string(exporter.export_full_body_mjcf(raw)[0])
    dressed = mujoco.MjModel.from_xml_string(
        exporter.export_full_body_mjcf(raw, appearance=document_from_dict(DEFAULT))[0]
    )
    assert dressed.nbody == plain.nbody and dressed.nq == plain.nq
    for attr in ("body_mass", "body_inertia", "body_ipos", "body_pos", "jnt_axis"):
        assert np.array_equal(getattr(dressed, attr), getattr(plain, attr)), attr


def test_visual_layer_plain_colours_come_from_the_library_palette() -> None:
    palette = library.PLAIN_PALETTE
    assert visual_layer.plain_rgba("capsule") == palette["capsule"]
    for key in ("capsule", "shape", "com", "frame", "floor", "ball"):
        assert visual_layer.plain_rgba(key) == palette[key]
    assert library.MATERIALS[library.PLAIN_BODY_MATERIAL].rgba == palette["capsule"]


def test_no_plain_palette_literal_remains_in_the_visual_layer() -> None:
    source = Path(visual_layer.__file__).read_text(encoding="utf-8")
    assert '"0.75 0.78 0.85' not in source
    assert "0.75, 0.78, 0.85" not in source


def test_mujoco_emitter_never_reads_classification_or_sizing_tables() -> None:
    """LoD: the emitter translates resolved meshes; it owns no appearance rules."""
    from src.engines.physics_engines.mujoco.python import appearance_layer

    source = Path(appearance_layer.__file__).read_text(encoding="utf-8")
    for name in (
        "PART_RULES",
        "PART_RADIUS_M",
        "PART_ASPECT",
        "GARMENT_COVERAGE",
        "classify_body",
        "body_sections",
        "lofted_segment",
    ):
        assert name not in source, name
