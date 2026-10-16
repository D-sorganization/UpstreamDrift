"""Drake and Pinocchio draw the shared appearance spec (R9a, #12688).

Engine-free where possible: both MeshCat backends are thin adapters over one
scene builder (``backends._meshcat_scene``) that reads ``model_appearance``,
so these tests run on the scene records and the OBJ files they write. Tests
that build the real engine objects are marked ``requires_drake`` /
``requires_pinocchio``. The MuJoCo layer consumes the same resolution, and
the cross-engine test checks colours and rigid-invariant mesh dimensions
against the MJCF it exports.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.backends import (
    _meshcat_scene,
    drake_meshcat,
    pinocchio_meshcat,
)
from src.tools.native_viewer_export.backends._meshcat_scene import (
    SceneVisuals,
    VisualItem,
    build_scene_visuals,
    check_bodies,
    default_appearance,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
MODELS = ROOT / "docs/development/full_body_models"
ANTHRO = (
    MODELS / "full_body_spec_anthro_driver.json",
    MODELS / "full_body_spec_anthro_iron7.json",
)
BACKENDS = ("drake", "pinocchio")
# Tolerance for the cross-engine comparison: MJCF writes numbers with 6
# significant digits, so colours agree to 1e-5 and distances to 1e-5 m.
TOL = 1e-5
TRUNK_PARTS = ("torso", "pelvis")


def _spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_obj(path: Path) -> np.ndarray:
    rows = [
        [float(x) for x in line.split()[1:]]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    return np.asarray(rows)


@pytest.fixture(scope="module", params=ANTHRO, ids=lambda p: p.stem)
def built(request, tmp_path_factory) -> tuple[dict[str, Any], SceneVisuals]:
    spec = _spec(request.param)
    out = tmp_path_factory.mktemp("scene")
    return spec, build_scene_visuals(spec, out, default_appearance())


def _scene_for(backend: str, tmp_path: Path, spec: dict[str, Any]) -> SceneVisuals:
    """What the named backend's adapter draws (one builder serves both)."""
    cls = {
        "drake": drake_meshcat.DrakeMeshcatBackend,
        "pinocchio": pinocchio_meshcat.PinocchioMeshcatBackend,
    }[backend]
    appearance = cls()._appearance  # noqa: SLF001 - the document the backend renders
    return build_scene_visuals(spec, tmp_path, appearance)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", ANTHRO, ids=lambda p: p.stem)
def test_backend_draws_the_default_appearance_document(backend, path, tmp_path):
    scene = _scene_for(backend, tmp_path, _spec(path))
    assert scene.appearance
    assert scene.items
    assert {i.kind for i in scene.items} == {"mesh"}  # no capsule or hint left


def test_every_dressed_body_has_registered_meshes(built) -> None:
    spec, scene = built
    doc = default_appearance()
    drawn = {s.body for s in derive_visual_skeleton(spec).capsules}
    for body in drawn:
        rule = library.rule_for_body(doc, body)
        if rule is not None and rule.mesh == "none":
            continue
        assert body in scene.bodies(), f"{body} has no registered mesh"


def test_pelvis_pants_trunk_shirt_feet_shoe_materials(built) -> None:
    _, scene = built
    doc = default_appearance()
    pants = library.pants_material(doc.clothing)
    garments = [i for i in scene.items if i.label.startswith("garment")]
    by_part: dict[str, set[str | None]] = {}
    for item in garments:
        part = library.classify_body(item.body)
        by_part.setdefault(part, set()).add(item.material)
    assert by_part["pelvis"] == {pants}
    assert by_part["torso"] == {library.CLOTHING[doc.clothing]["torso"]}
    assert by_part["torso"] != by_part["pelvis"]
    feet = {i.material for i in garments if library.classify_body(i.body) == "foot"}
    assert feet == {library.CLOTHING[doc.clothing]["foot"]}


def test_material_rgba_equals_the_library(built) -> None:
    _, scene = built
    materials = library.library_materials(default_appearance())
    dressed = [i for i in scene.items if i.material is not None]
    assert dressed
    for item in dressed:
        assert item.rgba == materials[item.material].rgba, item.name


def test_trunk_is_wider_mediolaterally_than_anteroposteriorly(built) -> None:
    spec, scene = built
    sections = body_sections(spec)
    seen = set()
    for item in scene.items:
        part = library.classify_body(item.body)
        if part not in TRUNK_PARTS or not item.label.startswith("skin"):
            continue
        if item.body not in sections:
            continue
        verts = _read_obj(item.mesh_path)  # spec body frame
        ml = float(np.ptp(verts @ sections[item.body].ml_axis))
        ap = float(np.ptp(verts @ sections[item.body].depth_axis))
        assert ml > ap, (item.body, ml, ap)
        seen.add(part)
    assert seen == set(TRUNK_PARTS)


def test_trunk_ml_extent_is_the_spec_hint_semi_axis(built) -> None:
    """Trunk sized from the spec's own hints (e.g. 0.158 m ML on the torso)."""
    spec, scene = built
    sections = body_sections(spec)
    hints = {s.body for s in derive_visual_skeleton(spec).shapes}
    from src.shared.python.model_appearance.body_meshes import section_size

    checked = 0
    for item in scene.items:
        if (
            library.classify_body(item.body) in TRUNK_PARTS
            and item.label.startswith("skin")
            and item.body in hints
        ):
            ml = float(np.ptp(_read_obj(item.mesh_path) @ sections[item.body].ml_axis))
            assert ml == pytest.approx(
                2 * section_size(spec, item.body).radius_m, rel=0.05
            )
            checked += 1
    assert checked >= 2


def test_head_is_the_shared_parametric_head_in_its_body_frame(built) -> None:
    _, scene = built
    heads = [i for i in scene.items if i.label.startswith("head_")]
    assert {"head_skull", "head_neck", "head_nose"} <= {i.label for i in heads}
    assert len({i.body for i in heads}) == 1 and heads[0].body.endswith("/Head")


def test_head_disabled_leaves_the_head_body_to_the_body_meshes(tmp_path) -> None:
    spec = _spec(ANTHRO[0])
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"enabled": False}}
    )
    scene = build_scene_visuals(spec, tmp_path, doc)
    assert not [i for i in scene.items if i.label.startswith("head_")]
    assert any(i.body.endswith("/Head") for i in scene.items)


def test_plain_mode_uses_the_library_plain_palette(tmp_path) -> None:
    spec = _spec(ANTHRO[0])
    scene = build_scene_visuals(spec, tmp_path, None)
    assert not scene.appearance
    plain = {
        "capsule": library.plain_rgba("capsule"),
        "ellipsoid": library.plain_rgba("shape"),
        "box": library.plain_rgba("shape"),
    }
    primitives = [i for i in scene.items if i.kind != "mesh"]
    assert {i.kind for i in primitives} >= {"capsule"}
    for item in primitives:
        assert item.rgba == plain[item.kind]
    assert scene.floor_rgba == library.plain_rgba("floor")
    assert any(i.label.startswith("head_") for i in scene.items)  # head stays
    assert any(i.label.startswith("club_") for i in scene.items)  # club stays


def test_floor_is_the_document_ground_material(built) -> None:
    _, scene = built
    doc = default_appearance()
    expected = library.library_materials(doc)[doc.environment.ground_material].rgba
    assert scene.floor_rgba == expected


@pytest.mark.parametrize("module", (drake_meshcat, pinocchio_meshcat))
def test_backends_carry_no_local_palette(module) -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\d\.\d+\s*,\s*\d\.\d+\s*,\s*\d\.\d+\s*,\s*[01]\.\d+", source)
    assert "_RGBA" not in source
    assert "_meshcat_scene" in source  # the shared scene supplies every colour


def test_backends_share_one_scene_module() -> None:
    assert drake_meshcat.build_scene_visuals is pinocchio_meshcat.build_scene_visuals
    assert drake_meshcat.default_appearance is _meshcat_scene.default_appearance


def test_scene_module_has_no_colour_literals() -> None:
    source = Path(_meshcat_scene.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\d\.\d+\s*,\s*\d\.\d+\s*,\s*\d\.\d+\s*,\s*[01]\.\d+", source)


# ---- design by contract: fail closed ------------------------------------


def test_item_rejects_a_bad_colour_kind_and_size() -> None:
    with pytest.raises(ValueError, match="rgba"):
        VisualItem("x", "b", "capsule", (2.0, 0.0, 0.0, 1.0), size=(0.1, 0.2))
    with pytest.raises(ValueError, match="unknown visual kind"):
        VisualItem("x", "b", "cone", (0.1, 0.1, 0.1, 1.0), size=(0.1,))
    with pytest.raises(ValueError, match="positive"):
        VisualItem("x", "b", "box", (0.1, 0.1, 0.1, 1.0), size=(0.1, -1.0, 0.1))
    with pytest.raises(ValueError, match="mesh kind"):
        VisualItem("x", "b", "mesh", (0.1, 0.1, 0.1, 1.0))


def test_items_on_unknown_bodies_fail_closed(built) -> None:
    _, scene = built
    with pytest.raises(ValueError, match="engine lacks"):
        check_bodies(scene, {"only/this/body"})
    check_bodies(scene, scene.bodies())


def test_unknown_material_in_the_document_fails_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="no_such_skin"):
        doc = document_from_dict(
            {"schema_version": "appearance-v1", "skin_tone": "no_such_skin"}
        )
        build_scene_visuals(_spec(ANTHRO[0]), tmp_path, doc)


def test_a_document_that_dresses_nothing_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_meshcat_scene, "_appearance_items", lambda *_: [])
    with pytest.raises(ValueError, match="dressed no body"):
        build_scene_visuals(_spec(ANTHRO[0]), tmp_path, default_appearance())


# ---- cross-engine: the MuJoCo layer draws the same colours and sizes ----


def _signature(vertices: np.ndarray) -> np.ndarray:
    """Rigid-motion invariant of a vertex cloud: sorted centroid distances."""
    return np.sort(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1))


def _mujoco_meshes(raw: bytes) -> dict[tuple[str, str], tuple[tuple, np.ndarray]]:
    from src.engines.physics_engines.mujoco.python import full_body_mjcf

    xml, _ = full_body_mjcf.export_full_body_mjcf(raw, appearance=default_appearance())
    root = ET.fromstring(xml)
    rgba = {
        m.get("name"): tuple(float(x) for x in (m.get("rgba") or "").split())
        for m in root.iter("material")
    }
    verts = {
        m.get("name"): np.array((m.get("vertex") or "").split(), float).reshape(-1, 3)
        for m in root.iter("mesh")
    }
    out = {}
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            name = geom.get("mesh") or ""
            match = re.fullmatch(r"vmesh_(?:\d+_)?(.+)", name)
            if match is None:
                continue
            out[(body.get("name") or "", match.group(1))] = (
                rgba[geom.get("material") or ""],
                _signature(verts[name]),
            )
    return out


@pytest.mark.parametrize("path", ANTHRO, ids=lambda p: p.stem)
def test_colours_and_dimensions_match_the_mujoco_layer(path, tmp_path) -> None:
    pytest.importorskip("mujoco")
    mujoco_side = _mujoco_meshes(path.read_bytes())
    scene = build_scene_visuals(_spec(path), tmp_path, default_appearance())
    compared = 0
    for item in scene.items:
        other = mujoco_side.get((item.body, item.label))
        if other is None:
            continue  # club parts carry engine-specific names in MJCF
        rgba, signature = other
        np.testing.assert_allclose(rgba, item.rgba, atol=TOL, err_msg=item.name)
        mine = _signature(_read_obj(item.mesh_path))
        assert mine.shape == signature.shape, item.name
        np.testing.assert_allclose(mine, signature, atol=TOL, err_msg=item.name)
        compared += 1
    skins = [i for i in scene.items if i.label.startswith("skin")]
    assert compared >= len(skins) + 6  # skins, garments and the head all match


# ---- real engines (skipped without pydrake / pinocchio) ------------------


@pytest.mark.requires_pinocchio
def test_pinocchio_geometry_model_carries_the_scene_colours() -> None:
    pin = pytest.importorskip("pinocchio")
    coal = pytest.importorskip("coal")
    pytest.importorskip("meshcat")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    spec = _spec(ANTHRO[0])
    backend = pinocchio_meshcat.PinocchioMeshcatBackend()
    adapter = FullBodyPinocchioModel(spec)
    scene = build_scene_visuals(
        spec,
        Path(__import__("tempfile").mkdtemp()),
        backend._appearance,  # noqa: SLF001
    )
    geometry = pin.GeometryModel()
    for item in scene.items:
        joint, body_pose = adapter._bodies[item.body]  # noqa: SLF001
        shape = coal.MeshLoader().load(str(item.mesh_path))
        obj = pin.GeometryObject(
            item.name, joint, body_pose, shape, str(item.mesh_path)
        )
        obj.meshColor = np.array(item.rgba)
        geometry.addGeometryObject(obj)
    assert geometry.ngeoms == len(scene.items)
    for item, obj in zip(scene.items, geometry.geometryObjects, strict=True):
        np.testing.assert_allclose(obj.meshColor, item.rgba, atol=1e-12)


@pytest.mark.requires_drake
def test_drake_registers_every_scene_item_as_visual_geometry() -> None:
    pytest.importorskip("pydrake.multibody.parsing")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    from src.tools.native_viewer_export.backends._subprocess import export_urdf

    spec_bytes = ANTHRO[0].read_bytes()
    xml, meta = export_urdf(spec_bytes)
    plant, _ = AddMultibodyPlantSceneGraph(DiagramBuilder(), 0.0)
    inst = Parser(plant).AddModelsFromString(xml, "urdf")[0]
    scene = build_scene_visuals(
        json.loads(spec_bytes),
        Path(__import__("tempfile").mkdtemp()),
        default_appearance(),
    )
    check_bodies(scene, meta["body_links"])
    drake_meshcat.DrakeMeshcatBackend()._register_visuals(  # noqa: SLF001
        plant, inst, meta["body_links"], scene
    )
    assert plant.geometry_source_is_registered()
