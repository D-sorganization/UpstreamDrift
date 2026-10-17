"""OpenSim and MyoSuite viewers adopt the shared appearance (R9b, #12689).

OpenSim registers R8's resolved body meshes (``backends._body``) on each body
in the spec body frame, which is the OpenSim body frame because the exporter
attaches every ``child_to_follower`` as a separate offset frame. MyoSuite
passes the same appearance document to the MJCF exporter, so its visual geoms
and materials are exactly the MuJoCo export's. No local colours remain.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.shared.python.model_appearance import library, library_materials
from src.shared.python.model_appearance.body_meshes import resolve_appearance
from src.shared.python.model_appearance.cross_section import body_sections
from src.tools.native_viewer_export.backends._body import default_appearance
from src.tools.native_viewer_export.backends._club import club_parts

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[4]
SPECS = REPO / "docs/development/full_body_models"
SPEC_NAMES = ("anthro_driver", "anthro_iron7")
PELVIS = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
TRUNK = (
    PELVIS,
    "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/UpperTorsoBase",
)
PRIMITIVES = ("cylinder", "sphere", "ellipsoid", "brick")


def _spec_bytes(name: str) -> bytes:
    return (SPECS / f"full_body_spec_{name}.json").read_bytes()


def _opensim_worker():
    from src.tools.native_viewer_export.backends import opensim_worker

    return opensim_worker


def _obj_vertices(path: Path) -> np.ndarray:
    rows = [
        [float(x) for x in line.split()[1:4]]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    return np.asarray(rows, dtype=float)


@pytest.fixture(scope="module", params=SPEC_NAMES)
def dressed(request, tmp_path_factory):
    spec = json.loads(_spec_bytes(request.param))
    out = tmp_path_factory.mktemp(f"dressed_{request.param}")
    parts = _opensim_worker().visual_parts(spec, default_appearance(), out)
    return spec, parts


def test_every_dressed_body_has_mesh_geometry_and_no_primitive(dressed) -> None:
    spec, parts = dressed
    resolved = resolve_appearance(spec, default_appearance())
    club_body = club_parts(spec)[0]
    expected = {m.body for m in resolved.meshes} - {club_body}
    meshed = {p.body for p in parts if p.geometry == "mesh"}
    assert expected <= meshed
    assert not [p for p in parts if p.geometry in PRIMITIVES]


def test_club_body_keeps_the_exported_club_meshes_only(dressed) -> None:
    spec, parts = dressed
    assert club_parts(spec)[0] not in {p.body for p in parts}


def test_pelvis_wears_the_trousers_colour(dressed) -> None:
    _, parts = dressed
    trousers = library_materials(default_appearance())["trousers_charcoal"].rgba
    assert trousers in {p.rgba for p in parts if p.body == PELVIS}


def test_every_colour_comes_from_the_library(dressed) -> None:
    spec, parts = dressed
    allowed = {m.rgba for m in library_materials(default_appearance()).values()}
    from src.tools.native_viewer_export.backends._head import head_mesh_files

    allowed |= {h.rgba for h in head_mesh_files(spec, Path(parts[0].params[0]).parent)}
    assert {p.rgba for p in parts} <= allowed


def test_meshes_sit_in_the_spec_body_frame_untransformed(dressed) -> None:
    _, parts = dressed
    for part in (p for p in parts if p.geometry == "mesh"):
        np.testing.assert_array_equal(part.rotation, np.eye(3))
        assert tuple(part.position) == (0.0, 0.0, 0.0)


def test_trunk_mesh_is_wider_mediolaterally_than_deep(dressed) -> None:
    spec, parts = dressed
    sections = body_sections(spec)
    for body in TRUNK:
        skin = [
            p
            for p in parts
            if p.body == body and Path(p.params[0]).stem.endswith("skin0")
        ]
        assert skin, body
        verts = _obj_vertices(Path(skin[0].params[0]))
        ml = np.ptp(verts @ np.asarray(sections[body].ml_axis, float))
        ap = np.ptp(verts @ np.asarray(sections[body].depth_axis, float))
        assert ml > ap, body


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_plain_mode_uses_the_library_plain_palette(name: str, tmp_path) -> None:
    spec = json.loads(_spec_bytes(name))
    parts = _opensim_worker().visual_parts(spec, None, tmp_path)
    capsule = library.plain_rgba("capsule")
    shape = library.plain_rgba("shape")
    prims = [p for p in parts if p.geometry in PRIMITIVES]
    assert prims
    assert {p.rgba for p in prims} <= {capsule, shape}
    assert {p.rgba for p in parts if p.geometry in ("cylinder", "sphere")} == {capsule}


def test_no_local_grey_literal_remains() -> None:
    src = (
        REPO / "src/tools/native_viewer_export/backends/opensim_worker.py"
    ).read_text(encoding="utf-8")
    assert "0.75, 0.78, 0.85" not in src
    assert "0.7, 0.72, 0.8" not in src


def _material_summary(xml: str) -> dict[str, list[str]]:
    """Sorted visual-geom material names per MJCF body (the golden comparison)."""
    out: dict[str, list[str]] = defaultdict(list)
    for body in ET.fromstring(xml).iter("body"):
        for geom in body.findall("geom"):
            if geom.get("material"):
                out[str(body.get("name"))].append(str(geom.get("material")))
    return {k: sorted(v) for k, v in sorted(out.items())}


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_myosuite_mjcf_carries_exactly_the_mujoco_appearance(name: str) -> None:
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml
    from src.tools.native_viewer_export.backends.myosuite_worker import build_spec_xml

    spec_bytes = _spec_bytes(name)
    myo, _ = build_spec_xml(spec_bytes, default_appearance())
    mj, _ = build_scene_xml(spec_bytes)
    summary = _material_summary(myo)
    assert summary == _material_summary(mj)
    assert len(summary) >= 20
    assert any("trousers_charcoal" in m for m in summary.get(PELVIS, []))


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_myosuite_plain_mode_keeps_the_capsule_layer_with_head(name: str) -> None:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.tools.native_viewer_export.backends.myosuite_worker import build_spec_xml

    spec_bytes = _spec_bytes(name)
    plain, _ = build_spec_xml(spec_bytes, None)
    legacy, _ = export_full_body_mjcf(spec_bytes, visual=True, with_head=True)
    assert plain == legacy
