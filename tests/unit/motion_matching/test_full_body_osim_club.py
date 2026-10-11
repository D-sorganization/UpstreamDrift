"""OSV-1 (#11727): the generated full-body OpenSim models show a club."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # test parses repo-owned XML

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import club_visuals
from src.engines.physics_engines.opensim.python.full_body_osim import (
    export_full_body_osim,
)
from src.shared.python.model_appearance import club_head_mesh as chm
from src.shared.python.model_appearance.club_assembly import assembly_from_spec

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "src/engines/physics_engines/opensim/models/generated"
SPEC_DIR = ROOT / "docs/development/full_body_models"
CLUBS = {"driver": "driver", "iron7": "iron7"}


def _parse(xml: str) -> ET.Element:
    """Parse OpenSim XML (its ``Class::Name`` tags are not namespace-clean)."""
    return ET.fromstring(xml.replace("::", "__"))


def _body(root: ET.Element, name: str) -> ET.Element:
    found = root.find(f".//BodySet/objects/Body[@name='{name}']")
    assert found is not None
    return found


@pytest.mark.parametrize("club", sorted(CLUBS))
def test_export_attaches_shaft_grip_and_head_meshes(club) -> None:
    raw = (SPEC_DIR / f"full_body_spec_anthro_{club}.json").read_bytes()
    xml, meta = export_full_body_osim(raw, model_name=f"m_{club}")
    meshes = _body(_parse(xml), "Clubhead").findall("attached_geometry/Mesh")
    assert sorted(m.get("name") for m in meshes) == [
        "club_grip_geom",
        "club_head_geom",
        "club_shaft_geom",
    ]
    for mesh in meshes:
        scale = [float(v) for v in mesh.findtext("scale_factors").split()]
        assert len(scale) == 3 and min(scale) > 0.0
        assert (club_visuals.GEOMETRY_DIR / mesh.findtext("mesh_file")).is_file()
    assert meta["club_geometry"]["head_alias"] == club
    assert _body(_parse(xml), "Grip").find("attached_geometry/Mesh") is None


@pytest.mark.parametrize("club", sorted(CLUBS))
def test_geometry_is_visual_only_masses_and_inertias_unchanged(club) -> None:
    raw = (SPEC_DIR / f"full_body_spec_anthro_{club}.json").read_bytes()
    with_meshes, _ = export_full_body_osim(raw, club_geometry_ref="")
    bare, _ = export_full_body_osim(raw, club_geometry_ref=None)
    assert "<Mesh" not in bare
    for tag in ("mass", "mass_center", "inertia"):
        left = [e.text for e in _parse(with_meshes).iter(tag)]
        right = [e.text for e in _parse(bare).iter(tag)]
        assert left == right


@pytest.mark.parametrize("club", sorted(CLUBS))
def test_committed_models_reference_existing_assets_with_provenance(club) -> None:
    root = _parse((MODELS / f"full_body_anthro_{club}.osim").read_text())
    files = [m.findtext("mesh_file") for m in root.iter("Mesh")]
    assert len(files) == 3
    manifest = json.loads(
        (club_visuals.GEOMETRY_DIR / club_visuals.PROVENANCE_NAME).read_text()
    )
    entry = manifest["clubs"][club]
    assert len(entry["spec_sha256"]) == 64
    for part, name in entry["files"].items():
        data = (club_visuals.GEOMETRY_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"][part]
        assert name in files


@pytest.mark.parametrize("club", sorted(CLUBS))
def test_head_face_normal_matches_the_spec_loft_at_address(club) -> None:
    head = chm.load_club_head(club)
    rot = chm.head_to_club_rotation(head)
    face_c = rot @ chm.measured_face_normal(head.head_mesh)
    up_c = rot @ np.array([0.0, 1.0, 0.0])  # head-up = world vertical at address
    elevation = np.degrees(np.arcsin(float(face_c @ up_c)))
    assert abs(elevation - head.loft_deg) <= 2.0
    spec = json.loads((SPEC_DIR / f"full_body_spec_anthro_{club}.json").read_text())
    assert assembly_from_spec(spec).head_alias == club


def test_generated_osim_loads_in_opensim_with_unchanged_mass() -> None:
    osim = pytest.importorskip("opensim")
    club_visuals.register_geometry_path()
    raw = (SPEC_DIR / "full_body_spec_anthro_driver.json").read_bytes()
    masses = []
    for ref in ("", None):
        xml, _ = export_full_body_osim(raw, club_geometry_ref=ref)
        path = Path(pytest.importorskip("tempfile").mkdtemp()) / "m.osim"
        path.write_text(xml, encoding="utf-8")
        model = osim.Model(str(path))
        state = model.initSystem()
        masses.append(model.getTotalMass(state))
    assert masses[0] == masses[1]


@pytest.mark.parametrize("key", sorted(CLUBS))
def test_club_is_jointed_to_the_lead_hand_and_welded_to_the_trail_hand(key) -> None:
    """OSV-1: the visible club chain is attached to both hands, not free floating."""
    root = _parse((MODELS / f"full_body_anthro_{key}.osim").read_text("utf-8"))
    joints = {j.get("name"): j for j in root.iter("CustomJoint")}
    lead = joints["joint_Clubhead"]
    assert lead.findtext(".//frames/PhysicalOffsetFrame/socket_parent") == "/bodyset/LF"
    assert "/bodyset/Clubhead" in {e.text for e in lead.iter("socket_parent") if e.text}
    assert "joint_Grip" in joints  # trail-hand standoff body
    weld = next(root.iter("WeldConstraint"))
    assert weld.get("name") == "two_hand_grip_closure"
    assert weld.findtext("isEnforced") == "true"
    parents = {e.text for e in weld.iter("socket_parent")}
    assert parents == {"/bodyset/Grip", "/bodyset/Clubhead"}
