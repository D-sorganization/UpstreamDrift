"""``contact`` grip model of the OpenSim exporter, XML level (#11739, OSV-7).

No OpenSim needed: the exported document is parsed with defusedxml.
"""

from __future__ import annotations

import json
from pathlib import Path

import defusedxml.ElementTree as ET
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.full_body_grip_contact import (
    CONTACT_FORCE_PREFIX,
    ContactGripConfig,
)
from src.engines.physics_engines.opensim.python.full_body_osim import (
    export_full_body_osim,
)
from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.pad_contact import build_pad_model

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
SQUEEZE_N = 1104.0


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exported(spec: dict, tmp_path_factory: pytest.TempPathFactory):
    mesh_dir = tmp_path_factory.mktemp("grip_meshes")
    interface = GripInterface.from_spec(spec)
    cfg = ContactGripConfig(build_pad_model(interface, SQUEEZE_N), mesh_dir)
    xml, meta = export_full_body_osim(
        spec, grip_model="contact", grip_interface=interface, grip_contact=cfg
    )
    return ET.fromstring(xml.replace("::", "__")), meta, cfg


def test_config_needs_an_existing_absolute_directory(spec: dict, tmp_path: Path):
    pads = build_pad_model(GripInterface.from_spec(spec), SQUEEZE_N)
    with pytest.raises(ValueError, match="mesh_dir"):
        ContactGripConfig(pads, Path("relative/dir"))
    with pytest.raises(ValueError, match="mesh_dir"):
        ContactGripConfig(pads, tmp_path / "missing")


def test_topology_is_the_free_club_without_weld_or_bushing(exported) -> None:
    root, meta, _ = exported
    assert meta["grip_model"] == "contact"
    assert root.find(".//WeldConstraint") is None
    assert root.find(".//BushingForce") is None
    assert root.find(".//FreeJoint") is not None
    assert meta["coordinate_count"] == 44 + 6


def test_one_force_and_sphere_per_pad_one_mesh_per_hand(exported) -> None:
    root, meta, cfg = exported
    n = 2 * cfg.pads.layout.pad_count
    forces = [
        e.get("name")
        for e in root.findall(".//ElasticFoundationForce")
        if e.get("name", "").startswith(CONTACT_FORCE_PREFIX)
    ]
    assert len(forces) == n == 24
    spheres = [e for e in root.findall(".//ContactSphere") if "grip_pad_" in e.get("name")]
    assert len(spheres) == n
    assert {e.get("name") for e in root.findall(".//ContactMesh")} == {
        "grip_mesh_L",
        "grip_mesh_R",
    }
    assert meta["grip_contact"]["pads_per_hand"] == 12


def test_parameters_are_the_matched_foundation_parameters(exported) -> None:
    root, _, cfg = exported
    ef = cfg.ef
    params = root.find(".//ElasticFoundationForce__ContactParameters")
    assert float(params.find("stiffness").text) == pytest.approx(ef.stiffness_n_m3)
    assert float(params.find("dissipation").text) == pytest.approx(ef.dissipation_s_m)
    assert float(params.find("static_friction").text) == pytest.approx(0.9)
    assert float(params.find("dynamic_friction").text) == pytest.approx(0.7)
    assert params.find("geometry").text.split() == ["grip_pad_L00", "grip_mesh_L"]


def test_meshes_are_closed_files_on_the_club_body(exported) -> None:
    root, meta, cfg = exported
    for side in "LR":
        assert cfg.mesh_path(side).is_file()
        lines = cfg.mesh_path(side).read_text().splitlines()
        assert sum(s.startswith("f ") for s in lines) == meta["grip_contact"][
            "meshes"
        ][side]["faces"]
    mesh = next(e for e in root.findall(".//ContactMesh") if e.get("name") == "grip_mesh_L")
    assert mesh.find("socket_frame").text == "/bodyset/Clubhead"
    assert Path(mesh.find("filename").text).is_absolute()


def test_pads_sit_on_the_hand_bodies_at_the_frame_positions(
    exported, spec: dict
) -> None:
    root, _, cfg = exported
    pad = next(e for e in root.findall(".//ContactSphere") if e.get("name") == "grip_pad_R03")
    loc = np.array(pad.find("location").text.split(), float)
    from src.engines.physics_engines.opensim.python.full_body_grip_topology import (
        build_bushing_spec,
    )

    interface = GripInterface.from_spec(spec)
    frame = np.asarray(build_bushing_spec(spec, interface)["grip_bushing"]["R"]["hand_frame"])
    local = cfg.ef.layout.positions_grip_frame("R")[3]
    np.testing.assert_allclose(loc, frame[:3, :3] @ local + frame[:3, 3], atol=1e-12)
    assert pad.find("radius").text and float(pad.find("radius").text) == 0.008


def test_pad_centres_rest_inside_the_grip_surface_by_the_preload(exported) -> None:
    _, _, cfg = exported
    layout = cfg.ef.layout
    for side in "LR":
        axis_pt = layout.axis_offset_grip_frame(side)
        rho = np.linalg.norm(
            layout.positions_grip_frame(side)[:, 1:] - axis_pt[1:], axis=1
        )
        penetration = layout.grip_radius_m + layout.pad_radius_m - rho
        np.testing.assert_allclose(penetration, cfg.ef.preload_penetration_m, atol=1e-12)
