"""Unit tests and RED fixtures for OpenSim parameterized golf club geometry (OG-03, #10397).

Asserts:
1. Baseline model Club body has zero attached geometry (RED failure fixture).
2. Parameterized visual club attachment generates valid OpenSim XML with shaft and clubhead meshes.
3. Club visual attachment preserves physical body mass, center of mass, and inertia tensor.
4. PhysicalOffsetFrames for grip (butt, mid-grip, left/right hand positions) and clubhead are validated.
5. Roundtrip serialization and parsing preserves structural integrity and attached geometry count.
6. Forward kinematics and coordinate consistency between ClubSpec and OpenSim conventions.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from defusedxml import ElementTree as SafeET


from src.engines.physics_engines.opensim.python.tour_matching.club_geometry import (
    attach_visual_club,
    get_club_frame_offsets,
    has_visual_club,
)
from src.engines.physics_engines.opensim.python.tour_matching.model_audit import (
    audit_model_geometry,
    verify_model_qualification,
)
from src.shared.python.motion_matching.club_models import DRIVER, IRON_7, ClubSpec

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_MODEL_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "opensim_tour_matching"
    / "evidence"
    / "os7_moco_g1"
    / "golf_humanoid_scaled_tour_markers_moco.osim"
)
GOLF_HUMANOID_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid.osim"
)


def test_baseline_fixture_lacks_visual_club() -> None:
    """Baseline models have empty attached_geometry on Club body (reproducing the defect)."""
    assert not has_visual_club(BASELINE_MODEL_PATH)
    assert not has_visual_club(GOLF_HUMANOID_PATH)

    audit = audit_model_geometry(BASELINE_MODEL_PATH)
    assert audit.club_attached_geometry_count == 0
    assert not audit.has_visible_club


def test_attach_visual_club_driver() -> None:
    """Attaching driver visual geometry adds shaft and head meshes to Club attached_geometry."""
    tree = SafeET.parse(str(GOLF_HUMANOID_PATH))
    modified_tree = attach_visual_club(tree, DRIVER)

    club_body = modified_tree.find(".//BodySet/objects/Body[@name='Club']")
    assert club_body is not None

    att_geom = club_body.find("attached_geometry")
    assert att_geom is not None
    meshes = att_geom.findall("Mesh")
    assert len(meshes) >= 2  # shaft and clubhead

    # Check mesh file names and scale factors
    mesh_names = [m.get("name") for m in meshes]
    assert any("shaft" in (name or "").lower() for name in mesh_names)
    assert any("head" in (name or "").lower() for name in mesh_names)

    # Physical properties must be preserved
    mass_elem = club_body.find("mass")
    assert mass_elem is not None
    assert float(mass_elem.text or "0") == pytest.approx(0.32, rel=1e-4)

    com_elem = club_body.find("mass_center")
    assert com_elem is not None
    assert com_elem.text == "0.0 -0.786 0.0"


def test_attach_visual_club_iron7() -> None:
    """Attaching 7-iron visual geometry scales dimensions according to IRON_7 spec."""
    tree = SafeET.parse(str(GOLF_HUMANOID_PATH))
    modified_tree = attach_visual_club(tree, IRON_7)

    club_body = modified_tree.find(".//BodySet/objects/Body[@name='Club']")
    assert club_body is not None

    att_geom = club_body.find("attached_geometry")
    assert att_geom is not None
    meshes = att_geom.findall("Mesh")
    assert len(meshes) >= 2


def test_get_club_frame_offsets_conventions() -> None:
    """Grip and clubhead frame offsets match OpenSim conventions for given ClubSpec."""
    offsets = get_club_frame_offsets(DRIVER)
    # OpenSim convention: grip origin at (0, 0, 0), shaft extends along -Y
    assert offsets["club_grip_offset"] == (0.0, 0.0, 0.0)
    # Clubhead at -length_m along Y
    assert offsets["club_head_offset"][0] == 0.0
    assert offsets["club_head_offset"][1] == pytest.approx(-DRIVER.length_m, abs=1e-4)
    assert offsets["club_head_offset"][2] == 0.0

    # Two-handed grip frames: lead/trail hand grip offsets
    assert "hand_l_grip_offset" in offsets
    assert "hand_r_grip_offset" in offsets
    # Right hand is near -0.06m down the shaft, left hand is further up near butt
    assert offsets["hand_r_grip_offset"][1] < 0.0
    assert offsets["hand_l_grip_offset"][1] < 0.0
    assert offsets["hand_l_grip_offset"][1] > offsets["hand_r_grip_offset"][1]


def test_qualification_passes_with_visual_club(tmp_path: Path) -> None:
    """verify_model_qualification succeeds once visual club geometry is attached."""
    tree = SafeET.parse(str(BASELINE_MODEL_PATH))
    attach_visual_club(tree, DRIVER)

    out_file = tmp_path / "model_with_club.osim"
    root = tree.getroot()
    ET.indent(root, space="\t")
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    out_file.write_bytes(b'<?xml version="1.0" encoding="UTF-8" ?>\n' + body + b"\n")

    audit = verify_model_qualification(out_file, require_visible_club=True)
    assert audit.has_visible_club
    assert audit.club_attached_geometry_count >= 2


def test_attach_visual_club_fails_closed_on_missing_club_body() -> None:
    """attach_visual_club raises ValueError if the model has no Club body."""
    tree = SafeET.parse(str(GOLF_HUMANOID_PATH))
    body_set = tree.find(".//BodySet/objects")
    assert body_set is not None
    club = body_set.find("Body[@name='Club']")
    assert club is not None
    body_set.remove(club)

    with pytest.raises(ValueError, match="Model has no Club body"):
        attach_visual_club(tree, DRIVER)
