"""Parameterized golf club visual geometry and grip offset frames (OG-03, #10397).

Provides parameterized visual geometry attachments and coordinate frames for
the OpenSim Club body according to shared ``ClubSpec`` parameters:
1. Inspects whether a model has visual geometry on the Club body.
2. Generates parameterized OpenSim visual mesh elements (shaft and clubhead)
   scaled from ``ClubSpec`` parameters without altering physical inertia or mass.
3. Computes canonical grip (butt, mid-grip, two-handed grip offsets) and
   clubhead frames in the OpenSim club body coordinate system.
4. Attaches visual geometry and grip frames into an OpenSim XML ElementTree.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml

from defusedxml import ElementTree as SafeET


from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.club_models import (
    DRIVER,
    IRON_7,
    ClubSpec,
)

# OpenSim default club shaft and head visual mesh identifiers
DEFAULT_SHAFT_MESH_FILE: str = "golf_club_shaft.vtp"
DEFAULT_HEAD_MESH_FILE: str = "golf_club_head.vtp"

# Default visual appearance parameters
DEFAULT_SHAFT_COLOR: str = "0.7 0.7 0.7"
DEFAULT_SHAFT_OPACITY: str = "1.0"
DEFAULT_HEAD_COLOR: str = "0.2 0.2 0.2"
DEFAULT_HEAD_OPACITY: str = "1.0"

# OpenSim hand grip anatomical offsets on the club shaft (metres from butt origin)
# Hand R (trail hand) is distally placed ~0.06 m down the shaft
TRAIL_HAND_OFFSET_M: float = -0.06
# Hand L (lead hand) is proximally placed ~0.025 m down the shaft
LEAD_HAND_OFFSET_M: float = -0.025


def has_visual_club(model_input: Path | str | ET.ElementTree) -> bool:
    """Return True if the OpenSim model's Club body has attached visual geometry."""
    if isinstance(model_input, ET.ElementTree):
        tree = model_input
    else:
        path = Path(model_input)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        tree = SafeET.parse(str(path))

    root = tree.getroot()
    if root is None:
        return False

    club_body = root.find(".//BodySet/objects/Body[@name='Club']")
    if club_body is None:
        return False

    att_geom = club_body.find("attached_geometry")
    if att_geom is None:
        return False

    return len(list(att_geom)) > 0


def get_club_frame_offsets(spec: ClubSpec) -> dict[str, tuple[float, float, float]]:
    """Compute physical offset frame translations for grip and clubhead in OpenSim club body frame.

    OpenSim Club body convention:
    - Origin (0, 0, 0) is at the butt end / grip origin of the shaft.
    - Shaft points along the negative Y axis toward -length_m.
    - Clubhead is positioned at (0, -length_m, 0).
    - Trail hand (R) grip sits at (0, -0.06, 0).
    - Lead hand (L) grip sits at (0, -0.025, 0).

    Preconditions:
    - spec must be a valid ClubSpec with length_m > 0.

    Postconditions:
    - Returns dictionary containing club_grip_offset, club_head_offset,
      hand_r_grip_offset, and hand_l_grip_offset.
    """
    require(isinstance(spec, ClubSpec), "spec must be an instance of ClubSpec")
    require(spec.length_m > 0, "spec.length_m must be strictly positive")

    offsets: dict[str, tuple[float, float, float]] = {
        "club_grip_offset": (0.0, 0.0, 0.0),
        "club_head_offset": (0.0, -float(spec.length_m), 0.0),
        "hand_r_grip_offset": (0.0, TRAIL_HAND_OFFSET_M, 0.0),
        "hand_l_grip_offset": (0.0, LEAD_HAND_OFFSET_M, 0.0),
    }

    ensure(
        offsets["club_grip_offset"] == (0.0, 0.0, 0.0),
        "club_grip_offset must be at butt origin",
    )
    ensure(
        offsets["club_head_offset"][1] < 0.0, "clubhead offset must be along negative Y"
    )
    return offsets


def _create_mesh_element(
    name: str,
    mesh_file: str,
    scale_factors: tuple[float, float, float],
    color: str = "1 1 1",
    opacity: str = "1",
) -> ET.Element:
    """Create an OpenSim <Mesh> XML element."""
    mesh = ET.Element("Mesh", attrib={"name": name})
    ET.SubElement(mesh, "socket_frame").text = ".."
    ET.SubElement(
        mesh, "scale_factors"
    ).text = f"{scale_factors[0]:.6g} {scale_factors[1]:.6g} {scale_factors[2]:.6g}"
    app = ET.SubElement(mesh, "Appearance")
    ET.SubElement(app, "opacity").text = opacity
    ET.SubElement(app, "color").text = color
    ET.SubElement(mesh, "mesh_file").text = mesh_file
    return mesh


def attach_visual_club(
    tree: ET.ElementTree,
    spec: ClubSpec | None = None,
    *,
    shaft_mesh_file: str = DEFAULT_SHAFT_MESH_FILE,
    head_mesh_file: str = DEFAULT_HEAD_MESH_FILE,
) -> ET.ElementTree:
    """Attach parameterized visual geometry to the Club body in an OpenSim ElementTree.

    Modifies the attached_geometry of the <Body name="Club"> element by adding
    shaft and clubhead visual meshes parameterized from ``spec``. Preserves existing
    mass, center of mass, and inertia properties.

    Preconditions:
    - tree must contain a Body with name="Club".

    Postconditions:
    - Club attached_geometry has at least 2 visual geometry components.
    - Physical mass and inertia are unmodified.
    """
    club_spec = spec if spec is not None else DRIVER
    require(
        isinstance(club_spec, ClubSpec), "club_spec must be an instance of ClubSpec"
    )

    root = tree.getroot()
    require(root is not None, "ElementTree must have a valid root")
    assert root is not None

    club_body = root.find(".//BodySet/objects/Body[@name='Club']")
    if club_body is None:
        raise ValueError("Model has no Club body")

    att_geom = club_body.find("attached_geometry")
    if att_geom is None:
        att_geom = ET.SubElement(club_body, "attached_geometry")

    # Clear existing attached_geometry on Club
    for child in list(att_geom):
        att_geom.remove(child)

    # Scale factors:
    # Shaft mesh: nominal length 1.0m, nominal radius 0.0065m
    shaft_scale_x = club_spec.shaft_radius_m / 0.0065
    shaft_scale_y = club_spec.length_m / 1.0
    shaft_scale_z = club_spec.shaft_radius_m / 0.0065
    shaft_scales = (shaft_scale_x, shaft_scale_y, shaft_scale_z)

    # Head mesh: nominal dimensions scaled from head_half_size_m
    hx, hy, hz = club_spec.head_half_size_m
    head_scales = (hx * 2.0, hy * 2.0, hz * 2.0)

    # Attach shaft mesh
    shaft_mesh = _create_mesh_element(
        name="club_shaft_geom",
        mesh_file=shaft_mesh_file,
        scale_factors=shaft_scales,
        color=DEFAULT_SHAFT_COLOR,
        opacity=DEFAULT_SHAFT_OPACITY,
    )
    att_geom.append(shaft_mesh)

    # Attach clubhead mesh
    head_mesh = _create_mesh_element(
        name="club_head_geom",
        mesh_file=head_mesh_file,
        scale_factors=head_scales,
        color=DEFAULT_HEAD_COLOR,
        opacity=DEFAULT_HEAD_OPACITY,
    )
    att_geom.append(head_mesh)

    ensure(
        len(list(att_geom)) >= 2,
        "Club body must have at least 2 attached visual geometries",
    )
    return tree
