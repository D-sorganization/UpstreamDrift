"""Parameterized golf club visual geometry and grip offset frames (OG-03, #10397).

Provides parameterized visual geometry attachments and coordinate frames for
the OpenSim Club body according to shared ``ClubSpec`` parameters:
1. Inspects whether a model has visual geometry on the Club body.
2. Attaches the generated shaft, grip and head STL meshes (``club_visuals``,
   shared with the full-body models) without altering physical inertia or mass.
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


from src.engines.physics_engines.opensim.python import club_visuals
from src.shared.python.contracts import ensure, require
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.motion_matching.club_models import (
    DRIVER,
    IRON_7,
    ClubSpec,
)

VISUAL_FRAME_NAME: str = "club_visual_frame"

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


def club_assembly_for(spec: ClubSpec) -> ClubAssembly:
    """The shared visual assembly for a ``motion_matching`` club spec."""
    return ClubAssembly(
        head_alias=spec.name,
        length_m=spec.length_m,
        grip_length_m=spec.grip_length_m,
        shaft_radius_m=spec.shaft_radius_m,
    )


def _visual_frame(club: ClubAssembly) -> ET.Element:
    """Offset frame mapping the shared club frame onto the Club body frame.

    The shared meshes sit in the head-origin frame (shaft toward the grip along
    -y, shaft axis at z = ``axis_offset_m``); this body has its origin at the
    butt with the head at -length_m along y. The map is a half turn about x
    plus the translation (0, -length, axis_offset).
    """
    frame = ET.Element("PhysicalOffsetFrame", attrib={"name": VISUAL_FRAME_NAME})
    ET.SubElement(frame, "socket_parent").text = ".."
    ET.SubElement(
        frame, "translation"
    ).text = f"0 {-club.length_m:.17g} {club.axis_offset_m:.17g}"
    ET.SubElement(frame, "orientation").text = f"{math.pi:.17g} 0 0"
    return frame


def attach_visual_club(
    tree: ET.ElementTree,
    spec: ClubSpec | None = None,
    *,
    mesh_dir_ref: str = "",
    geometry_dir: Path | str = club_visuals.GEOMETRY_DIR,
) -> ET.ElementTree:
    """Attach the shared shaft, grip and head meshes to the Club body.

    Replaces the old ``.vtp`` placeholders (never committed) with the generated
    STLs of ``club_visuals``. Mass, mass centre and inertia are untouched.

    Preconditions:
    - tree must contain a Body with name="Club".
    - the club's STLs exist in ``geometry_dir`` (committed for the driver and
      7-iron; write others with ``club_visuals.write_club_assets``).

    Postconditions:
    - Club attached_geometry holds shaft, grip and head meshes.
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
    club = club_assembly_for(club_spec)
    missing = [
        str(p)
        for p in club_visuals.asset_paths(club.head_alias, Path(geometry_dir)).values()
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Club visual assets missing: {', '.join(missing)}")

    components = club_body.find("components")
    if components is None:
        components = ET.SubElement(club_body, "components")
    for old in components.findall(f"PhysicalOffsetFrame[@name='{VISUAL_FRAME_NAME}']"):
        components.remove(old)
    components.append(_visual_frame(club))

    att_geom = club_body.find("attached_geometry")
    if att_geom is None:
        att_geom = ET.SubElement(club_body, "attached_geometry")
    for child in list(att_geom):
        att_geom.remove(child)
    att_geom.extend(
        club_visuals.club_mesh_elements(
            club.head_alias, mesh_dir_ref, socket_frame=f"../{VISUAL_FRAME_NAME}"
        )
    )
    ensure(
        len(list(att_geom)) >= 3,
        "Club body must have shaft, grip and head visual geometries",
    )
    return tree
