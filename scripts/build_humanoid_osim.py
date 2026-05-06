"""Build the golf-humanoid `.osim` from the Rajagopal2015 OpenSense base.

This script is the deterministic generator for
``src/engines/physics_engines/opensim/models/golf_humanoid.osim`` (issue #4110,
OpenSim Parity Spec §3.2).

Source base
-----------
``shared/models/opensim/opensim-models/Models/Rajagopal_OpenSense/``
``Rajagopal2015_opensense.osim``

The OpenSense variant is derived from Rajagopal et al. (2016) and is **already
muscle-stripped** — its ``ForceSet`` is empty. Using the OpenSense variant lets
us skip the muscle-removal pass entirely, which keeps the build deterministic
without depending on the OpenSim Python bindings (which are not pip-installable
on every platform — see PR body for details).

Modifications applied
---------------------
1. Rename ``Model name="OpenSense_Subject"`` → ``Model name="golf_humanoid"``.
2. Append a ``Club`` rigid body to the ``BodySet`` with golf-club mass +
   inertia (driver-class).
3. Append a ``WeldJoint`` named ``hand_r_to_club`` whose parent is the
   ``hand_r`` body and whose child is the ``Club`` body. The grip frame and
   clubhead frame are exposed as ``PhysicalOffsetFrame`` components on the
   ``Club`` body.
4. Add a ``CoordinateActuator`` to the ``ForceSet`` for every ``Coordinate``
   in the model. Naming convention: ``tau_<coord_name>``. ``optimal_force=1``,
   ``min_control=-Inf``, ``max_control=+Inf`` (we control torque directly in
   N·m via the polynomial controller — see OPENSIM_PARITY_SPEC §3.2 step 4).

The script emits **byte-identical** output across runs (no timestamps, no
non-deterministic iteration order) so the committed artifact is reproducible.

Usage
-----
::

    python3 scripts/build_humanoid_osim.py

Writes ``src/engines/physics_engines/opensim/models/golf_humanoid.osim``.

Notes
-----
- We deliberately **do not** import ``opensim`` here. The Rajagopal OpenSense
  XML is already a valid ``OpenSimDocument Version="40000"`` document; pure-XML
  manipulation keeps the builder runnable in any environment that has Python
  3.10+. The committed model is then validated by
  ``tests/test_opensim_model_loads.py`` whenever the OpenSim Python bindings
  are available.
- The Simscape body-chain coordinate naming alignment (cross-engine spec §2.6)
  is documented in ``models/README.md`` — the OpenSim coordinate names used
  here are the canonical Rajagopal names that the Simscape→OpenSim
  ``coordinate_map`` (issue ``OPENSIM-COORD-MAP``) translates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

# Repository root, derived from this script's location.
REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_OSIM = (
    REPO_ROOT
    / "shared"
    / "models"
    / "opensim"
    / "opensim-models"
    / "Models"
    / "Rajagopal_OpenSense"
    / "Rajagopal2015_opensense.osim"
)

OUTPUT_OSIM = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid.osim"
)

# Driver-class golf club mass / inertia.
# Mass per USGA/R&A: a typical driver head + shaft assembly is ~0.32 kg.
# The principal moment of inertia about the grip axis is dominated by the
# 1.14 m shaft length: I ≈ m·L²/3 ≈ 0.32 · 1.14² / 3 ≈ 0.139 kg·m². The
# transverse moments are a fraction of that (rod about its end). These
# numbers are MVP placeholders sized to keep the integrator stable — the
# canonical golf-club anthropometric YAML (issue ``PARITY-DIMENSIONS``)
# will replace them once that lands.
CLUB_MASS_KG = 0.32
CLUB_LENGTH_M = 1.14
CLUB_IXX = 0.139
CLUB_IYY = 0.0001
CLUB_IZZ = 0.139
CLUB_MASS_CENTER_M = (0.0, -CLUB_LENGTH_M / 2.0, 0.0)

# Grip offset on the right hand (meters). The hand_r origin is the wrist; the
# grip sits ~6 cm distally along the hand's local +y. This is an anatomical
# placeholder; the canonical value is set by ``coordinate_map`` once that
# issue lands.
HAND_R_GRIP_OFFSET = (0.0, -0.06, 0.0)


def _parse(path: Path) -> ET.ElementTree:
    """Parse an XML file preserving the original declaration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Base OSIM not found: {path}. "
            "Run `git submodule update --init shared/models/opensim/opensim-models`."
        )
    return ET.parse(path)


def _find_one(parent: ET.Element, tag: str) -> ET.Element:
    """Find a single child by tag; raise if missing or duplicated."""
    matches = parent.findall(tag)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one <{tag}> under <{parent.tag}>, found {len(matches)}."
        )
    return matches[0]


def _make_club_body() -> ET.Element:
    """Build the <Body name="Club"> element."""
    body = ET.Element("Body", attrib={"name": "Club"})

    # Frame geometry (axes display).
    fg = ET.SubElement(body, "FrameGeometry", attrib={"name": "frame_geometry"})
    ET.SubElement(fg, "socket_frame").text = ".."
    ET.SubElement(
        fg, "scale_factors"
    ).text = "0.20000000000000001 0.20000000000000001 0.20000000000000001"

    ET.SubElement(body, "attached_geometry")

    wos = ET.SubElement(body, "WrapObjectSet", attrib={"name": "wrapobjectset"})
    ET.SubElement(wos, "objects")
    ET.SubElement(wos, "groups")

    ET.SubElement(body, "mass").text = repr(CLUB_MASS_KG)
    ET.SubElement(body, "mass_center").text = " ".join(
        repr(v) for v in CLUB_MASS_CENTER_M
    )
    # Inertia: [Ixx Iyy Izz Ixy Ixz Iyz] about the mass center.
    ET.SubElement(body, "inertia").text = " ".join(
        repr(v) for v in (CLUB_IXX, CLUB_IYY, CLUB_IZZ, 0.0, 0.0, 0.0)
    )
    return body


def _make_offset_frame(
    name: str,
    socket_parent: str,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ET.Element:
    """Build a <PhysicalOffsetFrame> element."""
    frame = ET.Element("PhysicalOffsetFrame", attrib={"name": name})
    fg = ET.SubElement(frame, "FrameGeometry", attrib={"name": "frame_geometry"})
    ET.SubElement(fg, "socket_frame").text = ".."
    ET.SubElement(
        fg, "scale_factors"
    ).text = "0.20000000000000001 0.20000000000000001 0.20000000000000001"
    ET.SubElement(frame, "socket_parent").text = socket_parent
    ET.SubElement(frame, "translation").text = " ".join(repr(v) for v in translation)
    ET.SubElement(frame, "orientation").text = " ".join(repr(v) for v in orientation)
    return frame


def _make_club_weld_joint() -> ET.Element:
    """Build the <WeldJoint name="hand_r_to_club"> element.

    Per OPENSIM_PARITY_SPEC §3.4 we use a ``WeldJoint`` (Option A) rather
    than a ``WeldConstraint``: same rigid-attachment semantics, zero added
    DOFs, faster integration. The ``Club`` body is rigidly fixed to a frame
    on ``hand_r`` (the grip frame). The clubhead frame is exposed as a
    PhysicalOffsetFrame on the Club body itself (issue says the clubhead
    frame is on the ``Club`` body, not the joint frames).
    """
    joint = ET.Element("WeldJoint", attrib={"name": "hand_r_to_club"})
    ET.SubElement(joint, "socket_parent_frame").text = "hand_r_grip_offset"
    ET.SubElement(joint, "socket_child_frame").text = "club_grip_offset"
    frames = ET.SubElement(joint, "frames")

    # Parent-side offset: a frame on hand_r at the grip location.
    frames.append(
        _make_offset_frame(
            "hand_r_grip_offset",
            socket_parent="/bodyset/hand_r",
            translation=HAND_R_GRIP_OFFSET,
        )
    )
    # Child-side offset: the Club body's grip end.
    frames.append(
        _make_offset_frame(
            "club_grip_offset",
            socket_parent="/bodyset/Club",
            translation=(0.0, 0.0, 0.0),
        )
    )
    # Clubhead frame: distal end of the club. The cross-engine spec requires
    # extracting clubhead position via FK from this frame.
    frames.append(
        _make_offset_frame(
            "club_head_offset",
            socket_parent="/bodyset/Club",
            translation=(0.0, -CLUB_LENGTH_M, 0.0),
        )
    )
    return joint


def _make_coordinate_actuator(coord_name: str) -> ET.Element:
    """Build a <CoordinateActuator> for a single coordinate."""
    act = ET.Element("CoordinateActuator", attrib={"name": f"tau_{coord_name}"})
    ET.SubElement(act, "appliesForce").text = "true"
    ET.SubElement(act, "min_control").text = "-Inf"
    ET.SubElement(act, "max_control").text = "Inf"
    ET.SubElement(act, "coordinate").text = coord_name
    ET.SubElement(act, "optimal_force").text = "1"
    return act


def _collect_coordinate_names(model: ET.Element) -> list[str]:
    """Return coordinate names in document order (deterministic)."""
    jointset = _find_one(model, "JointSet")
    objects = _find_one(jointset, "objects")
    names: list[str] = []
    for joint in objects:
        coords = joint.find("coordinates")
        if coords is None:
            continue
        for coord in coords.findall("Coordinate"):
            cname = coord.get("name")
            if cname is None:
                continue
            names.append(cname)
    return names


def _indent(elem: ET.Element, level: int = 0, *, tab: str = "\t") -> None:
    """Pretty-print indenter that matches the Rajagopal source style (tabs)."""
    pad = "\n" + tab * level
    child_pad = pad + tab
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_pad
        for i, child in enumerate(elem):
            _indent(child, level + 1, tab=tab)
            if i < len(elem) - 1:
                if not child.tail or not child.tail.strip():
                    child.tail = child_pad
            else:
                if not child.tail or not child.tail.strip():
                    child.tail = pad
    else:
        if level and (elem.tail is None or not elem.tail.strip()):
            elem.tail = pad


def build() -> Path:
    """Build the golf_humanoid.osim file. Returns the output path."""
    tree = _parse(BASE_OSIM)
    root = tree.getroot()
    if root.tag != "OpenSimDocument":
        raise ValueError(f"Unexpected root element: {root.tag}")

    model = _find_one(root, "Model")
    model.set("name", "golf_humanoid")

    # ------------------------------------------------------------------
    # 1. Append the Club body to BodySet/objects.
    # ------------------------------------------------------------------
    bodyset = _find_one(model, "BodySet")
    body_objects = _find_one(bodyset, "objects")
    body_objects.append(_make_club_body())

    # ------------------------------------------------------------------
    # 2. Append the WeldJoint linking hand_r → Club.
    # ------------------------------------------------------------------
    jointset = _find_one(model, "JointSet")
    joint_objects = _find_one(jointset, "objects")
    joint_objects.append(_make_club_weld_joint())

    # ------------------------------------------------------------------
    # 3. Add a CoordinateActuator for every coordinate in the model.
    # ------------------------------------------------------------------
    coord_names = _collect_coordinate_names(model)
    if not coord_names:
        raise ValueError("No coordinates found in the base model — aborting.")
    forceset = _find_one(model, "ForceSet")
    force_objects = _find_one(forceset, "objects")
    for cname in coord_names:
        force_objects.append(_make_coordinate_actuator(cname))

    # Pretty-print using tabs to match the upstream style.
    _indent(root)

    # Emit with the same XML declaration the upstream uses.
    OUTPUT_OSIM.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    declaration = b'<?xml version="1.0" encoding="UTF-8" ?>\n'
    OUTPUT_OSIM.write_bytes(declaration + xml_bytes + b"\n")

    return OUTPUT_OSIM


def _summary(coord_names: list[str], output: Path) -> str:
    size_kb = output.stat().st_size / 1024.0
    return (
        f"Wrote {output.relative_to(REPO_ROOT)} ({size_kb:.1f} KiB) "
        f"with {len(coord_names)} coordinate actuators."
    )


def main() -> int:
    output = build()
    coord_names = _collect_coordinate_names(_parse(output).getroot().find("Model"))
    sys.stdout.write(_summary(coord_names, output) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
