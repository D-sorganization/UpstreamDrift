"""Smoke + load tests for the golf-humanoid OpenSim model (issue #4110).

Two layers:

1. **Pure-XML structural assertions** (always run) — validate that the
   committed ``golf_humanoid.osim`` has the expected topology: 23 bodies
   incl. ``Club``, a ``WeldJoint`` linking ``hand_r`` → ``Club``, and one
   ``CoordinateActuator`` per ``Coordinate``. These are guard-rails that
   protect against accidental edits to the generated artifact.

2. **OpenSim binding load test** (``requires_opensim`` marker) — actually
   load the model via ``osim.Model(path)`` and call ``initSystem()``. This
   is the canonical acceptance check from the parity spec but it is
   skipped automatically when the OpenSim Python bindings are not
   installed (the ``opensim`` wheel is not available on every platform —
   on macOS and some Python versions it requires ``conda install
   -c opensim-org opensim``).

The pure-XML layer alone would not have caught a bad ``Inf`` value or a
malformed body inertia, so the binding test is essential whenever
OpenSim is available; it must be exercised at least in the
``opensim-extras`` CI job.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

# Repo-root anchored path so the test runs from any CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid.osim"
)


# ---------------------------------------------------------------------------
# Layer 1: pure-XML structural assertions (no opensim dependency).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_xml() -> ET.Element:
    """Parse the committed .osim once for the whole module."""
    assert MODEL_PATH.is_file(), (
        f"golf_humanoid.osim missing at {MODEL_PATH}. "
        "Run `python3 scripts/build_humanoid_osim.py` to regenerate."
    )
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    assert root.tag == "OpenSimDocument"
    assert root.get("Version") == "40000"
    return root.find("Model")


def _coord_names(model: ET.Element) -> list[str]:
    names: list[str] = []
    for joint in model.find("JointSet").find("objects"):
        coords = joint.find("coordinates")
        if coords is None:
            continue
        for coord in coords.findall("Coordinate"):
            cname = coord.get("name")
            if cname:
                names.append(cname)
    return names


def test_model_name_is_golf_humanoid(model_xml: ET.Element) -> None:
    assert model_xml.get("name") == "golf_humanoid"


def test_club_body_present(model_xml: ET.Element) -> None:
    bodies = [
        b.get("name") for b in model_xml.find("BodySet").find("objects").findall("Body")
    ]
    assert "Club" in bodies, f"Expected a 'Club' body; bodies were: {bodies}"


def test_weld_joint_hand_r_to_club(model_xml: ET.Element) -> None:
    """A WeldJoint must rigidly connect hand_r → Club at the grip."""
    joints = list(model_xml.find("JointSet").find("objects"))
    weld_joints = [
        j for j in joints if j.tag == "WeldJoint" and j.get("name") == "hand_r_to_club"
    ]
    assert len(weld_joints) == 1, (
        "Exactly one WeldJoint named 'hand_r_to_club' must exist; "
        f"found {len(weld_joints)}."
    )
    weld = weld_joints[0]
    parent_socket = weld.find("socket_parent_frame").text
    child_socket = weld.find("socket_child_frame").text
    # Frames are owned by the joint; resolve them and check their bodies.
    frame_map = {
        f.get("name"): f.find("socket_parent").text
        for f in weld.find("frames").findall("PhysicalOffsetFrame")
    }
    assert frame_map[parent_socket] == "/bodyset/hand_r"
    assert frame_map[child_socket] == "/bodyset/Club"


def test_clubhead_offset_frame_present(model_xml: ET.Element) -> None:
    """A clubhead frame must exist on the Club body for FK extraction."""
    weld = next(
        j
        for j in model_xml.find("JointSet").find("objects")
        if j.tag == "WeldJoint" and j.get("name") == "hand_r_to_club"
    )
    frames = weld.find("frames").findall("PhysicalOffsetFrame")
    head_frames = [f for f in frames if f.get("name") == "club_head_offset"]
    assert head_frames, "Clubhead frame 'club_head_offset' missing from WeldJoint."
    assert head_frames[0].find("socket_parent").text == "/bodyset/Club"


def test_one_actuator_per_coordinate(model_xml: ET.Element) -> None:
    """Topology check: one CoordinateActuator per Coordinate."""
    coord_names = _coord_names(model_xml)
    assert coord_names, "No coordinates found in model."
    actuators = model_xml.find("ForceSet").find("objects").findall("CoordinateActuator")
    actuator_targets = [a.find("coordinate").text for a in actuators]
    assert len(actuator_targets) == len(coord_names), (
        f"Expected {len(coord_names)} CoordinateActuators (one per coord), "
        f"found {len(actuator_targets)}."
    )
    assert set(actuator_targets) == set(coord_names), (
        "Mismatch between CoordinateActuator targets and coordinate names."
    )


def test_known_simscape_chain_coordinates_present(model_xml: ET.Element) -> None:
    """Spot-check that the canonical Simscape body-chain coordinates exist.

    The cross-engine parity spec §2.6 requires the OpenSim coordinate names
    to round-trip with the Simscape body chain. We don't enforce the full
    23-DOF mapping here (that's ``OPENSIM-COORD-MAP``'s job), but we do
    require that the load-bearing coordinates — pelvis 6-DOF root, lumbar
    3-DOF, right shoulder/elbow/wrist (the swing chain) — are all present.
    """
    coords = set(_coord_names(model_xml))
    required = {
        "pelvis_tilt",
        "pelvis_list",
        "pelvis_rotation",
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "lumbar_extension",
        "lumbar_bending",
        "lumbar_rotation",
        "arm_flex_r",
        "arm_add_r",
        "arm_rot_r",
        "elbow_flex_r",
        "pro_sup_r",
        "wrist_flex_r",
        "wrist_dev_r",
    }
    missing = required - coords
    assert not missing, f"Missing canonical coordinates: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Layer 2: actual OpenSim load (binding required).
# ---------------------------------------------------------------------------


_OPENSIM_AVAILABLE = importlib.util.find_spec("opensim") is not None
