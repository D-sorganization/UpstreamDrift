"""Unit tests for OpenSim full-body OSIM exporter (MS-40 #10339).

Verifies pure-XML generation of 44-coordinate full-body models from
anthropometric specifications without requiring OpenSim SDK bindings.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import defusedxml.ElementTree as ET

import pytest

pytestmark = [pytest.mark.unit]


def _parse_osim_xml(xml_str: str) -> ET.Element:
    """Parse OpenSim XML string, escaping double colons for XML 1.0 parser compliance."""
    return ET.fromstring(xml_str.replace("::", "__"))


ROOT = Path(__file__).resolve().parents[3]
DRIVER_SPEC_PATH = (
    ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_driver.json"
)
IRON_SPEC_PATH = (
    ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_iron7.json"
)


@pytest.fixture(scope="module")
def driver_spec() -> dict:
    if not DRIVER_SPEC_PATH.exists():
        pytest.fail(f"Driver specification not found: {DRIVER_SPEC_PATH}")
    return json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def iron_spec() -> dict:
    if not IRON_SPEC_PATH.exists():
        pytest.fail(f"Iron specification not found: {IRON_SPEC_PATH}")
    return json.loads(IRON_SPEC_PATH.read_text(encoding="utf-8"))


def test_export_full_body_osim_driver_structure(driver_spec: dict) -> None:
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    spec_bytes = json.dumps(driver_spec).encode("utf-8")
    xml_str, metadata = export_full_body_osim(spec_bytes)

    assert metadata["representation"] == "native-full-body-osim-v1"
    assert metadata["model_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert (
        metadata["osim_sha256"] == hashlib.sha256(xml_str.encode("utf-8")).hexdigest()
    )
    assert len(metadata["coordinate_order"]) == 44
    assert metadata["coordinate_order"] == driver_spec["coordinate_order"]

    root = _parse_osim_xml(xml_str)
    assert root.tag == "OpenSimDocument"
    assert root.get("Version") == "40000"

    model = root.find("Model")
    assert model is not None
    assert model.get("name") == "full_body_anthro_driver"

    # Coordinates: exactly 44 coordinates matching document order and none locked
    jointset = model.find("JointSet")
    assert jointset is not None
    joint_objects = jointset.find("objects")
    assert joint_objects is not None

    coordinates: list[str] = []
    for joint in joint_objects:
        assert joint.tag == "CustomJoint"
        coords_elem = joint.find("coordinates")
        if coords_elem is not None:
            for coord in coords_elem.findall("Coordinate"):
                cname = coord.get("name")
                assert cname is not None
                coordinates.append(cname)
                locked = coord.find("locked")
                assert locked is not None and locked.text == "false"

    assert len(coordinates) == 44
    assert coordinates == driver_spec["coordinate_order"]

    # Bodies & Inertias
    bodyset = model.find("BodySet")
    assert bodyset is not None
    body_objects = bodyset.find("objects")
    assert body_objects is not None
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        clean_osim_body_name,
    )

    body_names = [b.get("name") for b in body_objects.findall("Body")]
    assert len(body_names) == 24
    for b in driver_spec["bodies"]:
        if b["name"] != "world":
            assert clean_osim_body_name(b["name"]) in body_names

    # Closure constraints in ConstraintSet
    constraintset = model.find("ConstraintSet")
    assert constraintset is not None
    constraint_objects = constraintset.find("objects")
    assert constraint_objects is not None
    constraints = list(constraint_objects)
    assert len(constraints) >= 1
    # Either WeldConstraint or PointConstraint pair
    closure_tags = {c.tag for c in constraints}
    assert ("WeldConstraint" in closure_tags) or ("PointConstraint" in closure_tags)

    # Contact forces in ForceSet
    forceset = model.find("ForceSet")
    assert forceset is not None
    force_objects = forceset.find("objects")
    assert force_objects is not None
    hc_forces = force_objects.findall("HuntCrossleyForce")
    assert len(hc_forces) >= 4  # At least 4 foot spheres (or 6 with toes)

    # Coordinate actuators: tau_<coord>
    actuators = force_objects.findall("CoordinateActuator")
    assert len(actuators) >= 38  # 38 internal or 44 all
    actuator_coords = [
        a.find("coordinate").text for a in actuators if a.find("coordinate") is not None
    ]
    for coord in driver_spec["coordinate_order"][6:]:
        assert coord in actuator_coords

    # Contact geometries in ContactGeometrySet: ground half-space + spheres
    contactset = model.find("ContactGeometrySet")
    assert contactset is not None
    contact_objects = contactset.find("objects")
    assert contact_objects is not None
    halfspaces = contact_objects.findall("ContactHalfSpace")
    assert len(halfspaces) >= 1
    spheres = contact_objects.findall("ContactSphere")
    assert len(spheres) >= 4

    # Markers in MarkerSet: exactly 34 tour markers
    markerset = model.find("MarkerSet")
    assert markerset is not None
    marker_objects = markerset.find("objects")
    assert marker_objects is not None
    markers = marker_objects.findall("Marker")
    assert len(markers) == 34
    marker_names = {m.get("name") for m in markers}
    for label in driver_spec["marker_attachments"]:
        assert label in marker_names


def test_export_full_body_osim_point_pair_closure(driver_spec: dict) -> None:
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    xml_str, _ = export_full_body_osim(driver_spec, closure_type="point_pair")
    root = _parse_osim_xml(xml_str)
    constraintset = root.find(".//ConstraintSet/objects")
    assert constraintset is not None
    point_constraints = constraintset.findall("PointConstraint")
    assert len(point_constraints) == 2


def test_export_full_body_osim_iron_structure(iron_spec: dict) -> None:
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    xml_str, metadata = export_full_body_osim(
        iron_spec, model_name="full_body_anthro_iron7"
    )
    assert metadata["representation"] == "native-full-body-osim-v1"
    assert len(metadata["coordinate_order"]) == 44

    root = _parse_osim_xml(xml_str)
    model = root.find("Model")
    assert model is not None
    assert model.get("name") == "full_body_anthro_iron7"


def test_cli_export_full_body_osim(tmp_path: Path) -> None:
    out_file = tmp_path / "test_model.osim"
    receipt_file = tmp_path / "test_receipt.json"

    cmd = [
        sys.executable,
        "-m",
        "src.engines.physics_engines.opensim.python.full_body_osim",
        "--spec",
        str(DRIVER_SPEC_PATH),
        "--out",
        str(out_file),
        "--receipt",
        str(receipt_file),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert out_file.exists()
    assert receipt_file.exists()

    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["representation"] == "native-full-body-osim-v1"
    assert receipt["coordinate_count"] == 44
    assert len(receipt["coordinate_order"]) == 44
