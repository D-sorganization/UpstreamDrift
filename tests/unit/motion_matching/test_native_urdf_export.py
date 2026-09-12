"""Native export preserves physical bodies and mandatory closure metadata."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.shared.python.motion_matching.native_urdf import export_native_urdf

pytestmark = pytest.mark.unit


@pytest.fixture
def spec() -> dict:
    eye = np.eye(4).tolist()
    return {
        "schema_version": 1,
        "gravity_m_s2": [0, 0, -9.81],
        "coordinate_order": ["angle"],
        "bodies": [
            {"name": "world", "solids": []},
            {
                "name": "body",
                "solids": [
                    {
                        "name": "solid",
                        "mass_kg": 2.0,
                        "com_m": [0.1, 0, 0],
                        "inertia_com_kg_m2": np.diag([0.01, 0.02, 0.03]).tolist(),
                        "placement": eye,
                    }
                ],
            },
        ],
        "joints": [
            {
                "name": "hinge",
                "parent": "world",
                "child": "body",
                "parent_to_base": eye,
                "child_to_follower": eye,
                "primitives": [{"primitive": "Rz", "coordinate": "angle"}],
            }
        ],
        "frames": [{"name": "marker", "body": "body", "placement": eye}],
        "closure": {
            "name": "weld",
            "body_a": "body",
            "body_b": "world",
            "placement_a": eye,
            "placement_b": eye,
        },
    }


def test_massless_primitives_and_explicit_sidecar(spec: dict) -> None:
    xml, sidecar = export_native_urdf(json.dumps(spec).encode())
    root = ET.fromstring(xml)
    masses = [float(x.attrib["value"]) for x in root.findall("link/inertial/mass")]
    assert sum(masses) == 2 and sum(m > 0 for m in masses) == 1
    assert root.find("joint[@name='angle']").attrib["type"] == "revolute"
    assert float(root.find("joint[@name='angle']/dynamics").attrib["damping"]) == 0
    assert sidecar["closure"] == spec["closure"]
    assert sidecar["coordinate_order"] == ["angle"]
    assert sidecar["requires_sidecar"] is True
    assert sidecar["limit_semantics"] == "restore-unbounded-before-dynamics"
    assert "marker" in sidecar["frame_links"]


def test_missing_closure_rejected(spec: dict) -> None:
    del spec["closure"]
    with pytest.raises(ValueError, match="closure"):
        export_native_urdf(json.dumps(spec).encode())


def test_lost_coordinate_rejected(spec: dict) -> None:
    spec["coordinate_order"].append("missing")
    with pytest.raises(ValueError, match="coordinate"):
        export_native_urdf(json.dumps(spec).encode())


def test_invalid_transform_rejected(spec: dict) -> None:
    spec["joints"][0]["parent_to_base"][0][0] = 2
    with pytest.raises(ValueError, match="transform"):
        export_native_urdf(json.dumps(spec).encode())
