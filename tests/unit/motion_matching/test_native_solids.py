"""Native solid conversion: units, active inertia settings and frame axes."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.native_solids import solid_properties


def cylinder() -> dict:
    values = {
        "InertiaType": "CalculateFromGeometry",
        "BasedOnType": "Mass",
        "Mass": 2.0,
        "MassUnits": "kg",
        "CylinderRadius": 0.5,
        "CylinderRadiusUnits": "m",
        "CylinderLength": 2.0,
        "CylinderLengthUnits": "m",
        "SerializedFrames": "",
    }
    return {
        "path": "m/body",
        "library_reference": "sm_lib/Body Elements/Cylindrical Solid",
        "parameters": [
            {
                "name": name,
                "expression": value,
                "resolved_numeric": isinstance(value, float),
                "numeric_value": value if isinstance(value, float) else [],
            }
            for name, value in values.items()
        ],
    }


def test_mass_and_inertia_use_active_enum_not_resolved_enum_value() -> None:
    block = cylinder()
    enum = next(p for p in block["parameters"] if p["name"] == "BasedOnType")
    enum.update(resolved_numeric=True, numeric_value=123.0)
    body = solid_properties(block)
    assert body.mass_kg == 2.0
    np.testing.assert_allclose(
        body.inertia_com_kg_m2, np.diag([19 / 24, 19 / 24, 0.25])
    )
    np.testing.assert_array_equal(body.com_m, np.zeros(3))


def test_converts_units_and_rejects_unresolved_active_parameter() -> None:
    block = cylinder()
    p = {v["name"]: v for v in block["parameters"]}
    p["MassUnits"]["expression"] = "lbm"
    assert solid_properties(block).mass_kg == pytest.approx(0.90718474)
    p["CylinderRadius"]["resolved_numeric"] = False
    with pytest.raises(ValueError, match="CylinderRadius"):
        solid_properties(block)


def test_custom_frame_handedness_and_origin() -> None:
    block = cylinder()
    frame = next(p for p in block["parameters"] if p["name"] == "SerializedFrames")
    frame["expression"] = """<Frames><Frame><Id>Frame1</Id><Value><Name>top</Name>
    <Origin><Source>GeometricFeature</Source><FeatureName>top curve</FeatureName></Origin>
    <PrimaryAxis><DefinedDirection>-X</DefinedDirection><Source>ReferenceFrame</Source><SourceDirection>+Z</SourceDirection></PrimaryAxis>
    <SecondaryAxis><DefinedDirection>+Y</DefinedDirection><Source>ReferenceFrame</Source><SourceDirection>+X</SourceDirection></SecondaryAxis>
    </Value></Frame></Frames>"""
    transform = solid_properties(block).frames["Frame1"]
    np.testing.assert_array_equal(transform[:3, 3], [0, 0, 1])
    np.testing.assert_array_equal(
        transform[:3, :3], [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    )
    assert np.linalg.det(transform[:3, :3]) == 1


def test_sphere_uses_density_and_rejects_unknown_inertia_mode() -> None:
    block = cylinder()
    block["library_reference"] = "sm_lib/Body Elements/Spherical Solid"
    p = {v["name"]: v for v in block["parameters"]}
    p["BasedOnType"]["expression"] = "Density"
    block["parameters"].extend(
        [
            {"name": "SphereRadius", "resolved_numeric": True, "numeric_value": 0.5},
            {"name": "SphereRadiusUnits", "expression": "m"},
            {"name": "Density", "resolved_numeric": True, "numeric_value": 6 / np.pi},
            {"name": "DensityUnits", "expression": "kg/m^3"},
        ]
    )
    body = solid_properties(block)
    assert body.mass_kg == pytest.approx(1.0)
    np.testing.assert_allclose(body.inertia_com_kg_m2, np.eye(3) * 0.1)
    p["InertiaType"]["expression"] = "Custom"
    with pytest.raises(ValueError, match="inertia mode"):
        solid_properties(block)


def test_matches_native_r2025b_custom_frame_fixture() -> None:
    path = (
        Path(__file__).parents[2]
        / "fixtures/motion_matching/native_left_upper_arm_reference.json"
    )
    fixture = json.loads(path.read_text())
    assert fixture["matlab_release"] == "2025b" and fixture["native_exit_code"] == 0
    transform = solid_properties(fixture["block"]).frames["Frame1"]
    np.testing.assert_allclose(transform[:3, 3], fixture["translation_m"], atol=1e-10)
    np.testing.assert_allclose(
        transform[:3, :3], fixture["rotation_matrix"], atol=1e-10
    )
