"""Rotation-order contracts for native rigid transforms."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.native_transforms import rigid_transform
from src.shared.python.motion_matching.native_transforms import bind_solid_ports


def fixture(axes: str) -> dict[str, Any]:
    values = {
        "RotationMethod": "RotationSequence",
        "RotationSequence": "XYZ",
        "RotationSequenceAxes": axes,
        "RotationSequenceAnglesUnits": "deg",
        "TranslationMethod": "None",
    }
    return {
        "parameters": [
            *[{"name": name, "expression": value} for name, value in values.items()],
            {
                "name": "RotationSequenceAngles",
                "resolved_numeric": True,
                "numeric_value": [90, 0, 90],
            },
        ]
    }


def test_base_and_follower_axes_have_different_rotation_order() -> None:
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(
        rigid_transform(fixture("FollowerAxes"))[:3, :3], rx @ rz, atol=1e-14
    )
    np.testing.assert_allclose(
        rigid_transform(fixture("BaseAxes"))[:3, :3], rz @ rx, atol=1e-14
    )


def test_rejects_unknown_axes_and_unresolved_angles() -> None:
    with pytest.raises(ValueError, match="axes"):
        rigid_transform(fixture("Guess"))
    block = fixture("BaseAxes")
    block["parameters"][-1]["resolved_numeric"] = False
    with pytest.raises(ValueError, match="RotationSequenceAngles"):
        rigid_transform(block)


def test_native_port_pose_measurements_bind_custom_frame_names() -> None:
    path = (
        Path(__file__).parents[2]
        / "fixtures/motion_matching/native_left_upper_arm_ports.json"
    )
    probe = json.loads(path.read_text())
    assert bind_solid_ports(probe) == {"LConn1": ("Frame1",), "RConn1": ("Frame2",)}
    port = next(v for v in probe["measurements"] if v["kind"] == "physical_port")
    port["translation_m"][0] += 0.001
    with pytest.raises(ValueError, match="No measured frame"):
        bind_solid_ports(probe)


def test_cartesian_translation_uses_si_and_ignores_inactive_angles() -> None:
    block = fixture("BaseAxes")
    p = {item["name"]: item for item in block["parameters"]}
    p["RotationMethod"]["expression"] = "None"
    p["RotationSequenceAngles"]["resolved_numeric"] = False
    p["TranslationMethod"]["expression"] = "Cartesian"
    block["parameters"].extend(
        [
            {
                "name": "TranslationCartesianOffset",
                "resolved_numeric": True,
                "numeric_value": [1, 2, 3],
            },
            {"name": "TranslationCartesianOffsetUnits", "expression": "cm"},
        ]
    )
    np.testing.assert_allclose(rigid_transform(block)[:3, 3], [0.01, 0.02, 0.03])
