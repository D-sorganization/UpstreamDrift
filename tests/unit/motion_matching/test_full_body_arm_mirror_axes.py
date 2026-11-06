"""Left/right arm joint axes are sagittal mirrors at the zero pose (G1, #14746).

A rotation about axis ``a`` reflects through the sagittal plane
``M = diag(1, -1, 1)`` into a rotation about ``-M @ a`` (rotations are
pseudovectors). So a positive left coordinate and the same positive right
coordinate must give mirrored motion: ``axis_R == -M @ axis_L``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPECS = ["full_body_spec_anthro_driver.json", "full_body_spec_anthro_driver_v2.json"]
MIRROR = np.diag([1.0, -1.0, 1.0])
AXIS_COL = {"Rx": 0, "Ry": 1, "Rz": 2}
CHAIN = (
    ("Scapula", "Hubto{s}S"),
    ("Shoulder", "{s}UpperArm"),
    ("Elbow", None),
    ("Forearm", "{s}LowerForearm"),
    ("Wrist", None),
)


def _joint_axes(spec: dict, side: str) -> dict[str, np.ndarray]:
    """Axis of every arm coordinate in the torso frame at the zero pose."""
    joints = {j["name"]: j for j in spec["joints"]}
    word = "Left" if side == "L" else "Right"
    frame = np.eye(3)
    axes: dict[str, np.ndarray] = {}
    for key in ("Scapula", "Shoulder", "Elbow", "Forearm", "Wrist"):
        joint = next(j for n, j in joints.items() if f"{word} {key}" in n)
        pb = np.asarray(joint["parent_to_base"], float)[:3, :3]
        cf = np.asarray(joint["child_to_follower"], float)[:3, :3]
        base = frame @ pb
        for prim in joint["primitives"]:
            axes[prim["coordinate"]] = base[:, AXIS_COL[prim["primitive"]]]
        frame = base @ np.linalg.inv(cf)
    return axes


@pytest.mark.parametrize("spec_name", SPECS)
def test_right_arm_axes_mirror_left(spec_name: str) -> None:
    path = MODELS / spec_name
    if not path.exists():
        pytest.skip(f"{spec_name} not present")
    spec = json.loads(path.read_text(encoding="utf-8"))
    left, right = _joint_axes(spec, "L"), _joint_axes(spec, "R")
    bad = []
    for name, axis in left.items():
        other = right["R" + name[1:]]
        if not np.allclose(other, -MIRROR @ axis, atol=1e-9):
            bad.append((name, np.round(axis, 3).tolist(), np.round(other, 3).tolist()))
    assert not bad, f"non-mirrored right-arm coordinates (L axis, R axis): {bad}"
