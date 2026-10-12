"""Canonical golfer initial pose must rest on the floor (Issue #11818).

Contract: at the canonical initial pose the lowest foot geom touches z = 0
within ``FLOOR_TOLERANCE_M`` (2 mm, the tolerance stated in the issue).
The feet must neither penetrate the floor nor hover above it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

mujoco = pytest.importorskip("mujoco")

from tools.model_converter.build_models import (  # noqa: E402
    DEFAULT_CANONICAL_YAML,
    DEFAULT_MJCF_OUT,
    build_models,
)

FLOOR_TOLERANCE_M = 0.002
FOOT_GEOM_NAMES = (
    "left_foot_geom",
    "left_toes_geom",
    "right_foot_geom",
    "right_toes_geom",
)


def _lowest_foot_z(xml_path: Path) -> float:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lowest = float("inf")
    for name in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"canonical model is missing {name}"
        centre_z = float(data.geom_xpos[gid][2])
        rot = data.geom_xmat[gid].reshape(3, 3)
        half = model.geom_size[gid]
        # Box/capsule vertical extent along world z.
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            extent = float(abs(rot[2]) @ half)
        else:
            extent = float(half[0] + half[1] * abs(rot[2][2]))
        lowest = min(lowest, centre_z - extent)
    return lowest


def test_generated_canonical_mjcf_feet_rest_on_floor(tmp_path: Path) -> None:
    out = tmp_path / "golfer.xml"
    rc = build_models(
        canonical_yaml=DEFAULT_CANONICAL_YAML,
        urdf_out=tmp_path / "golfer.urdf",
        mjcf_out=out,
        matlab_out=tmp_path / "InitModelParameters.m",
    )
    assert rc == 0
    assert abs(_lowest_foot_z(out)) <= FLOOR_TOLERANCE_M


def test_checked_in_mjcf_feet_rest_on_floor() -> None:
    assert abs(_lowest_foot_z(DEFAULT_MJCF_OUT)) <= FLOOR_TOLERANCE_M


def test_checked_in_mjcf_has_no_floor_penetration() -> None:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF_OUT))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    dists = [c.dist for c in data.contact[: data.ncon] if floor in (c.geom1, c.geom2)]
    assert all(d >= -FLOOR_TOLERANCE_M for d in dists)
