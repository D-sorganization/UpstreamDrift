"""Unit tests for motion_pipeline.model_bridge (epic #8390, B2/#8397)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.model_bridge import (
    rig_root_link_name,
    rig_to_urdf,
)


def _rig() -> SkeletonRig:
    joints = {
        "hip": JointDef(
            name="hip",
            parent=None,
            children=["knee"],
            tpose_offset=[0.0, 0.0, 1.0],
            axes=["X", "Y"],
            limits=[
                JointLimit(lower=-1.0, upper=1.0),
                JointLimit(lower=-0.5, upper=0.5),
            ],
        ),
        "knee": JointDef(
            name="knee",
            parent="hip",
            children=[],
            tpose_offset=[0.0, 0.0, -0.4],
            axes=["X"],
        ),
    }
    return SkeletonRig(id="test_rig", joints=joints, root_joint="hip")


def test_urdf_parses_and_counts_one_joint_per_dof() -> None:
    rig = _rig()
    root = ET.fromstring(rig_to_urdf(rig))
    assert root.tag == "robot"
    revolute = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    assert len(revolute) == rig.num_dofs == 3


def test_urdf_joint_order_matches_rig_dof_order() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    names = [j.get("name") for j in root.findall("joint")]
    assert names == ["hip_dof0", "hip_dof1", "knee_dof0"]


def test_urdf_applies_limits_and_default_limits() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    limits = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    assert float(limits["hip_dof0"].get("lower")) == -1.0
    assert float(limits["hip_dof1"].get("upper")) == 0.5
    # knee has no explicit limit -> defaults to +/- pi
    assert float(limits["knee_dof0"].get("lower")) == pytest.approx(-3.14159, abs=1e-3)


def test_urdf_offset_applied_on_first_dof_only() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    origins = {
        j.get("name"): j.find("origin").get("xyz") for j in root.findall("joint")
    }
    assert origins["hip_dof0"] == "0.0 0.0 1.0"
    assert origins["hip_dof1"] == "0.0 0.0 0.0"


def test_urdf_has_transmission_per_dof() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    assert len(root.findall("transmission")) == 3


def test_root_link_name_matches_helper() -> None:
    rig = _rig()
    root = ET.fromstring(rig_to_urdf(rig))
    link_names = {ln.get("name") for ln in root.findall("link")}
    assert rig_root_link_name(rig) in link_names


def test_zero_dof_rig_rejected() -> None:
    # SkeletonRig contracts require joints, so simulate emptiness via a
    # minimal object exposing the accessed attributes.
    class _FakeRig:
        num_dofs = 0
        id = "empty"
        joints: dict = {}

    with pytest.raises(ValueError, match="at least one DOF"):
        rig_to_urdf(_FakeRig())  # type: ignore[arg-type]
